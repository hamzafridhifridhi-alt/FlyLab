"""End-to-end check against a freshly started server on the default port."""
import asyncio, json, struct, time, urllib.request
import websockets

BASE = "http://127.0.0.1:8000"

def get(p):
    with urllib.request.urlopen(BASE + p, timeout=15) as r:
        return json.load(r)

def post(p, d):
    req = urllib.request.Request(BASE + p, data=json.dumps(d).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)

async def latest(ws, window=0.6):
    end = time.monotonic() + window
    h = None
    while time.monotonic() < end:
        try:
            buf = await asyncio.wait_for(ws.recv(), timeout=max(0.05, end - time.monotonic()))
        except asyncio.TimeoutError:
            break
        hl = struct.unpack_from("<I", buf, 0)[0]
        h = json.loads(buf[4:4+hl])
        assert len(buf) == 4 + hl + h["n_activity"] + h["n_spikes"]*4 + h["n_jpeg"]
    return h

async def main():
    m = get("/api/meta")
    print(f"meta: {m['n_neurons']:,} neurons, {m['n_edges']:,} edges, body={m['has_body']}")
    assert m["n_neurons"] == 138639 and m["n_edges"] == 15091983
    async with websockets.connect("ws://127.0.0.1:8000/ws", max_size=64*1024*1024) as ws:
        results = {}
        for name in ["rest", "visual_looming", "sugar_taste", "walk_command", "turn_left"]:
            post("/api/preset", {"name": name})
            await asyncio.sleep(2.6)
            h = await latest(ws)
            top = max(h["region_rates"], key=lambda k: h["region_rates"][k])
            results[name] = (round(h["realtime_factor"],2), h["body"]["behavior"],
                             h["body"]["speed_mm_s"], top, round(h["region_rates"][top],1),
                             h["n_jpeg"])
            print(f"  {name:15s} rt={results[name][0]}x  {results[name][1]:13s} "
                  f"{results[name][2]:6.1f} mm/s  peak={top}@{results[name][4]}Hz  jpeg={h['n_jpeg']}B")
        # Causality: lesioning descending output must stop the fly.
        post("/api/preset", {"name": "walk_command"}); await asyncio.sleep(3.0)
        before = (await latest(ws))["body"]
        post("/api/silence", {"region": "descending", "silenced": True}); await asyncio.sleep(4.0)
        after = (await latest(ws))["body"]
        print(f"  lesion: {before['speed_mm_s']:.1f} -> {after['speed_mm_s']:.1f} mm/s "
              f"({before['behavior']} -> {after['behavior']})")
        assert after["speed_mm_s"] < before["speed_mm_s"] / 3, "lesion had no effect"
        post("/api/silence/clear", {})
        # Neuron inspect + direct stimulation
        n = get("/api/neuron/1523")
        print(f"  neuron 1523: {n['cell_type'] or 'unnamed'} {n['region']}/{n['neurotransmitter']} "
              f"out={n['out_degree']}")
        print("  stimulate:", post("/api/stimulate", {"root_ids": [int(n["root_id"])], "rate_hz": 200}))
        post("/api/preset", {"name": "rest"})
    print("ALL CHECKS PASSED")

asyncio.run(main())
