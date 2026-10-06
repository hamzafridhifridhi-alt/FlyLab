"""FastAPI app: serves the web UI and streams simulation state over WebSocket.

Two transports, chosen by what the data looks like:

* WebSocket ``/ws`` carries a binary frame per tick. Per-neuron activity for
  138k neurons is a 138 KB uint8 block, so it goes out as raw bytes with a small
  JSON header rather than as JSON numbers.
* ``/api/*`` handles control and one-off metadata, including the neuron position
  buffer the 3D brain view loads once at startup.

The server binds to localhost only and has no authentication: it is a local
research tool that exposes simulation control to anything that can reach the
port. Do not expose it to a network without putting auth in front of it.
"""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from .engine import STIM_PRESETS, EngineConfig, SimulationEngine

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
DERIVED = ROOT / "data" / "derived"

app = FastAPI(title="FlyLab", version="0.1.0")
engine: SimulationEngine | None = None


def get_engine() -> SimulationEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not started")
    return engine


@app.on_event("startup")
async def _startup() -> None:
    global engine
    engine = SimulationEngine(EngineConfig())
    engine.start()
    engine.apply_preset("rest")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if engine is not None:
        engine.stop()


# --------------------------------------------------------------------- metadata
@app.get("/api/meta")
async def api_meta() -> dict:
    eng = get_engine()
    return {
        "n_neurons": eng.brain.n,
         "n_edges": int(eng.brain.indices.size),
        "regions": eng.brain.region_names,
        "region_counts": eng.brain.meta["region_counts"],
        "neurotransmitters": eng.brain.meta["neurotransmitters"],
        "bounds": eng.brain.meta["bounds"],
        "source": eng.brain.meta["source"],
        "presets": {
            k: {"label": v["label"], "description": v["description"]}
            for k, v in STIM_PRESETS.items()
        },
        "params": {
            "dt_ms": eng.brain.p.dt,
            "v_rest": eng.brain.p.v_rest,
            "v_th": eng.brain.p.v_th,
            "tau_m_ms": eng.brain.p.tau_m,
        },
        "has_body": eng.body is not None,
    }


@app.get("/api/points")
async def api_points() -> Response:
    """Neuron positions + labels as a raw Float32 buffer (x,y,z,region,nt)."""
    path = DERIVED / "brain_points.bin"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run build_connectome first")
    return Response(content=path.read_bytes(),
                    media_type="application/octet-stream",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/frame")
async def api_frame() -> Response:
    """Latest fly camera frame as JPEG (fallback for browsers without WS binary)."""
    eng = get_engine()
    data = eng.frame_jpeg()
    if data is None:
        raise HTTPException(status_code=404, detail="no frame yet")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------- control
class PresetRequest(BaseModel):
    name: str


class SilenceRequest(BaseModel):
    region: str
    silenced: bool = True


class StimulateRequest(BaseModel):
    root_ids: list[int] = Field(default_factory=list)
    rate_hz: float = 150.0


class SpeedRequest(BaseModel):
    scale: float = 1.0


class PauseRequest(BaseModel):
    paused: bool


@app.post("/api/preset")
async def api_preset(req: PresetRequest) -> dict:
    eng = get_engine()
    if req.name not in STIM_PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown preset {req.name}")
    eng.apply_preset(req.name)
    return {"ok": True, "preset": req.name}


@app.post("/api/silence")
async def api_silence(req: SilenceRequest) -> dict:
    eng = get_engine()
    if req.region not in eng.brain.region_names:
        raise HTTPException(status_code=400, detail=f"unknown region {req.region}")
    n = eng.silence_region(req.region, req.silenced)
    return {"ok": True, "region": req.region, "neurons": n, "silenced": req.silenced}


@app.post("/api/silence/clear")
async def api_silence_clear() -> dict:
    get_engine().clear_silenced()
    return {"ok": True}


@app.post("/api/stimulate")
async def api_stimulate(req: StimulateRequest) -> dict:
    n = get_engine().stimulate_neurons(req.root_ids, req.rate_hz)
    return {"ok": True, "matched": n}


@app.post("/api/reset")
async def api_reset() -> dict:
    get_engine().reset()
    return {"ok": True}


@app.post("/api/speed")
async def api_speed(req: SpeedRequest) -> dict:
    eng = get_engine()
    eng.set_speed_scale(req.scale)
    return {"ok": True, "scale": eng._speed_scale}


@app.post("/api/pause")
async def api_pause(req: PauseRequest) -> dict:
    eng = get_engine()
    eng.set_paused(req.paused)
    return {"ok": True, "paused": eng.paused}


@app.get("/api/neuron/{index}")
async def api_neuron(index: int) -> dict:
    """Detail for one neuron, for click-to-inspect in the 3D view."""
    eng = get_engine()
    if not 0 <= index < eng.brain.n:
        raise HTTPException(status_code=404, detail="index out of range")
    nt_names = eng.brain.meta["neurotransmitters"]
    side_names = eng.brain.meta["sides"]
    start, end = eng.brain.indptr[index], eng.brain.indptr[index + 1]
    return {
        "index": index,
        "root_id": str(eng.brain.root_id[index]),
        "cell_type": str(eng.brain.cell_type[index]),
        "region": eng.brain.region_names[int(eng.brain.region[index])],
        "neurotransmitter": nt_names[int(eng.brain.nt[index])],
        "side": side_names[int(eng.brain.side[index])],
        "out_degree": int(end - start),
        "rate_hz": round(float(eng.brain.rate_ewma[index] * 1000.0
                               / eng.brain.p.dt), 2),
        "membrane_mv": round(float(eng.brain.v[index]), 2),
        "silenced": bool(eng.brain.silenced_mask[index]),
    }


# -------------------------------------------------------------------- streaming
#  Binary frame layout, little-endian:
#    uint32 header_len | header JSON (utf-8) | uint8[n] activity | int32[k] spikes
#    | uint8[m] jpeg
def _pack(snapshot, include_frame: bool) -> bytes:
    activity = snapshot.activity
    if activity is None:
        activity = np.zeros(0, dtype=np.uint8)
    spikes = snapshot.spike_sample
    if spikes is None:
        spikes = np.zeros(0, dtype=np.int32)
    jpeg = snapshot.frame_jpeg if include_frame else None
    header = {
        "seq": snapshot.seq,
        "brain_time_ms": round(snapshot.brain_time_ms, 2),
        "wall_hz": round(snapshot.wall_hz, 2),
        "realtime_factor": round(snapshot.realtime_factor, 4),
        "population_rate_hz": round(snapshot.population_rate_hz, 3),
        "region_rates": {k: round(v, 3) for k, v in snapshot.region_rates.items()},
        "spike_count": snapshot.spike_count,
        "stimulus": snapshot.stimulus,
        "silenced_count": snapshot.silenced_count,
        "body": snapshot.body,
        "n_activity": int(activity.size),
        "n_spikes": int(spikes.size),
        "n_jpeg": int(len(jpeg) if jpeg else 0),
    }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    parts = [struct.pack("<I", len(blob)), blob, activity.tobytes(),
             spikes.tobytes()]
    if jpeg:
        parts.append(jpeg)
    return b"".join(parts)


@app.websocket("/ws")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    eng = get_engine()
    last_seq = -1
    last_frame_seq = -1
    try:
        while True:
            snap = eng.snapshot()
            if snap.seq != last_seq:
                include_frame = snap.frame_jpeg is not None and snap.seq != last_frame_seq
                await ws.send_bytes(_pack(snap, include_frame))
                last_seq = snap.seq
                if include_frame:
                    last_frame_seq = snap.seq
            await asyncio.sleep(1.0 / 32.0)
    except WebSocketDisconnect:
        return
    except (asyncio.CancelledError, RuntimeError):
        return


# --------------------------------------------------------------------- UI files
@app.get("/static/{name}")
async def static_file(name: str) -> FileResponse:
    """Serve UI assets with no-store so edits are picked up on reload."""
    safe = Path(name).name
    path = WEB / "src" / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail=safe)
    media = {
        ".js": "text/javascript",
        ".css": "text/css",
        ".map": "application/json",
    }.get(path.suffix, "application/octet-stream")
    return FileResponse(str(path), media_type=media,
                        headers={"Cache-Control": "no-store"})


@app.get("/")
async def index() -> HTMLResponse:
    path = WEB / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>FlyLab</h1><p>web/index.html is missing.</p>",
                            status_code=500)
    return HTMLResponse(path.read_text())


@app.get("/vendor/{name}")
async def vendor(name: str) -> FileResponse:
    """Serve the pinned Three.js bundle from web/vendor."""
    safe = Path(name).name
    path = WEB / "vendor" / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail=safe)
    media = "text/javascript" if safe.endswith(".js") else "application/octet-stream"
    return FileResponse(str(path), media_type=media)
