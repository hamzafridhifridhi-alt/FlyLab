# FlyLab

A live whole-brain *Drosophila* connectome simulation with an embodied fly, in the browser.

The left panel is the fly itself, simulated in MuJoCo physics. The right panel is
its brain: all 138,639 neurons of the FlyWire adult female brain, each point
brightening as that neuron fires. The two are wired together, so a stimulus lights
up a pathway in the brain and the fly changes what it does.

```
stimulus -> 138,639 LIF neurons / 15.1M synapses -> descending drive -> CPG -> legs -> MuJoCo
                        |                                                          |
                   glowing brain view                                    3D physics view
```

---

## Quick start

Four commands from a fresh clone to a running simulation:

```bash
cd ~/Project/flylab
./scripts/bootstrap_data.sh                     # fetch connectome + Three.js (~420 MB)
uv sync                                         # create .venv and install deps
uv run python src/flylab/build_connectome.py    # build simulation artifacts (~2 s)
uv run flylab                                   # serve http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000>. First boot takes roughly 15 seconds while Numba
compiles the network kernel and MuJoCo compiles the fly model; the terminal prints
`Application startup complete.` when the simulation is actually running.

Only the first two steps ever need repeating, and only after a fresh clone.
Afterwards `uv run flylab` is the single command to start.

### Requirements

| | |
|---|---|
| OS | macOS (Apple silicon verified) or Linux |
| Python | 3.13+, managed by [uv](https://docs.astral.sh/uv/) |
| Tools | `git`, `curl` |
| Disk | ~1.2 GB total (420 MB sources, 120 MB artifacts, ~600 MB venv) |
| RAM | ~2.5 GB resident while running |
| Network | Needed once, for `bootstrap_data.sh` and `uv sync` |

No GPU, no Docker, no cloud account. If `uv` is missing:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

### What each step does

**`./scripts/bootstrap_data.sh`** downloads the three things too large for git:
the Shiu et al. brain model including `Connectivity_783.parquet` (~380 MB), the
FlyWire neuron annotations (~30 MB), and the pinned Three.js build (~1.3 MB).
Re-running it skips whatever is already there, so an interrupted download is fixed
by running it again.

**`build_connectome.py`** turns those sources into what the simulator actually
reads, in `data/derived/`:

| File | Size | Purpose |
|---|---|---|
| `edges.npz` | 122 MB | CSR adjacency for the 15.1M synapses |
| `neurons.npz` | 2.1 MB | positions, cell types, transmitters, sides |
| `brain_points.bin` | 2.8 MB | Float32 buffer the browser loads once |
| `meta.json` | 1 KB | label vocabularies and region counts |

Expected output ends with `done: 138,639 neurons, 15,091,983 edges`. Different
numbers mean a truncated download; delete `data/` and re-bootstrap.

### Options

```bash
uv run flylab --port 9000     # different port
uv run flylab --reload        # auto-reload on source edits
uv run flylab --host 0.0.0.0  # expose to the network (prints a warning, see Security)
```

### Verify the install

With the server running, in a second terminal:

```bash
uv run python scripts/verify_stack.py
```

This drives every preset through the real API, asserts the binary WebSocket frames
are internally consistent, and checks that lesioning the descending neurons stops
the fly. It ends with `ALL CHECKS PASSED`.

---

## Using it

**Experiment presets** each drive a real neuron population:

| Preset | Drives | Typical result |
|---|---|---|
| Rest | nothing | spontaneous dynamics only, fly stands |
| Looming visual stimulus | 3,000 optic-lobe neurons | optic lobes light up at ~6 Hz |
| Sugar on the proboscis | 800 sensory afferents | sensory to central brain, ~25 Hz |
| Wind on the antennae | 400 mechanosensory afferents | mild steering bias |
| Descending walk command | 400 descending neurons | ~50 Hz, fly walks fast |
| Left / right turn command | 300 one-sided descending neurons | fly turns that way |

**Lesion a region** by clicking its name in the region indicators. Silencing the
descending neurons drops the fly from roughly 50 mm/s to a standstill, which is the
clearest evidence the coupling is causal rather than decorative. `clear lesions`
restores everything.

**Click any neuron** in the 3D brain to see its cell type, transmitter, out-degree,
live firing rate, and membrane voltage, and to stimulate it directly.

**Colour modes** switch the point cloud between anatomical region, neurotransmitter,
and activity-only heat. **Simulation speed** trades realtime factor against how much
model time passes per wall-clock second.

**In the brain view:** drag to orbit, scroll to zoom, click a neuron to inspect it.
Firing neurons flash bright and fade over about a quarter second, and the glow
shrinks as you zoom in so a close-up is not swamped by its neighbours’ bloom.

| Key | Action |
|---|---|
| `r` | reframe the brain at the default distance |
| `d` | toggle the render diagnostics overlay (camera distance, glow scale) |

---

## What is actually being simulated

| Layer | Source | Notes |
|---|---|---|
| Connectome | [FlyWire FAFB v783](https://codex.flywire.ai/) (Dorkenwald et al. 2024) | 138,639 neurons, 15,091,983 weighted edges |
| Neuron model | [Shiu et al. 2024, *Nature*](https://doi.org/10.1038/s41586-024-07763-9) | leaky integrate-and-fire, published constants |
| Annotations | [Schlegel et al. 2024](https://github.com/flyconnectome/flywire_annotations) | soma coordinates, cell classes, transmitters |
| Body | [NeuroMechFly v2 / FlyGym 2.1](https://github.com/NeLy-EPFL/flygym) (Wang-Chen et al. 2024) | full biomechanical fly in MuJoCo |

The brain reuses the reference Brian2 model’s equations and constants, rewritten as
a fused Numba kernel. The upstream implementation runs one batch trial and writes a
dataframe; this needs to run continuously and stream state, which is a different
execution model, not a different model.

### Honest limits

The connectome does not produce joint-level motor commands, so the brain-to-body
coupling is deliberately explicit: left/right **descending neuron firing rates** set
the walking gain and turn bias of a tripod CPG, which generates the gait. This is the
same abstraction the NeuroMechFly authors use for their own connectome-driven demos.
What the fly does is downstream of real connectome dynamics, but the leg trajectories
are CPG-generated, not connectome-derived.

Region groupings (optic lobe, sensory, descending, ...) are coarse folds of FlyWire
`super_class` labels, not anatomical neuropil boundaries. The 14 neurons with no soma
coordinate are jittered near the centroid so the point count stays exact.

---

## Performance

Measured on an M5 MacBook Pro, 32 GB, macOS 26.5:

| Metric | Value |
|---|---|
| Brain, driven | 0.23-0.35x realtime (~4,000 steps/s at dt = 0.2 ms) |
| Brain, idle | ~1.7x realtime |
| Body physics | ~0.8x realtime (dt = 0.2 ms) |
| Camera render | 18 ms/frame at 980x720 |
| Frame payload | ~32 KB JPEG, ~0.6 MB/s at 20 fps |
| UI stream | 27-31 Hz |
| Startup | ~15 s (Numba JIT + MuJoCo compile) |

Everything runs on CPU. The brain is the bottleneck, not the graphics: MuJoCo
rendering costs about the same at 480p as at 980p, which is why the fly view is
rendered large.

---

## Architecture

```
src/flylab/
  build_connectome.py   FlyWire parquet/TSV -> CSR edges + point buffer
  brain.py              fused-Numba LIF network over 15.1M synapses
  body.py               NeuroMechFly + tripod CPG, descending-drive controlled
  engine.py             one thread owning both sims, publishing snapshots
  server.py             FastAPI: REST control + binary WebSocket stream
web/
  index.html            layout: fly view left, brain top right
  src/app.js            Three.js point cloud, glow shader, WebSocket decode
  src/style.css
  vendor/               pinned Three.js 0.169 (fetched by bootstrap)
scripts/
  bootstrap_data.sh     fetch connectome, annotations, Three.js
  verify_stack.py       end-to-end check against a running server
```

Two decisions worth knowing before changing anything:

**One simulation thread owns both sims.** On macOS a MuJoCo OpenGL context belongs
to the thread that created it, and rendering from another thread blocks forever. The
body is therefore constructed inside the simulation thread, frames are encoded there,
and control calls are queued onto it. The web layer only reads finished snapshots.

**The WebSocket speaks binary.** Per-neuron activity is 138,639 bytes per tick; as
JSON that is roughly 40x larger and stalls the main thread. Each frame is
`uint32 header_len | JSON header | uint8[n] activity | int32[k] spikes | JPEG`.

### HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | neuron counts, regions, presets, model parameters |
| `GET /api/points` | Float32 position buffer for the 3D brain |
| `GET /api/neuron/{i}` | one neuron’s type, transmitter, rate, voltage |
| `GET /api/frame` | latest camera frame as JPEG |
| `POST /api/preset` | apply a named stimulus |
| `POST /api/stimulate` | drive specific FlyWire root IDs |
| `POST /api/silence` | lesion or restore a region |
| `POST /api/pause`, `/api/reset`, `/api/speed` | run control |
| `WS /ws` | live activity, spikes, body state, camera frames |

---

## Troubleshooting

**`run build_connectome first`** in the browser, or a blank brain panel: the
artifacts are missing. Run `uv run python src/flylab/build_connectome.py`.

**Blank brain panel with the fly view working:** Three.js did not load. Confirm
`web/vendor/three.module.js` exists and that `web/vendor/OrbitControls.js` imports
`/vendor/three.module.js` rather than bare `three`. Re-running the bootstrap script
fixes both.

**`body unavailable, running brain only`** in the server log: MuJoCo could not open
a GL context. The brain and all indicators still work; the camera panel stays empty.
On headless Linux, install EGL or run under `xvfb-run`.

**Port already in use:** `uv run flylab --port 9000`.

**Speed reads 0.00x and nothing moves:** the simulation is paused. Click `resume`.

---

## Security

The server binds to `127.0.0.1` and has **no authentication**. Any client that can
reach the port can start, stop, reset, and lesion the simulation. That is fine for a
local research tool, but `--host 0.0.0.0` exposes simulation control to your whole
network, so put a reverse proxy with auth in front of it first.

---

## Citations

For anything publishable, cite the upstream work rather than this wrapper:

- Dorkenwald et al. (2024) *Neuronal wiring diagram of an adult brain.* Nature 634, 124-138.
- Schlegel et al. (2024) *Whole-brain annotation and multi-connectome cell typing of Drosophila.* Nature 634, 139-152.
- Shiu et al. (2024) *A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain.* Nature 634, 210-219.
- Wang-Chen et al. (2024) *NeuroMechFly v2: simulating embodied sensorimotor control in adult Drosophila.* Nature Methods 21, 2353-2362.

FlyGym is Apache-2.0; FlyWire data is CC-BY-4.0. The Shiu et al. model carries its
own repository licence. This project only orchestrates them.
