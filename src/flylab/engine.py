"""Simulation engine: runs the brain and the body in one background thread.

One thread owns both simulations so there is no lock contention on the hot path,
and the web layer only ever reads immutable snapshots. The loop keeps brain and
body on a shared clock: every cycle advances the brain by ``brain_chunk_ms`` of
model time and the body by the same amount of physical time, then republishes a
snapshot for any connected browser.

On macOS a MuJoCo/OpenGL context belongs to the thread that created it: calling
render from a different thread blocks forever. So the body is constructed inside
the simulation thread, and camera frames are encoded to JPEG there as well. The
web layer only ever receives finished bytes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .body import FlyBody
from .brain import BrainSimulation, LIFParams


# Named stimulus presets. Each maps to a selector over the annotation metadata,
# so the UI can offer meaningful experiments instead of raw neuron IDs.
STIM_PRESETS: dict[str, dict[str, Any]] = {
    "rest": {
        "label": "Rest",
        "description": "No external drive. Only spontaneous connectome dynamics.",
        "region": None,
        "count": 0,
        "rate": 0.0,
    },
    "visual_looming": {
        "label": "Looming visual stimulus",
        "description": "Drive optic-lobe neurons, as when something approaches fast.",
        "region": "optic_lobe",
        "count": 3000,
        "rate": 150.0,
    },
    "sugar_taste": {
        "label": "Sugar on the proboscis",
        "description": "Drive gustatory/sensory afferents, the classic feeding trigger.",
        "region": "sensory",
        "count": 800,
        "rate": 180.0,
    },
    "antennal_wind": {
        "label": "Wind on the antennae",
        "description": "Mechanosensory drive that normally biases walking direction.",
        "region": "sensory",
        "count": 400,
        "rate": 150.0,
        "offset": 4000,
    },
    "walk_command": {
        "label": "Descending walk command",
        "description": "Drive descending neurons directly: the brain telling the legs to go.",
        "region": "descending",
        "count": 400,
        "rate": 160.0,
    },
    "turn_left": {
        "label": "Left turn command",
        "description": "Drive left-side descending neurons only, steering the fly.",
        "region": "descending",
        "count": 300,
        "rate": 170.0,
        "side": "left",
    },
    "turn_right": {
        "label": "Right turn command",
        "description": "Drive right-side descending neurons only.",
        "region": "descending",
        "count": 300,
        "rate": 170.0,
        "side": "right",
    },
}


@dataclass
class EngineConfig:
    # 10 ms of model time per cycle is the knee of the curve on Apple silicon:
    # it holds a ~30 Hz UI while keeping the brain near a third of realtime.
    brain_chunk_ms: float = 10.0
    target_hz: float = 30.0
    activity_ceiling_hz: float = 45.0
    max_spike_sample: int = 4000
    enable_body: bool = True


@dataclass
class Snapshot:
    """Immutable view of the world, safe to hand to any number of readers."""

    seq: int = 0
    brain_time_ms: float = 0.0
    wall_hz: float = 0.0
    realtime_factor: float = 0.0
    population_rate_hz: float = 0.0
    region_rates: dict[str, float] = field(default_factory=dict)
    spike_count: int = 0
    activity: np.ndarray | None = None
    spike_sample: np.ndarray | None = None
    body: dict[str, Any] = field(default_factory=dict)
    stimulus: str = "rest"
    silenced_count: int = 0
    frame_jpeg: bytes | None = None


class SimulationEngine:
    """Owns the brain, the body, and the thread that drives them."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.brain = BrainSimulation(LIFParams())
        self.body: FlyBody | None = None

        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._paused = threading.Event()
        self._ready = threading.Event()
        self._seq = 0
        self._current_stim = "rest"
        self._speed_scale = 1.0
        self._commands: deque = deque()
        self._frame_jpeg: bytes | None = None
        self._frame_seq = 0

        self._descending = self.brain.region_indices("descending")
        side_codes = self.brain.meta["sides"]
        left_code = side_codes.index("left") if "left" in side_codes else 0
        right_code = side_codes.index("right") if "right" in side_codes else 1
        dn_side = self.brain.side[self._descending]
        self._dn_left = self._descending[dn_side == left_code]
        self._dn_right = self._descending[dn_side == right_code]
        self._rate_to_hz = 1000.0 / self.brain.p.dt

    # ------------------------------------------------------------------ control
    def _submit(self, fn) -> None:
        """Queue work to run on the simulation thread.

        Body operations touch the GL context, so they must not run on a request
        handler thread. Brain-only operations are queued too, to keep every
        mutation ordered relative to the integration steps.
        """
        if self._thread is None:
            fn()
            return
        self._commands.append(fn)

    def _drain_commands(self) -> None:
        while self._commands:
            try:
                self._commands.popleft()()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[flylab] command failed: {exc}")

    def apply_preset(self, name: str) -> None:
        preset = STIM_PRESETS.get(name)
        if preset is None:
            raise KeyError(name)
        self._submit(lambda: self._apply_preset_now(name, preset))

    def _apply_preset_now(self, name: str, preset: dict[str, Any]) -> None:
        self.brain.clear_stimulus()
        self._current_stim = name
        region = preset.get("region")
        if not region or not preset.get("count"):
            return
        idx = self.brain.region_indices(region)
        side = preset.get("side")
        if side:
            codes = self.brain.meta["sides"]
            if side in codes:
                idx = idx[self.brain.side[idx] == codes.index(side)]
        offset = int(preset.get("offset", 0))
        if offset:
            idx = idx[offset:]
        count = int(preset["count"])
        if idx.size > count:
            # Deterministic even sampling keeps runs comparable between sessions.
            idx = idx[np.linspace(0, idx.size - 1, count).astype(np.int64)]
        self.brain.set_stimulus(idx, float(preset["rate"]))

    def stimulate_neurons(self, root_ids: list[int], rate_hz: float) -> int:
        idx = [i for i in (self.brain.index_of(r) for r in root_ids) if i is not None]
        self._submit(lambda: self.brain.set_stimulus(idx, rate_hz))
        return len(idx)

    def silence_region(self, region: str, silenced: bool = True) -> int:
        idx = self.brain.region_indices(region)
        self._submit(lambda: self.brain.set_silenced(idx, silenced))
        return int(idx.size)

    def clear_silenced(self) -> None:
        self._submit(self.brain.clear_silenced)

    def reset(self) -> None:
        def _do() -> None:
            self.brain.reset()
            if self.body is not None:
                self.body.reset()

        self._submit(_do)

    def set_speed_scale(self, scale: float) -> None:
        self._speed_scale = float(np.clip(scale, 0.1, 4.0))

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # --------------------------------------------------------------------- loop
    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="flylab-sim",
                                        daemon=True)
        self._thread.start()
        # Wait for the body/GL setup to finish so the first request sees a world.
        self._ready.wait(timeout=60.0)

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self.body is not None:
            self.body.close()

    def _loop(self) -> None:
        cfg = self.config
        # The body must be built here: its GL context is bound to this thread.
        if cfg.enable_body and self.body is None:
            try:
                self.body = FlyBody(timestep=2e-4)
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(f"[flylab] body unavailable, running brain only: {exc}")
                self.body = None
        self._ready.set()

        target_dt = 1.0 / cfg.target_hz
        brain_steps = max(1, int(round(cfg.brain_chunk_ms / self.brain.p.dt)))
        last = time.perf_counter()
        ema_hz = cfg.target_hz
        frame_every = max(1, int(round(cfg.target_hz / 20.0)))
        cycle = 0

        while self._running.is_set():
            cycle_start = time.perf_counter()
            if self._paused.is_set():
                # Still apply queued control changes so the UI stays responsive
                # while the integration is halted.
                self._drain_commands()
                self._publish_paused_snapshot()
                time.sleep(0.05)
                last = time.perf_counter()
                continue

            # Apply queued mutations before stepping, so the snapshot published
            # at the end of this cycle already reflects them.
            self._drain_commands()
            steps = max(1, int(brain_steps * self._speed_scale))
            spikes_total = 0
            sample: list[np.ndarray] = []
            for _ in range(steps):
                spiked = self.brain.step()
                spikes_total += spiked.size
                if spiked.size and len(sample) < 8:
                    sample.append(spiked.copy())

            # Descending drive -> body. Rates are per-neuron means in Hz.
            if self.body is not None:
                left = right = 0.0
                # A lesioned neuron still spikes, but its output never reaches
                # the ventral nerve cord, so it must not drive the legs either.
                if self._dn_left.size:
                    live = self._dn_left[~self.brain.silenced_mask[self._dn_left]]
                    if live.size:
                        left = float(self.brain.rate_ewma[live].sum()
                                     / self._dn_left.size * self._rate_to_hz)
                if self._dn_right.size:
                    live = self._dn_right[~self.brain.silenced_mask[self._dn_right]]
                    if live.size:
                        right = float(self.brain.rate_ewma[live].sum()
                                      / self._dn_right.size * self._rate_to_hz)
                self.body.set_descending_drive(left, right)
                body_steps = int(round(steps * self.brain.p.dt * 1e-3
                                       / self.body.timestep))
                self.body.step(max(1, body_steps))

            now = time.perf_counter()
            elapsed = now - last
            last = now
            if elapsed > 0:
                ema_hz = 0.85 * ema_hz + 0.15 * (1.0 / elapsed)
            model_ms = steps * self.brain.p.dt

            spike_sample = None
            if sample:
                joined = np.concatenate(sample)
                if joined.size > cfg.max_spike_sample:
                    joined = joined[: cfg.max_spike_sample]
                spike_sample = joined.astype(np.int32)

            # Encode the camera view on this thread, where the GL context lives.
            if self.body is not None and cycle % frame_every == 0:
                try:
                    self._frame_jpeg = self._encode_frame()
                except Exception:
                    self._frame_jpeg = None
            cycle += 1

            snap = Snapshot(
                seq=self._seq,
                brain_time_ms=self.brain.sim_time_ms,
                wall_hz=ema_hz,
                realtime_factor=(model_ms / 1000.0) / max(1e-9, elapsed),
                population_rate_hz=self.brain.population_rate_hz(),
                region_rates=self.brain.region_rates_hz(),
                spike_count=spikes_total,
                activity=self.brain.activity_u8(cfg.activity_ceiling_hz),
                spike_sample=spike_sample,
                body=self._body_payload(),
                stimulus=self._current_stim,
                silenced_count=int(self.brain.silenced_mask.sum()),
                frame_jpeg=self._frame_jpeg,
            )
            self._seq += 1
            with self._lock:
                self._snapshot = snap

            sleep = target_dt - (time.perf_counter() - cycle_start)
            if sleep > 0:
                time.sleep(sleep)

    def _body_payload(self) -> dict[str, Any]:
        if self.body is None:
            return {}
        s = self.body.state
        return {
            "position": [round(v, 3) for v in s.position],
            "heading_deg": round(s.heading_deg, 1),
            "speed_mm_s": round(s.speed_mm_s, 2),
            "distance_mm": round(s.distance_mm, 2),
            "turn_bias": round(s.turn_bias, 3),
            "drive": round(s.drive, 3),
            "contacts": s.contacts,
            "leg_phases": [round(p, 3) for p in s.leg_phases],
            "behavior": s.behavior,
            "sim_time_s": round(s.sim_time_s, 3),
        }

    def _publish_paused_snapshot(self) -> None:
        """Refresh counters while paused so control feedback is not stuck."""
        with self._lock:
            prev = self._snapshot
        snap = Snapshot(
            seq=self._seq,
            brain_time_ms=self.brain.sim_time_ms,
            wall_hz=prev.wall_hz,
            realtime_factor=0.0,
            population_rate_hz=self.brain.population_rate_hz(),
            region_rates=self.brain.region_rates_hz(),
            spike_count=0,
            activity=prev.activity,
            spike_sample=None,
            body=self._body_payload(),
            stimulus=self._current_stim,
            silenced_count=int(self.brain.silenced_mask.sum()),
            frame_jpeg=self._frame_jpeg,
        )
        self._seq += 1
        with self._lock:
            self._snapshot = snap

    # ----------------------------------------------------------------- snapshot
    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def _encode_frame(self, quality: int = 78) -> bytes | None:
        """Encode the body camera view as JPEG. Must run on the sim thread."""
        if self.body is None:
            return None
        img = self.body.latest_frame()
        if img is None:
            return None
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def frame_jpeg(self) -> bytes | None:
        """Latest camera frame, encoded by the simulation thread."""
        return self._frame_jpeg
