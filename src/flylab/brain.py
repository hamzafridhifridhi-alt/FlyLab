"""Whole-brain leaky integrate-and-fire engine over the FlyWire FAFB v783 connectome.

The neuron model, constants, and synaptic scaling follow Shiu et al. 2024 (Nature)
and the reference Brian2 implementation in philshiu/Drosophila_brain_model. The
difference here is the execution model: instead of running one batch trial and
writing a dataframe, this runs continuously in a background thread so a browser
can watch the network live.

Neuron (Kakaria & de Bivort 2017; Jurgensen et al. 2021):

    tau_m dv/dt = (v_rest - v) + g
    tau_s dg/dt = -g
    spike when v > v_th  ->  v := v_reset, g := 0, refractory for t_rfc

Synapse: a presynaptic spike adds w_syn * (excitatory x connectivity) to the
postsynaptic conductance after a 1.8 ms delay. Inhibitory neurons already carry a
negative sign in the connectome table, so no extra sign handling is needed.

The per-step update is a single fused Numba kernel: iterating 138k neurons and
scattering into 15M CSR edges from Python costs more in NumPy temporaries than in
actual arithmetic, so fusing the integrate/threshold/propagate passes into one
loop is what makes realtime streaming possible on a laptop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"


@dataclass(frozen=True)
class LIFParams:
    """Constants from the Shiu et al. 2024 model. Voltages in mV, times in ms."""

    v_rest: float = -52.0
    v_reset: float = -52.0
    v_th: float = -45.0
    tau_m: float = 20.0
    tau_s: float = 5.0
    t_rfc: float = 2.2
    t_delay: float = 1.8
    w_syn: float = 0.275
    f_poi: int = 250
    r_poi: float = 150.0
    dt: float = 0.2


@njit(cache=True, fastmath=True, nogil=True)
def _fused_step(
    v, g, refrac_until, ring, ring_pos, delay_steps,
    indptr, indices, weight, rate_ewma, silenced_mask,
    stim_idx, poisson_draw,
    v_rest, v_reset, v_th, decay_v, decay_g, t_rfc,
    sim_time_ms, kick, rate_decay, spike_out,
):
    """One step: deliver, stimulate, integrate, threshold, propagate.

    Returns the spike count, with indices written into spike_out.
    """
    n = v.shape[0]

    # Conductance whose axonal delay expires on this step.
    due = ring[ring_pos]
    for i in range(n):
        g[i] += due[i]
        due[i] = 0.0

    # Poisson stimulation drives v directly, as in the reference PoissonInput.
    for k in range(stim_idx.shape[0]):
        c = poisson_draw[k]
        if c > 0:
            v[stim_idx[k]] += kick * c

    slot = (ring_pos + delay_steps - 1) % delay_steps
    row = ring[slot]
    one_minus_dv = 1.0 - decay_v
    ns = 0

    for i in range(n):
        if sim_time_ms >= refrac_until[i]:
            vi = v_rest + (v[i] - v_rest) * decay_v + g[i] * one_minus_dv
            if vi > v_th:
                v[i] = v_reset
                g[i] = 0.0
                refrac_until[i] = sim_time_ms + t_rfc
                rate_ewma[i] = rate_ewma[i] * rate_decay + (1.0 - rate_decay)
                spike_out[ns] = i
                ns += 1
                if not silenced_mask[i]:
                    for e in range(indptr[i], indptr[i + 1]):
                        row[indices[e]] += weight[e]
            else:
                v[i] = vi
                g[i] *= decay_g
                rate_ewma[i] *= rate_decay
        else:
            # Refractory: conductance still decays, membrane stays clamped.
            g[i] *= decay_g
            rate_ewma[i] *= rate_decay

    return ns


@njit(cache=True, fastmath=True, nogil=True)
def _bin_activity(rate_ewma, region, n_regions, out):
    """Sum EWMA activity per coarse region in one pass."""
    for i in range(n_regions):
        out[i] = 0.0
    for i in range(rate_ewma.shape[0]):
        out[region[i]] += rate_ewma[i]
    return out


class BrainSimulation:
    """Continuously running whole-brain LIF network with live spike readout."""

    def __init__(
        self,
        params: LIFParams | None = None,
        *,
        derived_dir: Path = DERIVED,
    ) -> None:
        self.p = params or LIFParams()
        neurons = np.load(derived_dir / "neurons.npz", allow_pickle=True)
        edges = np.load(derived_dir / "edges.npz", allow_pickle=False)
        self.meta = json.loads((derived_dir / "meta.json").read_text())

        self.root_id: np.ndarray = neurons["root_id"]
        self.pos: np.ndarray = neurons["pos"]
        self.region: np.ndarray = neurons["region"].astype(np.int32)
        self.nt: np.ndarray = neurons["nt"]
        self.side: np.ndarray = neurons["side"]
        self.cell_type: np.ndarray = neurons["cell_type"]
        self.region_names: list[str] = list(self.meta["regions"])
        self.n = int(self.pos.shape[0])

        self.indptr: np.ndarray = edges["indptr"].astype(np.int64)
        self.indices: np.ndarray = edges["indices"].astype(np.int32)
        self.weight: np.ndarray = edges["weight"].astype(np.float32) * self.p.w_syn

        self._id_to_index = {int(r): i for i, r in enumerate(self.root_id)}
        self._region_index = {name: i for i, name in enumerate(self.region_names)}
        self._region_sizes = np.bincount(
            self.region, minlength=len(self.region_names)
        ).astype(np.float32)
        self._region_sizes[self._region_sizes == 0] = 1.0

        # State
        self.v = np.full(self.n, self.p.v_rest, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refrac_until = np.full(self.n, -1e9, dtype=np.float32)
        self.rate_ewma = np.zeros(self.n, dtype=np.float32)
        self.silenced_mask = np.zeros(self.n, dtype=np.bool_)
        self._spike_out = np.empty(self.n, dtype=np.int32)
        self._region_accum = np.zeros(len(self.region_names), dtype=np.float32)

        self.delay_steps = max(1, int(round(self.p.t_delay / self.p.dt)))
        self._ring = np.zeros((self.delay_steps, self.n), dtype=np.float32)
        self._ring_pos = 0

        self.step_count = 0
        self.sim_time_ms = 0.0

        self.stim_rates: dict[int, float] = {}
        self._stim_idx = np.zeros(0, dtype=np.int32)
        self._stim_lambda = np.zeros(0, dtype=np.float64)
        self._rng = np.random.default_rng(1234)

        self._decay_v = np.float32(np.exp(-self.p.dt / self.p.tau_m))
        self._decay_g = np.float32(np.exp(-self.p.dt / self.p.tau_s))
        self._kick = np.float32(self.p.w_syn * self.p.f_poi)

        self._rate_tau_ms = 100.0
        self._rate_decay = np.float32(np.exp(-self.p.dt / self._rate_tau_ms))
        self._rate_to_hz = 1000.0 / self.p.dt

        self.warmup()

    def warmup(self) -> None:
        """Trigger Numba compilation so the first real step is not slow."""
        _fused_step(
            self.v, self.g, self.refrac_until, self._ring, 0, self.delay_steps,
            self.indptr, self.indices, self.weight, self.rate_ewma,
            self.silenced_mask, np.zeros(0, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.float32(self.p.v_rest), np.float32(self.p.v_reset),
            np.float32(self.p.v_th), self._decay_v, self._decay_g,
            np.float32(self.p.t_rfc), np.float32(0.0), self._kick,
            self._rate_decay, self._spike_out,
        )
        _bin_activity(self.rate_ewma, self.region, len(self.region_names),
                      self._region_accum)
        self.reset()

    # ---------------------------------------------------------------- stimulus
    def index_of(self, root_id: int) -> int | None:
        return self._id_to_index.get(int(root_id))

    def region_indices(self, name: str) -> np.ndarray:
        code = self._region_index.get(name)
        if code is None:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(self.region == code)

    def _rebuild_stim(self) -> None:
        if not self.stim_rates:
            self._stim_idx = np.zeros(0, dtype=np.int32)
            self._stim_lambda = np.zeros(0, dtype=np.float64)
            return
        idx = np.fromiter(self.stim_rates.keys(), dtype=np.int32,
                          count=len(self.stim_rates))
        rates = np.fromiter(self.stim_rates.values(), dtype=np.float64,
                            count=len(self.stim_rates))
        self._stim_idx = idx
        self._stim_lambda = rates * self.p.dt / 1000.0

    def set_stimulus(self, indices, rate_hz: float | None = None) -> None:
        rate = self.p.r_poi if rate_hz is None else float(rate_hz)
        for i in np.asarray(indices, dtype=np.int64).ravel():
            i = int(i)
            if 0 <= i < self.n:
                if rate <= 0:
                    self.stim_rates.pop(i, None)
                else:
                    self.stim_rates[i] = rate
        self._rebuild_stim()

    def clear_stimulus(self) -> None:
        self.stim_rates.clear()
        self._rebuild_stim()

    def set_silenced(self, indices, silenced: bool = True) -> None:
        idx = np.asarray(indices, dtype=np.int64).ravel()
        idx = idx[(idx >= 0) & (idx < self.n)]
        self.silenced_mask[idx] = silenced

    def clear_silenced(self) -> None:
        self.silenced_mask[:] = False

    def reset(self) -> None:
        self.v[:] = self.p.v_rest
        self.g[:] = 0.0
        self.refrac_until[:] = -1e9
        self.rate_ewma[:] = 0.0
        self._ring[:] = 0.0
        self._ring_pos = 0
        self.step_count = 0
        self.sim_time_ms = 0.0

    # ------------------------------------------------------------------ update
    def step(self) -> np.ndarray:
        """Advance by one dt; returns indices of neurons that spiked."""
        if self._stim_idx.size:
            draw = self._rng.poisson(self._stim_lambda).astype(np.int32)
        else:
            draw = np.zeros(1, dtype=np.int32)

        ns = _fused_step(
            self.v, self.g, self.refrac_until, self._ring, self._ring_pos,
            self.delay_steps, self.indptr, self.indices, self.weight,
            self.rate_ewma, self.silenced_mask, self._stim_idx, draw,
            np.float32(self.p.v_rest), np.float32(self.p.v_reset),
            np.float32(self.p.v_th), self._decay_v, self._decay_g,
            np.float32(self.p.t_rfc), np.float32(self.sim_time_ms), self._kick,
            self._rate_decay, self._spike_out,
        )
        self._ring_pos = (self._ring_pos + 1) % self.delay_steps
        self.step_count += 1
        self.sim_time_ms += self.p.dt
        return self._spike_out[:ns]

    def run(self, n_steps: int) -> int:
        """Run several steps, returning the total spike count."""
        total = 0
        for _ in range(n_steps):
            total += self.step().size
        return total

    # ----------------------------------------------------------------- readout
    def region_rates_hz(self) -> dict[str, float]:
        """Mean per-neuron firing rate for each coarse region, in Hz."""
        _bin_activity(self.rate_ewma, self.region, len(self.region_names),
                      self._region_accum)
        mean = self._region_accum / self._region_sizes * self._rate_to_hz
        return {name: float(mean[i]) for i, name in enumerate(self.region_names)}

    def activity_u8(self, ceiling_hz: float = 50.0) -> np.ndarray:
        """Per-neuron activity as uint8 in [0, 255] for the WebGL glow buffer."""
        scaled = self.rate_ewma * (self._rate_to_hz * 255.0 / ceiling_hz)
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def population_rate_hz(self) -> float:
        return float(self.rate_ewma.sum() / self.n * self._rate_to_hz)
