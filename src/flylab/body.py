"""Embodied Bitcraze Crazyflie 2 Drone: driven by Drosophila neural descending activity.

The fruit fly's central nervous system (CNS) descending neurons (DNs) modulate the
quadrotor thrust, orientation, and flight dynamics. Left/right descending neuron firing
rates map to drone collective thrust and differential yaw/roll steering:

    left/right descending-neuron firing rate -> thrust + turn bias (Crazyflie 2 actuators)

Rates originate from the live LIF connectome simulation (FlyLab brain engine), allowing
the Drosophila brain to directly pilot the Bitcraze Crazyflie 2 quadrotor drone in MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import mujoco

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "bitcraze_crazyflie_2" / "crazyflie.xml"


@dataclass
class BodyState:
    """Current state of the Crazyflie 2 drone piloted by the neural network."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    heading_deg: float = 0.0
    speed_mm_s: float = 0.0
    distance_mm: float = 0.0
    turn_bias: float = 0.0
    drive: float = 0.0
    contacts: list[bool] = field(default_factory=lambda: [True] * 4)
    leg_phases: list[float] = field(default_factory=lambda: [0.0] * 4)
    behavior: str = "hovering"
    sim_time_s: float = 0.0


class FlyBody:
    """MuJoCo Bitcraze Crazyflie 2 Drone wrapper controlled by descending neural drive."""

    ROTOR_NAMES = ("front_right", "rear_right", "rear_left", "front_left")

    def __init__(
        self,
        *,
        timestep: float = 1e-3,
        camera_res: tuple[int, int] = (720, 980),
        output_fps: int = 24,
        base_frequency: float = 12.0,
    ) -> None:
        self.timestep = timestep
        self.base_frequency = base_frequency

        # Load Bitcraze Crazyflie 2 model in MuJoCo
        if MODEL_PATH.exists():
            self.mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        else:
            raise FileNotFoundError(f"Crazyflie model file not found at {MODEL_PATH}")

        self.mj_model.opt.timestep = timestep
        self.mj_data = mujoco.MjData(self.mj_model)

        # Offscreen Renderer setup for tracking camera
        self.camera_res = camera_res
        self.renderer = mujoco.Renderer(self.mj_model, height=camera_res[0], width=camera_res[1])

        self.reset()

        self._prev_pos = self._get_position()
        self._distance = 0.0
        self._speed = 0.0
        self.state = BodyState()
        self._last_frame: np.ndarray | None = None
        self._rotor_angles = np.zeros(4, dtype=float)

    # ------------------------------------------------------------------ helpers
    def _get_position(self) -> np.ndarray:
        return np.asarray(self.mj_data.qpos[:3], dtype=float)

    def _heading_deg(self) -> float:
        # Extract quaternion (w, x, y, z) from qpos[3:7]
        q = self.mj_data.qpos[3:7]
        if not np.any(q):
            return 0.0
        w, x, y, z = q
        fwd_x = 1.0 - 2.0 * (y * y + z * z)
        fwd_y = 2.0 * (x * y + w * z)
        return float(np.degrees(np.arctan2(fwd_y, fwd_x)))

    # -------------------------------------------------------------------- drive
    def set_descending_drive(self, left_hz: float, right_hz: float, reference_hz: float = 12.0) -> None:
        """Translate left/right descending neural rates into Crazyflie 2 motor thrusts."""
        left = max(0.0, float(left_hz))
        right = max(0.0, float(right_hz))
        total = left + right
        ref = max(1e-6, reference_hz * 2.0)
        drive = float(np.clip(total / ref, 0.0, 1.8))
        bias = 0.0 if total < 1e-6 else float(np.clip((right - left) / total, -1.0, 1.0))

        self.state.drive = drive
        self.state.turn_bias = bias

        # Crazyflie 2 mass = 0.027 kg -> hover gravity force = 0.027 * 9.81 = 0.265 N (~0.066 N per rotor)
        hover_thrust = 0.06625

        if drive < 0.12:
            # Idle / landed state
            base_t = hover_thrust * 0.4
            self.state.behavior = "landed / idle"
        elif abs(bias) > 0.35:
            base_t = hover_thrust * (0.98 + 0.25 * drive)
            self.state.behavior = "steering right" if bias > 0 else "steering left"
        elif drive > 1.0:
            base_t = hover_thrust * (1.1 + 0.35 * drive)
            self.state.behavior = "fast autonomous flight"
        else:
            base_t = hover_thrust * (1.0 + 0.15 * drive)
            self.state.behavior = "hovering / inspecting"

        # Apply differential motor thrust for 4 rotors
        # Motors: m1 (Front-Right), m2 (Rear-Right), m3 (Rear-Left), m4 (Front-Left)
        t_fr = base_t * (1.0 - 0.3 * bias)
        t_rr = base_t * (1.0 - 0.3 * bias)
        t_rl = base_t * (1.0 + 0.3 * bias)
        t_fl = base_t * (1.0 + 0.3 * bias)

        self.mj_data.ctrl[:] = [t_fr, t_rr, t_rl, t_fl]

    # --------------------------------------------------------------------- step
    def step(self, n_steps: int = 1) -> bool:
        """Step the MuJoCo simulation of the Crazyflie 2 drone."""
        for _ in range(n_steps):
            mujoco.mj_step(self.mj_model, self.mj_data)

        pos = self._get_position()
        delta = pos - self._prev_pos
        dt = self.timestep * n_steps
        self._speed = float(np.linalg.norm(delta[:2]) / dt * 1000.0) if dt > 0 else 0.0  # mm/s
        self._distance += float(np.linalg.norm(delta[:2]) * 1000.0)  # mm
        self._prev_pos = pos

        # Update rotor spin visual phases
        self._rotor_angles += (0.5 + self.state.drive) * np.array([1.0, -1.0, 1.0, -1.0])
        self._rotor_angles %= (2 * np.pi)

        st = self.state
        st.position = (float(pos[0]), float(pos[1]), float(pos[2]))
        st.heading_deg = self._heading_deg()
        st.speed_mm_s = self._speed
        st.distance_mm = self._distance
        st.contacts = [t > 0.01 for t in self.mj_data.ctrl[:4]]
        st.leg_phases = [float(a) for a in self._rotor_angles]
        st.sim_time_s = float(self.mj_data.time)

        return True

    def latest_frame(self) -> np.ndarray | None:
        """Render the Crazyflie 2 drone tracking camera view."""
        try:
            self.renderer.update_scene(self.mj_data, camera="trackcam")
            img = self.renderer.render()
            self._last_frame = img
            return img
        except Exception:
            return self._last_frame

    def reset(self) -> None:
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # Set initial altitude z = 0.5 meters
        self.mj_data.qpos[:3] = [0.0, 0.0, 0.5]
        self.mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self._prev_pos = self._get_position()
        self._distance = 0.0
        self._speed = 0.0
        self.state = BodyState()

    def close(self) -> None:
        pass
