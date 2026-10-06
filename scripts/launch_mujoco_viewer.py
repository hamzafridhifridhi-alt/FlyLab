"""Launch interactive 3D MuJoCo desktop GUI viewer for the fly body.

Controls (no Enter key needed):
  W           Walk forward
  A           Turn left
  D           Turn right
  S           Stop
  Q           Quit
"""
import sys
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
from flygym.compose import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    CPGController,
    PreprogrammedSteps,
    apply_locomotion_action,
    get_default_locomotion_dof_order,
    make_locomotion_fly,
    make_tripod_cpg_network,
)

# --- Shared command state ---
command = {
    "drive": 0.0,   # 0.0 = stop, 1.0 = full walk
    "bias":  0.0,   # -1.0 = hard left, +1.0 = hard right
}
quit_flag = threading.Event()


def _print_cmd(msg):
    print(f"\r[CMD] {msg:<20}", end="", flush=True)


def key_listener():
    """Non-blocking single-key reader; uses msvcrt on Windows."""
    print("\n=== MuJoCo Fly Viewer ===")
    print("  W  = Walk forward")
    print("  A  = Turn left")
    print("  D  = Turn right")
    print("  S  = Stop")
    print("  Q  = Quit")
    print("=========================\n")

    if sys.platform == "win32":
        import msvcrt
        while not quit_flag.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch().lower()
                if ch == "w":
                    command["drive"] = 1.0; command["bias"] = 0.0
                    _print_cmd("Walk forward")
                elif ch == "a":
                    command["drive"] = 0.6; command["bias"] = -0.8
                    _print_cmd("Turn left")
                elif ch == "d":
                    command["drive"] = 0.6; command["bias"] = 0.8
                    _print_cmd("Turn right")
                elif ch == "s":
                    command["drive"] = 0.0; command["bias"] = 0.0
                    _print_cmd("Stop")
                elif ch == "q":
                    quit_flag.set(); break
            time.sleep(0.02)
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not quit_flag.is_set():
                ch = sys.stdin.read(1).lower()
                if ch == "w":
                    command["drive"] = 1.0; command["bias"] = 0.0
                    _print_cmd("Walk forward")
                elif ch == "a":
                    command["drive"] = 0.6; command["bias"] = -0.8
                    _print_cmd("Turn left")
                elif ch == "d":
                    command["drive"] = 0.6; command["bias"] = 0.8
                    _print_cmd("Turn right")
                elif ch in ("s", " "):
                    command["drive"] = 0.0; command["bias"] = 0.0
                    _print_cmd("Stop")
                elif ch == "q":
                    quit_flag.set(); break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def set_cpg_from_command(cpg, drive, bias, base_freq=12.0):
    """Translate high-level command into per-leg CPG amplitude/frequency."""
    amp = np.ones(6) * drive
    freq = np.ones(6) * base_freq * (0.35 + 0.65 * drive)
    turn = 0.6 * bias
    amp[0:3] *= max(0.0, 1.0 - max(0.0,  turn))
    amp[3:6] *= max(0.0, 1.0 - max(0.0, -turn))
    cpg.intrinsic_amps = amp
    cpg.intrinsic_freqs = freq


def main():
    print("Initializing NeuroMechFly body model in MuJoCo...")
    fly = make_locomotion_fly(colorize=True)
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position=[0.0, 0.0, 0.5],
        spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
        add_ground_contact_sensors=True,
    )

    timestep = 2e-4
    sim = Simulation(world, timestep=timestep)
    sim.reset()

    cpg = make_tripod_cpg_network(timestep, intrinsic_frequency=12.0)
    controller = CPGController(
        cpg_network=cpg,
        preprogrammed_steps=PreprogrammedSteps(),
        output_dof_order=get_default_locomotion_dof_order(),
    )
    cpg.reset()
    set_cpg_from_command(cpg, 0.0, 0.0)  # start stopped

    mj_model = sim.mj_model
    mj_data  = sim.mj_data

    # Find thorax body for camera tracking
    thorax_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "nmf/c_thorax")
    if thorax_id < 0:
        for i in range(mj_model.nbody):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if "thorax" in name.lower():
                thorax_id = i
                break
    print(f"Camera tracking: body id={thorax_id}")

    # Start keyboard thread
    kt = threading.Thread(target=key_listener, daemon=True)
    kt.start()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Lock camera onto fly's thorax
        viewer.cam.type         = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid  = thorax_id
        viewer.cam.distance     = 5.0      # close enough to see legs
        viewer.cam.elevation    = -20.0    # slightly above ground plane
        viewer.cam.azimuth      = 90.0     # side view

        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        while viewer.is_running() and not quit_flag.is_set():
            step_start = time.perf_counter()

            set_cpg_from_command(cpg, command["drive"], command["bias"])
            action = controller.step()
            apply_locomotion_action(sim, fly.name, action)
            sim.step()
            viewer.sync()

            elapsed = time.perf_counter() - step_start
            if elapsed < timestep:
                time.sleep(timestep - elapsed)

    quit_flag.set()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
