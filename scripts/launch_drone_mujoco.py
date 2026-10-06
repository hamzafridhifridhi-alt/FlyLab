"""Interactive MuJoCo 3D Viewer for Bitcraze Crazyflie 2 Drone piloted by Drosophila Nervous System."""

from __future__ import annotations

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from flylab.body import FlyBody


def main() -> None:
    print("=========================================================")
    print("  Bitcraze Crazyflie 2 Drone MuJoCo Simulation Viewer")
    print("=========================================================")
    print("\nInitializing Bitcraze Crazyflie 2 drone model...")

    drone = FlyBody()

    print("\nDrone initialized successfully.")
    print("\nFlight Controls (Neural Descending Drive Modulation):")
    print("  [W] Ascend / Forward Flight Drive (160 Hz DN Drive)")
    print("  [A] Steer / Turn Left (Left DN > Right DN)")
    print("  [D] Steer / Turn Right (Right DN > Left DN)")
    print("  [S] Hover / Soft Landing (Resting Drive)")
    print("  [Q] Exit Simulation")
    print("\nLaunching native 3D MuJoCo viewport...\n")

    # Launch native MuJoCo passive 3D viewer
    with mujoco.viewer.launch_passive(drone.mj_model, drone.mj_data) as viewer:
        # Default flight state
        left_dn = 60.0
        right_dn = 60.0

        # Sync viewer camera
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45

        step_count = 0
        while viewer.is_running():
            step_start = time.perf_counter()

            # Pass neural descending rates to Crazyflie 2 quadrotor flight controller
            drone.set_descending_drive(left_dn, right_dn)
            drone.step(1)

            viewer.sync()

            step_count += 1
            if step_count % 200 == 0:
                pos = drone.state.position
                print(
                    f"[Drone Telemetry] Pos: ({pos[0]:.2f}m, {pos[1]:.2f}m, {pos[2]:.2f}m) | "
                    f"Heading: {drone.state.heading_deg:.0f}° | Speed: {drone.state.speed_mm_s:.1f} mm/s | "
                    f"Behavior: {drone.state.behavior}"
                )

            # Match 1ms simulation timestep
            elapsed = time.perf_counter() - step_start
            if elapsed < drone.timestep:
                time.sleep(drone.timestep - elapsed)

    drone.close()
    print("\nSimulation closed successfully.")


if __name__ == "__main__":
    main()
