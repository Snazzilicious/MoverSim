import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.math.coordinates import lla_to_ecef
from mover_sim.scenario_surface_launched_cruise_missile import (
    run_surface_launched_cruise_missile_scenario,
)


def run_example():
    print("=== Running Scenario 1: Surface-Launched Cruise Missile ===")

    os.makedirs("output", exist_ok=True)
    output_path = Path("output/scenario_surface_launched_cruise_missile.h5")

    with h5py.File(output_path, "w") as h5:
        result = run_surface_launched_cruise_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.6193, -122.3750, 10.0),
            cruise_speed=250.0,
            cruise_altitude=1200.0,
            cruise_heading=0.0,
            boost_duration=2.0,
            boost_acceleration=30.0,
            launch_pitch_angle=0.35,
            t_end=20.0,
            sample_interval=0.1,
            output_group=h5.create_group("surface_run_001"),
        )

    print(f"Simulation ended at t = {result['engine'].t:.2f}s. Telemetry written to {output_path}")


if __name__ == "__main__":
    run_example()
