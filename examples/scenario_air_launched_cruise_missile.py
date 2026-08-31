import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.math.coordinates import lla_to_ecef
from mover_sim.scenario_air_launched_cruise_missile import (
    run_air_launched_cruise_missile_scenario,
)


def run_example():
    print("=== Running Scenario 2: Air-Launched Cruise Missile ===")

    os.makedirs("output", exist_ok=True)
    output_path = Path("output/scenario_air_launched_cruise_missile.h5")

    with h5py.File(output_path, "w") as h5:
        result = run_air_launched_cruise_missile_scenario(
            mothership_initial_position_ecef=lla_to_ecef(37.6193, -122.3750, 1500.0),
            mothership_cruise_speed=200.0,
            mothership_cruise_altitude=1500.0,
            mothership_cruise_heading=0.0,
            mothership_rtb_position_ecef=lla_to_ecef(37.6193, -122.3650, 1500.0),
            missile_launch_time=2.0,
            missile_cruise_speed=250.0,
            missile_cruise_altitude=1200.0,
            missile_cruise_heading=0.0,
            missile_drop_duration=0.5,
            t_end=20.0,
            sample_interval=0.1,
            output_group=h5.create_group("air_run_001"),
        )

    print(f"Simulation ended at t = {result['engine'].t:.2f}s. Telemetry written to {output_path}")


if __name__ == "__main__":
    run_example()
