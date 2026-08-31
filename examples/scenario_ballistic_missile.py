import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.math.coordinates import lla_to_ecef
from mover_sim.scenario_ballistic_missile import run_ballistic_missile_scenario


def run_example():
    print("=== Running Scenario 3: Ballistic Missile ===")

    os.makedirs("output", exist_ok=True)

    one_stage_output = Path("output/scenario_ballistic_missile_one_stage.h5")
    two_stage_output = Path("output/scenario_ballistic_missile_two_stage.h5")

    one_stage_result = run_ballistic_missile_scenario(
        initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
        target_position_ecef=lla_to_ecef(37.2, -121.8, 0.0),
        peak_altitude=20000.0,
        stages=[
            {
                "dry_mass": 1000.0,
                "propellant_mass": 500.0,
                "burn_duration": 10.0,
                "thrust": 10000.0,
                "drag_coefficient": 0.1,
                "reference_area": 1.0,
                "separation_delay": 1.0,
            }
        ],
        t_end=20.0,
        sample_interval=0.1,
        output_path=one_stage_output,
    )

    two_stage_result = run_ballistic_missile_scenario(
        initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
        target_position_ecef=lla_to_ecef(37.5, -121.5, 0.0),
        peak_altitude=40000.0,
        stages=[
            {
                "dry_mass": 1000.0,
                "propellant_mass": 500.0,
                "burn_duration": 10.0,
                "thrust": 10000.0,
                "drag_coefficient": 0.1,
                "reference_area": 1.0,
                "separation_delay": 2.0,
            },
            {
                "dry_mass": 500.0,
                "propellant_mass": 250.0,
                "burn_duration": 5.0,
                "thrust": 8000.0,
                "drag_coefficient": 0.08,
                "reference_area": 0.8,
                "separation_delay": 1.0,
            },
        ],
        t_end=30.0,
        sample_interval=0.1,
        output_path=two_stage_output,
    )

    print(
        f"One-stage simulation ended at t = {one_stage_result['engine'].t:.2f}s. "
        f"Telemetry written to {one_stage_output}"
    )
    print(
        f"Two-stage simulation ended at t = {two_stage_result['engine'].t:.2f}s. "
        f"Telemetry written to {two_stage_output}"
    )


if __name__ == "__main__":
    run_example()
