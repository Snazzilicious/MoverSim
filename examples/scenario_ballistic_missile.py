import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.math.coordinates import lla_to_ecef

# XXX Should be able to completely reuse RocketMover and RocketController after
# TODO add registration of expended stage to RocketMover

SCENARIO_EVENT_TOPICS = [
    "platform_registered",
    "stage_1_burnout",
    "stage_1_separation",
    "stage_2_burnout",
    "stage_2_separation",
    "ballistic_coast_start",
    "spent_stage_ground_impact",
    "active_body_ground_impact",
]


def run_ballistic_missile_scenario(
    initial_position_ecef,
    target_position_ecef,
    peak_altitude,
    stages,
    t_end,
    sample_interval,
    output_group,
):
    """Run ballistic missile tracking a given location and save HDF5 telemetry.

    Args:
        initial_position_ecef: Initial ballistic-missile ECEF position vector in meters.
        target_position_ecef: Target ECEF position vector in meters.
        peak_altitude: Desired ballistic peak altitude above the WGS-84 ellipsoid in meters.
        stages: One-stage or two-stage stack definition containing the physical parameters for each stage.
        t_end: Maximum scenario run time in seconds.
        sample_interval: HDF5 logging sample interval in seconds.
        output_group: Caller-created `h5py.Group` used as the root for this scenario run.

    Returns:
        A dictionary containing the simulation engine, active platform, logger, and output group.
    """

    # XXX If we need to validate all inputs here, can re-include that from old scenario
    # But I think constructors decently cover all that

    ascent_program = _derive_ascent_program(
        initial_position_ecef,
        target_position_ecef,
        peak_altitude,
    )
    initial_orientation = _orientation_from_ascent_azimuth(
        initial_position_ecef,
        ascent_program["ascent_azimuth"],
        ascent_program["initial_ascent_pitch"],
    )
    initial_velocity = np.zeros(3)
    initial_body_rates = np.zeros(3)

    engine = SimulationEngine()
    # TODO Need to translate the user-provided arguments into Ballistic missile and guidance constructor arguments
    mover = BallisticMissileMover(
        initial_position=initial_position_ecef,
        initial_velocity=initial_velocity,
        initial_orientation=initial_orientation,
        initial_body_rates=initial_body_rates,
        stages=stages,
    )
    controller = BallisticMissileController(
        ascent_program=ascent_program,
        stages=stages,
        peak_altitude=peak_altitude,
    )
    platform = Platform("ballistic_missile", mover, controller)
    engine.register_platform(platform)

    logger = HDF5Logger(
        engine,
        output_group,
        sample_interval=sample_interval,
        include_state=True,
        include_lla=True,
        include_events=True,
        event_topics=SCENARIO_EVENT_TOPICS,
    )

    engine.run(t_end)

    return {
        "engine": engine,
        "platform": platform,
        "logger": logger,
        "output_group": output_group,
    }


def run_example():
    print("=== Running Scenario: Ballistic Missile ===")

    os.makedirs("output", exist_ok=True)

    output_path = Path("output/scenario_ballistic_missile.h5")

    with h5py.File(output_path, "w") as h5:
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
            output_group=h5.create_group("one_stage_run"),
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
            output_group=h5.create_group("two_stage_run"),
        )

    print(
        f"One-stage simulation ended at t = {one_stage_result['engine'].t:.2f}s. "
        f"Telemetry written to {output_path} under /one_stage_run"
    )
    print(
        f"Two-stage simulation ended at t = {two_stage_result['engine'].t:.2f}s. "
        f"Telemetry written to {output_path} under /two_stage_run"
    )


if __name__ == "__main__":
    run_example()
