import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.math.coordinates import lla_to_ecef


SCENARIO_EVENT_TOPICS = [
    "platform_registered",
    "missile_release",
    "missile_drop_start",
    "missile_drop_end",
    "missile_ignite",
    "missile_cruise_established",
    "mothership_rtb_start",
    "mothership_rtb_arrival",
    "missile_ground_impact",
]


def run_air_launched_cruise_missile_scenario(
    mothership_initial_position_ecef,
    mothership_cruise_speed,
    mothership_cruise_altitude,
    mothership_cruise_heading,
    mothership_rtb_position_ecef,
    missile_launch_time,
    missile_cruise_speed,
    missile_cruise_altitude,
    missile_cruise_heading,
    missile_drop_duration,
    t_end,
    sample_interval,
    output_group,
):
    """Run air-launched cruise missile tracking a given heading and save HDF5 telemetry.

    Args:
        mothership_initial_position_ecef: Initial mothership ECEF position vector in meters.
        mothership_cruise_speed: Commanded mothership cruise speed in meters/second.
        mothership_cruise_altitude: Commanded mothership cruise altitude above the WGS-84 ellipsoid in meters.
        mothership_cruise_heading: Commanded mothership local-ENU azimuth heading in radians.
        mothership_rtb_position_ecef: Return-to-base destination in ECEF meters.
        missile_launch_time: Scenario time at which the missile is released in seconds.
        missile_cruise_speed: Commanded missile cruise speed in meters/second.
        missile_cruise_altitude: Commanded missile cruise altitude above the WGS-84 ellipsoid in meters.
        missile_cruise_heading: Commanded missile local-ENU azimuth heading in radians.
        missile_drop_duration: Duration of the unpowered drop phase in seconds.
        t_end: Maximum scenario run time in seconds.
        sample_interval: HDF5 logging sample interval in seconds.
        output_group: Caller-created `h5py.Group` used as the root for this scenario run.

    Returns:
        A dictionary containing the simulation engine, mothership platform, logger, and output group.
    """

    # XXX If we need to validate all inputs here, can re-include that from old scenario
    # But I think constructors decently cover all that

    if missile_launch_time > t_end:
        raise ValueError("missile_launch_time must be less than or equal to t_end")

    mothership_initial_orientation = _orientation_from_heading_pitch(
        mothership_initial_position_ecef,
        mothership_cruise_heading,
        0.0,
    )
    mothership_initial_velocity = _velocity_from_heading_speed(
        mothership_initial_position_ecef,
        mothership_cruise_heading,
        mothership_cruise_speed,
    )
    mothership_initial_body_rates = np.zeros(3)

    engine = SimulationEngine()
    mothership_mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=mothership_initial_position_ecef,
        initial_velocity=mothership_initial_velocity,
        initial_orientation=mothership_initial_orientation,
        initial_body_rates=mothership_initial_body_rates,
    )
    mothership_controller = AirLaunchedCruiseMissileMothershipController(
        cruise_speed=mothership_cruise_speed,
        cruise_altitude=mothership_cruise_altitude,
        cruise_heading=mothership_cruise_heading,
        rtb_position_ecef=mothership_rtb_position_ecef,
    )
    mothership_platform = Platform("mothership", mothership_mover, mothership_controller)
    engine.register_platform(mothership_platform)

    def spawn_missile(current_engine, current_mothership_platform):
        _spawn_released_missile(
            current_engine,
            current_mothership_platform,
            missile_launch_time,
            missile_cruise_speed,
            missile_cruise_altitude,
            missile_cruise_heading,
            missile_drop_duration,
        )
        _begin_mothership_rtb(current_engine, current_mothership_platform)

    release_callback = _make_missile_release_callback(mothership_platform, spawn_missile)
    engine.schedule(missile_launch_time, release_callback, "MissileRelease")

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
        "mothership_platform": mothership_platform,
        "logger": logger,
        "output_group": output_group,
    }


def run_example():
    print("=== Running Scenario: Air-Launched Cruise Missile ===")

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
