import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import HDF5Logger
from mover_sim.core.platform import Platform
from mover_sim.math.coordinates import ecef_to_lla, lla_to_ecef
from mover_sim.math.orientation import build_aircraft_body_axes, project_to_rotation_matrix
from mover_sim.models.aircraft_mover import RocketController, RocketMover

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


def _local_enu_basis(position_ecef):
    position = np.asarray(position_ecef, dtype=float)
    if position.shape != (3,):
        raise ValueError("position_ecef must have shape (3,)")

    lat_deg, lon_deg, _ = ecef_to_lla(position[0], position[1], position[2])
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ])
    up = position / max(np.linalg.norm(position), 1e-6)
    return east, north, up


def _derive_ascent_azimuth(initial_position_ecef, target_position_ecef):
    initial_position = np.asarray(initial_position_ecef, dtype=float)
    target_position = np.asarray(target_position_ecef, dtype=float)
    east, north, up = _local_enu_basis(initial_position)

    rel = target_position - initial_position
    rel_horizontal = rel - np.dot(rel, up) * up
    east_component = np.dot(rel_horizontal, east)
    north_component = np.dot(rel_horizontal, north)
    return np.arctan2(east_component, north_component)


def _orientation_from_ascent_azimuth(position_ecef, ascent_azimuth, ascent_pitch):
    east, north, up = _local_enu_basis(position_ecef)
    forward_horizontal = np.cos(ascent_azimuth) * north + np.sin(ascent_azimuth) * east
    forward = np.cos(ascent_pitch) * forward_horizontal + np.sin(ascent_pitch) * up
    forward_axis, right_axis, up_axis = build_aircraft_body_axes(forward, up)
    return project_to_rotation_matrix(np.column_stack([forward_axis, right_axis, up_axis]))


def _derive_rocket_controller_kwargs(
    initial_position_ecef,
    target_position_ecef,
    peak_altitude,
    stages,
):
    """Derive `RocketController` kwargs heuristically from mission-level inputs.

    `peak_altitude` is treated as an ascent-shaping request rather than an exact
    apogee solver. Larger requested apogees produce a steeper, longer initial
    ascent program; smaller requested apogees produce a flatter, shorter one.
    """
    initial_position = np.asarray(initial_position_ecef, dtype=float)
    target_position = np.asarray(target_position_ecef, dtype=float)
    peak_altitude = float(peak_altitude)
    if peak_altitude <= 0.0:
        raise ValueError("peak_altitude must be greater than 0")

    _, _, initial_altitude = ecef_to_lla(
        initial_position[0],
        initial_position[1],
        initial_position[2],
    )
    altitude_gain = max(peak_altitude - initial_altitude, 1.0)

    # Map requested apogee into a normalized shaping factor. This keeps the
    # public API mission-oriented while producing controller inputs that vary
    # monotonically with the requested peak altitude.
    low_gain = 10_000.0
    high_gain = 50_000.0
    shaping = np.clip((altitude_gain - low_gain) / (high_gain - low_gain), 0.0, 1.0)

    target_ascent_pitch = np.radians(55.0 + shaping * (82.0 - 55.0))
    vertical_rise_time = 1.0 + shaping * (3.0 - 1.0)
    pitch_over_duration = 2.0 + shaping * (6.0 - 2.0)

    separation_delay = 0.0
    if stages:
        separation_delay = float(stages[0].get("separation_delay", 0.0))

    return {
        "target_position_ecef": target_position,
        "launch_azimuth": _derive_ascent_azimuth(initial_position, target_position),
        "vertical_rise_time": vertical_rise_time,
        "pitch_over_duration": pitch_over_duration,
        "target_ascent_pitch": target_ascent_pitch,
        "separation_delay": separation_delay,
    }


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
        peak_altitude: Desired apogee/ascent-shaping altitude above the WGS-84 ellipsoid in meters.
        stages: One-stage or two-stage `RocketMover` stage definitions.
        t_end: Maximum scenario run time in seconds.
        sample_interval: HDF5 logging sample interval in seconds.
        output_group: Caller-created `h5py.Group` used as the root for this scenario run.

    Returns:
        A dictionary containing the simulation engine, active platform, logger, and output group.
    """

    # XXX If we need to validate all inputs here, can re-include that from old scenario
    # But I think constructors decently cover all that

    controller_kwargs = _derive_rocket_controller_kwargs(
        initial_position_ecef,
        target_position_ecef,
        peak_altitude,
        stages,
    )
    initial_orientation = _orientation_from_ascent_azimuth(
        initial_position_ecef,
        controller_kwargs["launch_azimuth"],
        0.5 * np.pi,
    )
    initial_velocity = np.zeros(3)
    initial_body_rates = np.zeros(3)

    engine = SimulationEngine()
    mover = RocketMover(
        initial_position=initial_position_ecef,
        initial_velocity=initial_velocity,
        initial_orientation=initial_orientation,
        initial_body_rates=initial_body_rates,
        stages=stages,
    )
    controller = RocketController(**controller_kwargs)
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
                    "rotational_mass": np.diag([6.0e4, 1.2e5, 1.2e5]),
                    "angular_damping": np.array([2.0e4, 4.0e4, 4.0e4]),
                    "max_thrust": 10000.0,
                    "max_steering_moment": 5.0e4,
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
                    "rotational_mass": np.diag([6.0e4, 1.2e5, 1.2e5]),
                    "angular_damping": np.array([2.0e4, 4.0e4, 4.0e4]),
                    "max_thrust": 10000.0,
                    "max_steering_moment": 5.0e4,
                    "separation_delay": 2.0,
                },
                {
                    "dry_mass": 500.0,
                    "propellant_mass": 250.0,
                    "burn_duration": 5.0,
                    "thrust": 8000.0,
                    "drag_coefficient": 0.08,
                    "reference_area": 0.8,
                    "rotational_mass": np.diag([3.0e4, 6.0e4, 6.0e4]),
                    "angular_damping": np.array([1.0e4, 2.0e4, 2.0e4]),
                    "max_thrust": 8000.0,
                    "max_steering_moment": 2.5e4,
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
