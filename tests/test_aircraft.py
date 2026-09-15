import os
import csv
import pytest
import numpy as np
from mover_sim.core.platform import Platform
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import CSVLogger
from mover_sim.models.aircraft_mover import (
    Aircraft6DOFAutopilot,
    Aircraft6DOFMover,
    AircraftMover,
    AircraftAutopilot,
    FixedWingAutopilot,
    FixedWingMover,
    RocketController,
    RocketMover,
)
from mover_sim.math.coordinates import lla_to_ecef, ecef_to_enu, ecef_to_lla
from mover_sim.math.orientation import rotate_vector_by_quaternion


def _rotation_about_body_right(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [s, 0.0, c],
    ])


def _rotation_about_body_up(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _rotation_about_body_forward(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])

def test_aircraft_mover_initialization():
    pos = lla_to_ecef(0.0, 0.0, 5000.0)
    vel = [150.0, 0.0, 0.0]
    
    mover = AircraftMover(pos, vel, mass=12000.0, area=35.0, cd0=0.015, t_max=90000.0)
    assert mover.mass == 12000.0
    assert mover.area == 35.0
    assert mover.cd0 == 0.015
    assert mover.t_max == 90000.0
    assert mover.thrust_cmd == 0.0
    assert mover.bank_angle_cmd == 0.0
    assert mover.lift_cmd == 0.0


def test_aircraft_mover_remains_translational_point_mass_model():
    pos = lla_to_ecef(0.0, 0.0, 3000.0)
    vel = np.array([0.0, 150.0, 0.0])

    mover = AircraftMover(pos, vel)
    state = mover.get_state()

    assert mover.get_state_dimension() == 6
    assert state.shape == (6,)
    assert np.allclose(mover.position, state[:3])
    assert np.allclose(mover.velocity, state[3:])

def test_aircraft_autopilot_guidance():
    engine = SimulationEngine()
    engine.max_step = 0.2
    
    # Starting position at equator, prime meridian, altitude 2000 m
    pos0 = lla_to_ecef(0.0, 0.0, 2000.0)
    # Flying East at 150 m/s: velocity in ECEF is [0.0, 150.0, 0.0] (approx, at Prime Meridian Equator)
    # Let's verify: at (0,0,H), ECEF y direction is East! So [0, 150, 0] is East.
    vel0 = np.array([0.0, 150.0, 0.0])
    
    mover = AircraftMover(pos0, vel0)
    
    # Waypoints:
    # Waypoint 1: 5 km East, altitude 2200 m (climbing 200m)
    wp1 = lla_to_ecef(0.0, 0.045, 2200.0) # approx 5km east in longitude
    # Waypoint 2: 10 km East, altitude 2200 m
    wp2 = lla_to_ecef(0.0, 0.090, 2200.0)
    
    autopilot = AircraftAutopilot([wp1, wp2], target_speed=160.0, waypoint_radius=500.0)
    
    platform = Platform("f16", mover, autopilot)
    engine.register_platform(platform)
    
    # Track initial distance to waypoint 1
    lat, lon, alt = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    assert np.isclose(alt, 2000.0, atol=1.0)
    
    # Run simulation for 5.0 seconds
    engine.run(5.0)
    
    # Check that speed is tracking target (approx) and aircraft climbed
    speed = np.linalg.norm(mover.velocity)
    assert np.isclose(speed, 160.0, atol=10.0)
    
    lat_end, lon_end, alt_end = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    # The aircraft should have climbed towards 2200m
    assert alt_end > 2000.0
    # The aircraft should have moved East (longitude increased)
    assert lon_end > 0.0

def test_csv_logger(tmp_path):
    engine = SimulationEngine()
    
    pos = lla_to_ecef(45.0, 45.0, 1000.0)
    vel = [100.0, 0.0, 0.0]
    mover = AircraftMover(pos, vel)
    platform = Platform("test_plane", mover)
    engine.register_platform(platform)
    
    log_file = os.path.join(tmp_path, "telemetry.csv")
    logger = CSVLogger(engine, log_file, log_interval=0.5)
    
    # Run for 1.5 seconds
    engine.run(1.5)
    
    # Check that file exists and contains data
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        rows = list(csv.reader(f))
        
    # Expect: Header + one row per sampled platform state
    assert len(rows) >= 4
    
    # Verify header columns
    header = rows[0]
    assert header[0] == "time"
    assert "platform_id" in header
    assert "state_dim" in header
    assert "x" in header
    assert "lat" in header
    assert "vx" in header
    assert "state_json" in header
    
    # Verify first sampled row
    first_row = rows[1]
    assert np.isclose(float(first_row[0]), 0.0)
    assert first_row[1] == "test_plane"
    assert int(first_row[2]) == 6
    assert first_row[-1].startswith("[")


def test_aircraft_6dof_mover_initialization():
    pos = lla_to_ecef(0.0, 0.0, 5000.0)
    vel = np.array([0.0, 150.0, 0.0])

    mover = Aircraft6DOFMover(pos, vel)

    assert mover.get_state_dimension() == 13
    assert mover.mass == 10000.0
    assert mover.orientation.shape == (4,)
    assert mover.body_rates.shape == (3,)
    assert np.isclose(np.linalg.norm(mover.orientation), 1.0, atol=1e-7)


def test_aircraft_6dof_roll_command_changes_bank():
    engine = SimulationEngine()
    engine.max_step = 0.02

    pos0 = lla_to_ecef(0.0, 0.0, 2000.0)
    vel0 = np.array([0.0, 180.0, 0.0])
    mover = Aircraft6DOFMover(pos0, vel0, area=0.0, angular_damping=[1.0e4, 1.0e4, 1.0e4])
    mover.roll_moment_cmd = 2.0e5

    engine.register_platform(Platform("roll_test", mover))
    engine.run(1.0)

    local_up = mover.position / np.linalg.norm(mover.position)
    body_up = rotate_vector_by_quaternion([0.0, 0.0, 1.0], mover.orientation)

    assert mover.body_rates[0] > 0.0
    assert np.dot(body_up, local_up) < 0.999


def test_aircraft_6dof_pitch_command_changes_flight_path():
    engine = SimulationEngine()
    engine.max_step = 0.02

    pos0 = lla_to_ecef(0.0, 0.0, 2000.0)
    vel0 = np.array([0.0, 180.0, 0.0])
    mover = Aircraft6DOFMover(
        pos0,
        vel0,
        mass=1500.0,
        area=0.0,
        t_max=2.0e5,
        angular_damping=[1.0e4, 1.0e4, 1.0e4],
    )
    mover.thrust_cmd = 2.0e5
    mover.pitch_moment_cmd = 1.0e5

    engine.register_platform(Platform("pitch_test", mover))
    engine.run(5.0)

    _, _, alt_end = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    local_up = mover.position / np.linalg.norm(mover.position)
    vertical_speed = np.dot(mover.velocity, local_up)

    assert alt_end > 2000.0
    assert vertical_speed > 0.0


def test_aircraft_6dof_quaternion_remains_normalized():
    engine = SimulationEngine()
    engine.max_step = 0.02

    pos0 = lla_to_ecef(0.0, 0.0, 2000.0)
    vel0 = np.array([0.0, 180.0, 0.0])
    mover = Aircraft6DOFMover(pos0, vel0, area=0.0)
    mover.thrust_cmd = 5.0e4
    mover.roll_moment_cmd = 5.0e4
    mover.pitch_moment_cmd = 2.5e4
    mover.yaw_moment_cmd = 1.5e4

    engine.register_platform(Platform("quat_test", mover))
    engine.run(5.0)

    assert np.isclose(np.linalg.norm(mover.orientation), 1.0, atol=1e-3)


def test_aircraft_6dof_yaw_roll_commands_change_trajectory():
    engine = SimulationEngine()
    engine.max_step = 0.02

    ref_lat, ref_lon, ref_alt = 0.0, 0.0, 2000.0
    pos0 = lla_to_ecef(ref_lat, ref_lon, ref_alt)
    vel0 = np.array([0.0, 180.0, 0.0])

    baseline = Aircraft6DOFMover(
        pos0,
        vel0,
        mass=2000.0,
        area=0.0,
        t_max=2.0e5,
        angular_damping=[8.0e3, 8.0e3, 8.0e3],
        use_coriolis=False,
    )
    maneuvering = Aircraft6DOFMover(
        pos0,
        vel0,
        mass=2000.0,
        area=0.0,
        t_max=2.0e5,
        angular_damping=[8.0e3, 8.0e3, 8.0e3],
        use_coriolis=False,
    )

    baseline.thrust_cmd = 1.2e5
    maneuvering.thrust_cmd = 1.2e5
    maneuvering.roll_moment_cmd = 8.0e4
    maneuvering.yaw_moment_cmd = 8.0e4

    engine.register_platform(Platform("baseline", baseline))
    engine.register_platform(Platform("maneuvering", maneuvering))
    engine.run(4.0)

    _, baseline_north, _ = ecef_to_enu(
        baseline.position[0],
        baseline.position[1],
        baseline.position[2],
        ref_lat,
        ref_lon,
        ref_alt,
    )
    _, maneuver_north, _ = ecef_to_enu(
        maneuvering.position[0],
        maneuvering.position[1],
        maneuvering.position[2],
        ref_lat,
        ref_lon,
        ref_alt,
    )

    separation = np.linalg.norm(maneuvering.position - baseline.position)

    assert abs(maneuver_north - baseline_north) > 10.0
    assert separation > 50.0


def test_aircraft_6dof_autopilot_generates_moment_commands():
    engine = SimulationEngine()

    pos0 = lla_to_ecef(0.0, 0.0, 2000.0)
    vel0 = np.array([0.0, 180.0, 0.0])
    mover = Aircraft6DOFMover(pos0, vel0, mass=2000.0, area=0.0, t_max=2.0e5, use_coriolis=False)
    wp = lla_to_ecef(0.0, 0.02, 2400.0)
    autopilot = Aircraft6DOFAutopilot([wp], target_speed=220.0, waypoint_radius=100.0, update_interval=0.1)

    platform = Platform("f16_6dof", mover, autopilot)
    engine.register_platform(platform)
    autopilot.initialize(engine)
    autopilot.update(0.0, engine)

    assert mover.thrust_cmd > 0.0
    assert abs(mover.pitch_moment_cmd) > 0.0


def test_aircraft_6dof_autopilot_changes_trajectory_toward_waypoint():
    engine = SimulationEngine()
    engine.max_step = 0.02

    pos0 = lla_to_ecef(0.0, 0.0, 2000.0)
    vel0 = np.array([0.0, 180.0, 0.0])
    mover = Aircraft6DOFMover(
        pos0,
        vel0,
        mass=2000.0,
        area=0.02,
        t_max=2.0e5,
        angular_damping=[8.0e3, 8.0e3, 8.0e3],
    )
    wp = lla_to_ecef(0.0, 0.03, 2400.0)
    autopilot = Aircraft6DOFAutopilot([wp], target_speed=220.0, waypoint_radius=150.0, update_interval=0.05)

    platform = Platform("f16_6dof", mover, autopilot)
    engine.register_platform(platform)

    distance0 = np.linalg.norm(wp - mover.position)
    _, lon0, alt0 = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    engine.run(6.0)
    _, lon_end, alt_end = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    distance_end = np.linalg.norm(wp - mover.position)

    assert lon_end > lon0
    assert distance_end < distance0
    assert alt_end > alt0 - 300.0


def test_fixed_wing_initial_orientation_is_orthonormal():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])

    mover = FixedWingMover(pos, vel, use_coriolis=False)
    orientation = mover.orientation
    forward = orientation[:, 0]

    assert np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(orientation), 1.0, atol=1e-7)
    assert np.allclose(forward, vel / np.linalg.norm(vel), atol=1e-7)


def test_fixed_wing_projects_non_orthonormal_initial_orientation():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    noisy_orientation = np.array([
        [1.0, 0.05, 0.0],
        [0.0, 0.98, -0.08],
        [0.03, 0.02, 1.02],
    ])

    mover = FixedWingMover(pos, vel, initial_orientation=noisy_orientation, use_coriolis=False)
    orientation = mover.orientation

    assert np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(orientation), 1.0, atol=1e-7)


def test_fixed_wing_aero_force_is_zero_at_low_speed():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([1.0e-8, 0.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    assert np.allclose(mover._aerodynamic_force(pos, vel, mover.orientation), np.zeros(3))


def test_fixed_wing_restoring_moment_is_zero_at_low_speed():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([1.0e-8, 0.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    assert np.allclose(
        mover._restoring_moment_components(pos, vel, mover.orientation, np.array([1.0, -1.0, 0.5])),
        np.zeros(3),
    )


def test_fixed_wing_aero_force_is_pure_drag_when_aligned():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    aero_force = mover._aerodynamic_force(pos, vel, mover.orientation)
    body_force = mover.orientation.T @ aero_force

    assert body_force[0] < 0.0
    assert np.isclose(body_force[1], 0.0, atol=1e-9)
    assert np.isclose(body_force[2], 0.0, atol=1e-9)


def test_fixed_wing_restoring_moment_is_zero_when_aligned_and_rates_zero():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    assert np.allclose(
        mover._restoring_moment_components(pos, vel, mover.orientation, np.zeros(3)),
        np.zeros(3),
        atol=1e-9,
    )


def test_fixed_wing_drag_increases_with_alpha_and_beta():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    aligned_body_force = mover.orientation.T @ mover._aerodynamic_force(pos, vel, mover.orientation)

    pitched_orientation = mover.orientation @ _rotation_about_body_right(np.radians(10.0))
    pitched_body_force = pitched_orientation.T @ mover._aerodynamic_force(pos, vel, pitched_orientation)

    yawed_orientation = mover.orientation @ _rotation_about_body_up(np.radians(-10.0))
    yawed_body_force = yawed_orientation.T @ mover._aerodynamic_force(pos, vel, yawed_orientation)

    assert abs(pitched_body_force[0]) > abs(aligned_body_force[0])
    assert abs(yawed_body_force[0]) > abs(aligned_body_force[0])


def test_fixed_wing_bank_error_fallback_near_vertical_flight():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    local_up = pos / np.linalg.norm(pos)
    vel = 180.0 * local_up
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    banked_orientation = mover.orientation @ _rotation_about_body_forward(np.radians(20.0))
    speed, alpha, beta, bank_error = mover._aerodynamic_angles(pos, vel, banked_orientation)
    restoring = mover._restoring_moment_components(pos, vel, banked_orientation, np.zeros(3))

    assert speed > 0.0
    assert np.isfinite(alpha)
    assert np.isfinite(beta)
    assert np.isfinite(bank_error)
    assert np.isclose(bank_error, 0.0, atol=1e-9)
    assert np.isclose(restoring[0], 0.0, atol=1e-9)


def test_fixed_wing_rate_damping_opposes_each_axis_independently():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    roll_only = mover._restoring_moment_components(pos, vel, mover.orientation, np.array([0.2, 0.0, 0.0]))
    pitch_only = mover._restoring_moment_components(pos, vel, mover.orientation, np.array([0.0, 0.3, 0.0]))
    yaw_only = mover._restoring_moment_components(pos, vel, mover.orientation, np.array([0.0, 0.0, 0.4]))

    assert roll_only[0] < 0.0
    assert np.isclose(roll_only[1], 0.0, atol=1e-9)
    assert np.isclose(roll_only[2], 0.0, atol=1e-9)

    assert pitch_only[1] > 0.0
    assert np.isclose(pitch_only[0], 0.0, atol=1e-9)
    assert np.isclose(pitch_only[2], 0.0, atol=1e-9)

    assert yaw_only[2] < 0.0
    assert np.isclose(yaw_only[0], 0.0, atol=1e-9)
    assert np.isclose(yaw_only[1], 0.0, atol=1e-9)


def test_fixed_wing_roll_rate_decays_with_damping_only():
    engine = SimulationEngine()
    engine.max_step = 0.01

    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(
        pos,
        vel,
        initial_body_rates=np.array([0.3, 0.0, 0.0]),
        bank_restoring_coeff=0.0,
        alpha_restoring_coeff=0.0,
        beta_restoring_coeff=0.0,
        roll_damping_coeff=3.0e4,
        pitch_damping_coeff=0.0,
        yaw_damping_coeff=0.0,
        use_coriolis=False,
    )
    initial_roll_rate = mover.get_state()[mover.get_omega_slice()][0]

    engine.register_platform(Platform("fixed_wing_damping", mover))
    engine.run(0.25)

    final_roll_rate = mover.get_state()[mover.get_omega_slice()][0]
    assert abs(final_roll_rate) < abs(initial_roll_rate)


def test_fixed_wing_bank_relaxes_toward_level():
    engine = SimulationEngine()
    engine.max_step = 0.01

    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    base_mover = FixedWingMover(pos, vel, use_coriolis=False)
    banked_orientation = base_mover.orientation @ _rotation_about_body_forward(np.radians(12.0))
    mover = FixedWingMover(
        pos,
        vel,
        initial_orientation=banked_orientation,
        bank_restoring_coeff=2.0e5,
        alpha_restoring_coeff=0.0,
        beta_restoring_coeff=0.0,
        roll_damping_coeff=2.0e4,
        pitch_damping_coeff=0.0,
        yaw_damping_coeff=0.0,
        use_coriolis=False,
    )

    initial_bank_error = mover._aerodynamic_angles(pos, vel, mover.orientation)[3]
    engine.register_platform(Platform("fixed_wing_bank", mover))
    engine.run(2.0)
    final_bank_error = mover._aerodynamic_angles(mover.position, mover.velocity, mover.orientation)[3]

    assert abs(final_bank_error) < abs(initial_bank_error)


def test_fixed_wing_alpha_relaxes_toward_zero():
    engine = SimulationEngine()
    engine.max_step = 0.01

    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 220.0, 0.0])
    base_mover = FixedWingMover(pos, vel, use_coriolis=False)
    pitched_orientation = base_mover.orientation @ _rotation_about_body_right(np.radians(10.0))
    mover = FixedWingMover(
        pos,
        vel,
        initial_orientation=pitched_orientation,
        bank_restoring_coeff=0.0,
        alpha_restoring_coeff=5.0e4,
        beta_restoring_coeff=0.0,
        roll_damping_coeff=0.0,
        pitch_damping_coeff=2.0e4,
        yaw_damping_coeff=0.0,
        use_coriolis=False,
    )

    initial_alpha = mover._aerodynamic_angles(pos, vel, mover.orientation)[1]
    engine.register_platform(Platform("fixed_wing_alpha", mover))
    engine.run(0.5)
    final_alpha = mover._aerodynamic_angles(mover.position, mover.velocity, mover.orientation)[1]

    assert abs(final_alpha) < abs(initial_alpha)


def test_fixed_wing_orientation_correction_event_projects_committed_state():
    engine = SimulationEngine()

    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    platform = Platform("fixed_wing_event", mover)
    engine.register_platform(platform)

    mover_slice = engine.context.get_state_slice(mover)
    orientation_slice = mover.get_orientation_slice()
    start = mover_slice.start + orientation_slice.start
    stop = mover_slice.start + orientation_slice.stop

    engine.context.committed_y[start:stop] = np.array([
        1.0, 0.05, 0.0,
        0.0, 0.98, -0.08,
        0.03, 0.02, 1.02,
    ])

    event = mover.add_orientation_correction_event(engine)
    event.fix_orientation(engine)

    corrected = engine.context.committed_y[start:stop].reshape((3, 3))
    assert np.allclose(corrected.T @ corrected, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(corrected), 1.0, atol=1e-7)


def test_fixed_wing_autopilot_constructor_stores_normalized_route():
    wp1 = lla_to_ecef(0.0, 0.00, 2000.0)
    wp2 = lla_to_ecef(0.0, 0.02, 2200.0)
    wp3 = lla_to_ecef(0.0, 0.04, 2400.0)

    autopilot = FixedWingAutopilot(
        [list(wp1), list(wp2), list(wp3)],
        target_speed=180.0,
        waypoint_radius=750.0,
        update_interval=0.2,
    )

    assert len(autopilot.waypoints) == 3
    assert all(isinstance(wp, np.ndarray) and wp.shape == (3,) for wp in autopilot.waypoints)
    assert np.allclose(autopilot.target_speeds, [180.0, 180.0, 180.0])
    assert autopilot.target_speed == 180.0
    assert autopilot.max_climb_rate == 4.0
    assert autopilot.waypoint_radius == 750.0
    assert autopilot.update_interval == 0.2
    assert autopilot.current_wp_idx == 0
    assert autopilot.completed is False
    assert autopilot.hold_active is False
    assert autopilot.hold_speed is None
    assert autopilot.hold_altitude is None
    assert autopilot.hold_horizontal_direction is None


def test_fixed_wing_autopilot_constructor_accepts_per_waypoint_speeds():
    wp1 = lla_to_ecef(0.0, 0.00, 2000.0)
    wp2 = lla_to_ecef(0.0, 0.02, 2200.0)
    wp3 = lla_to_ecef(0.0, 0.04, 2400.0)

    autopilot = FixedWingAutopilot(
        [wp1, wp2, wp3],
        target_speed=180.0,
        target_speeds=[160.0, 190.0, 210.0],
        max_climb_rate=6.0,
    )

    assert np.allclose(autopilot.target_speeds, [160.0, 190.0, 210.0])
    assert autopilot.max_climb_rate == 6.0


def test_fixed_wing_autopilot_constructor_allows_single_waypoint_route():
    wp = lla_to_ecef(0.0, 0.00, 2000.0)

    autopilot = FixedWingAutopilot([wp], target_speed=170.0)

    assert len(autopilot.waypoints) == 1
    assert autopilot.target_speeds.shape == (1,)
    assert autopilot.hold_active is False


def test_fixed_wing_autopilot_constructor_allows_empty_route():
    autopilot = FixedWingAutopilot([], target_speed=170.0)

    assert autopilot.waypoints == []
    assert autopilot.target_speeds.shape == (0,)
    assert autopilot.current_wp_idx == 0
    assert autopilot.completed is True
    assert autopilot.hold_active is False
    assert autopilot.hold_speed is None
    assert autopilot.hold_altitude is None
    assert autopilot.hold_horizontal_direction is None


def test_fixed_wing_autopilot_constructor_allows_empty_route_with_empty_target_speeds():
    autopilot = FixedWingAutopilot([], target_speed=170.0, target_speeds=[])

    assert autopilot.waypoints == []
    assert autopilot.target_speeds.shape == (0,)
    assert autopilot.completed is True
    assert autopilot.hold_active is False


def test_fixed_wing_autopilot_enter_hold_mode_uses_last_route_speed_and_current_flight_direction():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 20.0])
    wp1 = lla_to_ecef(0.0, 0.01, 2100.0)
    wp2 = lla_to_ecef(0.0, 0.02, 2200.0)

    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([wp1, wp2], target_speeds=[160.0, 210.0])
    expected_horizontal = vel - np.dot(vel, pos / np.linalg.norm(pos)) * (pos / np.linalg.norm(pos))
    expected_horizontal /= np.linalg.norm(expected_horizontal)

    autopilot._enter_hold_mode(mover)

    assert autopilot.hold_active is True
    assert autopilot.hold_speed == 210.0
    assert np.isclose(autopilot.hold_altitude, 2000.0, atol=1.0)
    assert np.allclose(autopilot.hold_horizontal_direction, expected_horizontal, atol=1e-7)


def test_fixed_wing_autopilot_enter_hold_mode_uses_default_speed_for_empty_route():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])

    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([], target_speed=190.0)
    expected_horizontal = vel / np.linalg.norm(vel)

    autopilot._enter_hold_mode(mover)

    assert autopilot.hold_active is True
    assert autopilot.hold_speed == 190.0
    assert np.isclose(autopilot.hold_altitude, 2000.0, atol=1.0)
    assert np.allclose(autopilot.hold_horizontal_direction, expected_horizontal, atol=1e-7)


def test_fixed_wing_autopilot_enter_hold_mode_falls_back_to_body_forward_when_horizontal_speed_is_small():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    local_up = pos / np.linalg.norm(pos)
    base_mover = FixedWingMover(pos, np.array([0.0, 180.0, 0.0]), use_coriolis=False)
    mover = FixedWingMover(
        pos,
        180.0 * local_up,
        initial_orientation=base_mover.orientation,
        use_coriolis=False,
    )
    autopilot = FixedWingAutopilot([], target_speed=190.0)
    expected_horizontal = mover.orientation[:, 0] - np.dot(mover.orientation[:, 0], local_up) * local_up
    expected_horizontal /= np.linalg.norm(expected_horizontal)

    autopilot._enter_hold_mode(mover)

    assert autopilot.hold_active is True
    assert np.allclose(autopilot.hold_horizontal_direction, expected_horizontal, atol=1e-7)


def test_fixed_wing_autopilot_hold_heading_error_is_zero_when_aligned():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([], target_speed=190.0)
    autopilot.hold_horizontal_direction = vel / np.linalg.norm(vel)

    assert np.isclose(
        autopilot._heading_error_to_direction(mover, autopilot.hold_horizontal_direction),
        0.0,
        atol=1e-9,
    )


def test_fixed_wing_autopilot_hold_heading_error_has_expected_sign():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([], target_speed=190.0)

    local_up = pos / np.linalg.norm(pos)
    current_horizontal = vel / np.linalg.norm(vel)
    left_turn_direction = np.cross(local_up, current_horizontal)
    left_turn_direction /= np.linalg.norm(left_turn_direction)

    autopilot.hold_horizontal_direction = left_turn_direction
    left_error = autopilot._heading_error_to_direction(mover, autopilot.hold_horizontal_direction)
    autopilot.hold_horizontal_direction = -left_turn_direction
    right_error = autopilot._heading_error_to_direction(mover, autopilot.hold_horizontal_direction)

    assert left_error > 0.0
    assert right_error < 0.0


def test_fixed_wing_autopilot_hold_heading_error_falls_back_to_projected_body_forward():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    local_up = pos / np.linalg.norm(pos)
    base_mover = FixedWingMover(pos, np.array([0.0, 180.0, 0.0]), use_coriolis=False)
    mover = FixedWingMover(
        pos,
        180.0 * local_up,
        initial_orientation=base_mover.orientation,
        use_coriolis=False,
    )
    autopilot = FixedWingAutopilot([], target_speed=190.0)
    autopilot.hold_horizontal_direction = base_mover.orientation[:, 0] - np.dot(base_mover.orientation[:, 0], local_up) * local_up
    autopilot.hold_horizontal_direction /= np.linalg.norm(autopilot.hold_horizontal_direction)

    assert np.isclose(
        autopilot._heading_error_to_direction(mover, autopilot.hold_horizontal_direction),
        0.0,
        atol=1e-9,
    )


def test_fixed_wing_autopilot_update_hold_mode_generates_commands_from_hold_targets():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([], target_speed=190.0)

    local_up = pos / np.linalg.norm(pos)
    current_horizontal = vel / np.linalg.norm(vel)
    left_turn_direction = np.cross(local_up, current_horizontal)
    left_turn_direction /= np.linalg.norm(left_turn_direction)

    autopilot.hold_active = True
    autopilot.hold_horizontal_direction = left_turn_direction
    autopilot.hold_altitude = 2400.0
    autopilot.hold_speed = 220.0

    autopilot._update_hold_mode(mover)

    assert mover.roll_cmd > 0.0
    assert mover.pitch_cmd > 0.0
    assert mover.thrust_cmd > 0.0
    assert mover.yaw_cmd == 0.0
    assert abs(mover.roll_cmd) <= 100.0
    assert abs(mover.pitch_cmd) <= 100.0
    assert 0.0 <= mover.thrust_cmd <= 100.0


def test_fixed_wing_autopilot_update_hold_mode_uses_hold_speed_target():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    low_speed_mover = FixedWingMover(pos, vel, use_coriolis=False)
    high_speed_mover = FixedWingMover(pos, vel, use_coriolis=False)
    low_speed_autopilot = FixedWingAutopilot([], target_speed=190.0)
    high_speed_autopilot = FixedWingAutopilot([], target_speed=190.0)

    aligned_direction = vel / np.linalg.norm(vel)
    for autopilot, hold_speed in ((low_speed_autopilot, 120.0), (high_speed_autopilot, 260.0)):
        autopilot.hold_active = True
        autopilot.hold_horizontal_direction = aligned_direction
        autopilot.hold_altitude = 2000.0
        autopilot.hold_speed = hold_speed

    low_speed_autopilot._update_hold_mode(low_speed_mover)
    high_speed_autopilot._update_hold_mode(high_speed_mover)

    assert np.isclose(low_speed_mover.roll_cmd, 0.0, atol=1e-9)
    assert np.isclose(high_speed_mover.roll_cmd, 0.0, atol=1e-9)
    assert low_speed_mover.thrust_cmd < high_speed_mover.thrust_cmd


def test_fixed_wing_autopilot_update_enters_hold_mode_for_empty_route():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([], target_speed=220.0)
    engine = SimulationEngine()
    engine.register_platform(Platform("fixed_wing_hold", mover, autopilot))

    autopilot.update(0.0, engine)

    assert autopilot.completed is True
    assert autopilot.hold_active is True
    assert autopilot.hold_speed == 220.0
    assert mover.thrust_cmd > 0.0
    assert mover.yaw_cmd == 0.0


def test_fixed_wing_autopilot_update_enters_hold_mode_after_final_waypoint():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    wp = pos
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([wp], target_speeds=[210.0], waypoint_radius=100.0)
    engine = SimulationEngine()
    engine.register_platform(Platform("fixed_wing_hold", mover, autopilot))

    autopilot.update(0.0, engine)

    assert autopilot.current_wp_idx == 1
    assert autopilot.completed is True
    assert autopilot.hold_active is True
    assert autopilot.hold_speed == 210.0
    assert mover.thrust_cmd > 0.0


def test_fixed_wing_autopilot_update_disables_hold_mode_when_waypoint_guidance_is_active():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    wp = lla_to_ecef(0.0, 0.02, 2200.0)
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([wp], target_speed=220.0)
    autopilot.hold_active = True
    autopilot.hold_speed = 150.0
    autopilot.hold_altitude = 1800.0
    autopilot.hold_horizontal_direction = np.array([1.0, 0.0, 0.0])
    engine = SimulationEngine()
    engine.register_platform(Platform("fixed_wing_route", mover, autopilot))

    autopilot.update(0.0, engine)

    assert autopilot.completed is False
    assert autopilot.hold_active is False


def test_fixed_wing_autopilot_update_does_not_advance_outside_radius():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    wp1 = lla_to_ecef(0.0, 0.02, 2000.0)
    wp2 = lla_to_ecef(0.0, 0.04, 2000.0)

    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([wp1, wp2], waypoint_radius=100.0)
    engine = SimulationEngine()
    engine.register_platform(Platform("fixed_wing_route", mover, autopilot))

    autopilot.update(0.0, engine)

    assert autopilot.current_wp_idx == 0
    assert autopilot.completed is False


def test_fixed_wing_autopilot_update_advances_to_first_route_leg():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    wp1 = pos
    wp2 = lla_to_ecef(0.0, 0.02, 2000.0)

    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([wp1, wp2], waypoint_radius=100.0)
    engine = SimulationEngine()
    events = []
    engine.broker.subscribe("waypoint_reached", lambda platform, idx: events.append((platform.id, idx)))
    engine.register_platform(Platform("fixed_wing_route", mover, autopilot))

    autopilot.update(0.0, engine)

    assert autopilot.current_wp_idx == 1
    assert autopilot.completed is False
    assert events == [("fixed_wing_route", 0)]


def test_fixed_wing_autopilot_update_advances_through_multiple_waypoints_and_completes():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    wp1 = pos
    wp2 = pos
    wp3 = pos

    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([wp1, wp2, wp3], waypoint_radius=100.0)
    engine = SimulationEngine()
    events = []
    engine.broker.subscribe("waypoint_reached", lambda platform, idx: events.append((platform.id, idx)))
    engine.register_platform(Platform("fixed_wing_route", mover, autopilot))

    autopilot.update(0.0, engine)

    assert autopilot.current_wp_idx == 3
    assert autopilot.completed is True
    assert events == [
        ("fixed_wing_route", 0),
        ("fixed_wing_route", 1),
        ("fixed_wing_route", 2),
    ]


def test_fixed_wing_autopilot_update_generates_guidance_commands():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    wp = lla_to_ecef(0.02, 0.0, 2400.0)

    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([wp], target_speed=220.0, waypoint_radius=100.0)
    engine = SimulationEngine()
    engine.register_platform(Platform("fixed_wing_route", mover, autopilot))

    autopilot.update(0.0, engine)

    assert abs(mover.roll_cmd) > 0.0
    assert mover.pitch_cmd > 0.0
    assert mover.thrust_cmd > 0.0
    assert mover.yaw_cmd == 0.0
    assert abs(mover.roll_cmd) <= 100.0
    assert abs(mover.pitch_cmd) <= 100.0
    assert 0.0 <= mover.thrust_cmd <= 100.0


def test_fixed_wing_autopilot_update_uses_active_waypoint_speed_target_after_advancement():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    wp1 = pos
    wp2 = lla_to_ecef(0.0, 0.02, 2000.0)

    low_speed_mover = FixedWingMover(pos, vel, use_coriolis=False)
    high_speed_mover = FixedWingMover(pos, vel, use_coriolis=False)
    low_speed_autopilot = FixedWingAutopilot([wp1, wp2], target_speeds=[220.0, 100.0], waypoint_radius=100.0)
    high_speed_autopilot = FixedWingAutopilot([wp1, wp2], target_speeds=[220.0, 260.0], waypoint_radius=100.0)

    low_speed_engine = SimulationEngine()
    high_speed_engine = SimulationEngine()
    low_speed_engine.register_platform(Platform("fixed_wing_low", low_speed_mover, low_speed_autopilot))
    high_speed_engine.register_platform(Platform("fixed_wing_high", high_speed_mover, high_speed_autopilot))

    low_speed_autopilot.update(0.0, low_speed_engine)
    high_speed_autopilot.update(0.0, high_speed_engine)

    assert low_speed_autopilot.current_wp_idx == 1
    assert high_speed_autopilot.current_wp_idx == 1
    assert low_speed_mover.thrust_cmd < high_speed_mover.thrust_cmd


def test_fixed_wing_autopilot_update_uses_hold_mode_when_already_completed():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    wp = lla_to_ecef(0.0, 0.02, 2000.0)

    mover = FixedWingMover(pos, vel, use_coriolis=False)
    mover.thrust_cmd = 50.0
    mover.roll_cmd = 25.0
    mover.pitch_cmd = -10.0
    mover.yaw_cmd = 5.0
    autopilot = FixedWingAutopilot([wp], target_speed=220.0, waypoint_radius=100.0)
    autopilot.current_wp_idx = len(autopilot.waypoints)
    engine = SimulationEngine()
    engine.register_platform(Platform("fixed_wing_route", mover, autopilot))

    autopilot.update(0.0, engine)

    assert autopilot.completed is True
    assert autopilot.hold_active is True
    assert autopilot.hold_speed == 220.0
    assert mover.thrust_cmd > 0.0
    assert mover.yaw_cmd == 0.0


def test_fixed_wing_autopilot_empty_route_accelerates_toward_target_speed():
    engine = SimulationEngine()
    engine.max_step = 0.01

    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([], target_speed=220.0, update_interval=0.05)
    initial_speed = np.linalg.norm(mover.velocity)

    engine.register_platform(Platform("fixed_wing_hold", mover, autopilot))
    engine.run(1.0)

    assert autopilot.completed is True
    assert autopilot.hold_active is True
    assert mover.thrust_cmd > 0.0
    assert np.linalg.norm(mover.velocity) > initial_speed


def test_fixed_wing_autopilot_completed_route_maintains_hold_commands_during_run():
    engine = SimulationEngine()
    engine.max_step = 0.01

    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)
    autopilot = FixedWingAutopilot([pos], target_speeds=[210.0], waypoint_radius=100.0, update_interval=0.05)

    engine.register_platform(Platform("fixed_wing_hold", mover, autopilot))
    engine.run(1.0)

    assert autopilot.current_wp_idx == 1
    assert autopilot.completed is True
    assert autopilot.hold_active is True
    assert autopilot.hold_speed == 210.0
    assert mover.thrust_cmd > 0.0
    assert mover.yaw_cmd == 0.0


def test_fixed_wing_autopilot_empty_route_preserves_heading_and_altitude_better_than_zero_commands():
    initial_pos = lla_to_ecef(0.0, 0.0, 2000.0)
    initial_vel = np.array([0.0, 180.0, 0.0])
    base_mover = FixedWingMover(initial_pos, initial_vel, use_coriolis=False)
    disturbed_orientation = (
        base_mover.orientation
        @ _rotation_about_body_forward(np.radians(10.0))
        @ _rotation_about_body_right(np.radians(5.0))
    )
    initial_local_up = initial_pos / np.linalg.norm(initial_pos)
    initial_horizontal = initial_vel - np.dot(initial_vel, initial_local_up) * initial_local_up
    initial_horizontal /= np.linalg.norm(initial_horizontal)

    def run_case(controller):
        engine = SimulationEngine()
        engine.max_step = 0.01
        mover = FixedWingMover(
            initial_pos,
            initial_vel,
            initial_orientation=disturbed_orientation,
            use_coriolis=False,
        )
        engine.register_platform(Platform("fixed_wing_case", mover, controller))
        engine.run(3.0)

        local_up = mover.position / np.linalg.norm(mover.position)
        horizontal_velocity = mover.velocity - np.dot(mover.velocity, local_up) * local_up
        horizontal_velocity /= np.linalg.norm(horizontal_velocity)
        heading_error = abs(np.arctan2(
            np.dot(np.cross(horizontal_velocity, initial_horizontal), local_up),
            np.clip(np.dot(horizontal_velocity, initial_horizontal), -1.0, 1.0),
        ))
        altitude_error = abs(ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])[2] - 2000.0)
        return heading_error, altitude_error

    hold_heading_error, hold_altitude_error = run_case(
        FixedWingAutopilot([], target_speed=180.0, update_interval=0.05)
    )
    free_heading_error, free_altitude_error = run_case(None)

    assert hold_heading_error < free_heading_error
    assert hold_altitude_error < free_altitude_error


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"waypoints": [np.zeros(2)]}, r"shape \(3,\)"),
        ({"waypoints": [np.zeros(3)], "target_speed": -1.0}, "target_speed"),
        ({"waypoints": [np.zeros(3)], "waypoint_radius": 0.0}, "waypoint_radius"),
        ({"waypoints": [np.zeros(3)], "max_climb_rate": -1.0}, "max_climb_rate"),
        ({"waypoints": [np.zeros(3), np.ones(3)], "target_speeds": [100.0]}, r"shape \(2,\)"),
        ({"waypoints": [np.zeros(3), np.ones(3)], "target_speeds": [100.0, -5.0]}, "non-negative"),
        ({"waypoints": [], "target_speeds": [100.0]}, r"shape \(0,\)"),
    ],
)
def test_fixed_wing_autopilot_constructor_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FixedWingAutopilot(**kwargs)


def test_fixed_wing_aerodynamic_force_generates_lift_for_positive_alpha():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    pitched_orientation = mover.orientation @ _rotation_about_body_right(np.radians(10.0))
    aero_force = mover._aerodynamic_force(pos, vel, pitched_orientation)

    assert np.dot(aero_force, pitched_orientation[:, 2]) > 0.0
    assert np.dot(aero_force, pitched_orientation[:, 0]) < 0.0


def test_fixed_wing_aerodynamic_force_opposes_positive_sideslip():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    yawed_orientation = mover.orientation @ _rotation_about_body_up(np.radians(-10.0))
    aero_force = mover._aerodynamic_force(pos, vel, yawed_orientation)
    lateral_velocity_body = yawed_orientation.T @ vel

    assert lateral_velocity_body[1] > 0.0
    assert np.dot(aero_force, yawed_orientation[:, 1]) < 0.0


def test_fixed_wing_restoring_moment_opposes_positive_bank_error():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    banked_orientation = mover.orientation @ _rotation_about_body_forward(np.radians(10.0))
    restoring = mover._restoring_moment_components(pos, vel, banked_orientation, np.zeros(3))

    assert restoring[0] < 0.0


def test_fixed_wing_restoring_moment_opposes_alpha_beta_and_rates():
    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = FixedWingMover(pos, vel, use_coriolis=False)

    pitched_orientation = mover.orientation @ _rotation_about_body_right(np.radians(10.0))
    alpha_restoring = mover._restoring_moment_components(pos, vel, pitched_orientation, np.zeros(3))
    assert alpha_restoring[1] < 0.0

    yawed_orientation = mover.orientation @ _rotation_about_body_up(np.radians(-10.0))
    beta_restoring = mover._restoring_moment_components(pos, vel, yawed_orientation, np.zeros(3))
    assert beta_restoring[2] > 0.0

    damping = mover._restoring_moment_components(pos, vel, mover.orientation, np.array([0.2, -0.3, 0.4]))
    assert damping[0] < 0.0
    assert damping[1] < 0.0
    assert damping[2] < 0.0


def test_rocket_mover_constructor_stores_expected_state_layout():
    pos = lla_to_ecef(0.0, 0.0, 1500.0)
    vel = np.array([0.0, 220.0, 15.0])
    stages = [
        {
            "dry_mass": 1200.0,
            "propellant_mass": 800.0,
            "reference_area": 1.8,
            "drag_coefficient": 0.14,
            "rotational_mass": np.diag([2500.0, 8000.0, 8000.0]),
            "angular_damping": np.array([4000.0, 7000.0, 7000.0]),
            "max_thrust": 160000.0,
            "max_steering_moment": 18000.0,
            "mass_flow_rate": 20.0,
            "thrust": 150000.0,
        }
    ]

    mover = RocketMover(
        pos,
        vel,
        stages=stages,
        mass=300.0,
        area=2.2,
        cd0=0.1,
        use_coriolis=False,
    )

    assert mover.get_state_dimension() == 19
    assert mover.get_state().shape == (19,)
    assert mover.orientation.shape == (3, 3)
    assert mover.omega_body.shape == (3,)
    assert np.isclose(mover.propellant_mass, 800.0)
    assert mover.get_propellant_mass_slice() == slice(18, 19)
    assert mover.active_stage_index == 0
    assert np.isclose(mover.max_thrust, 160000.0)
    assert np.isclose(mover.max_steering_moment, 18000.0)
    assert np.isclose(mover.area, 1.8)
    assert np.isclose(mover.cd0, 0.14)
    assert np.isclose(mover.current_total_mass(), 2300.0)


def test_rocket_controller_defaults_to_boost_vertical_phase():
    controller = RocketController()

    assert controller.phase == RocketController.BOOST_VERTICAL
    assert controller.phase_start_time is None


def test_rocket_controller_rejects_invalid_initial_phase():
    with pytest.raises(ValueError, match="initial_phase"):
        RocketController(initial_phase="invalid")


def test_rocket_controller_initialize_and_update_hold_safe_zero_commands():
    engine = SimulationEngine()

    pos = lla_to_ecef(0.0, 0.0, 1500.0)
    vel = np.array([0.0, 0.0, 0.0])
    mover = RocketMover(pos, vel, use_coriolis=False)
    controller = RocketController(update_interval=0.1)
    platform = Platform("rocket_controller", mover, controller)
    engine.register_platform(platform)

    controller.initialize(engine)
    assert controller.phase_start_time == engine.t

    mover.thrust_cmd = 65.0
    mover.steer_cmd = 40.0
    controller.update(engine.t, engine)

    assert mover.thrust_cmd == 0.0
    assert mover.steer_cmd == 0.0


def test_rocket_orientation_correction_event_projects_committed_state():
    engine = SimulationEngine()

    pos = lla_to_ecef(0.0, 0.0, 2000.0)
    vel = np.array([0.0, 250.0, 0.0])
    mover = RocketMover(pos, vel, use_coriolis=False)
    platform = Platform("rocket_event", mover)
    engine.register_platform(platform)

    mover_slice = engine.context.get_state_slice(mover)
    orientation_slice = mover.get_orientation_slice()
    start = mover_slice.start + orientation_slice.start
    stop = mover_slice.start + orientation_slice.stop

    engine.context.committed_y[start:stop] = np.array([
        1.0, 0.08, 0.0,
        -0.02, 0.97, -0.06,
        0.04, 0.03, 1.03,
    ])

    event = mover.add_orientation_correction_event(engine)
    event.fix_orientation(engine)

    corrected = engine.context.committed_y[start:stop].reshape((3, 3))
    assert np.allclose(corrected.T @ corrected, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(corrected), 1.0, atol=1e-7)


def test_rocket_propellant_mass_decreases_during_burn():
    engine = SimulationEngine()
    engine.max_step = 0.02

    pos = lla_to_ecef(0.0, 0.0, 1500.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = RocketMover(
        pos,
        vel,
        stages=[
            {
                "dry_mass": 800.0,
                "propellant_mass": 50.0,
                "reference_area": 1.0,
                "drag_coefficient": 0.0,
                "rotational_mass": np.diag([1500.0, 4000.0, 4000.0]),
                "angular_damping": np.array([1000.0, 2000.0, 2000.0]),
                "max_thrust": 50000.0,
                "max_steering_moment": 5000.0,
                "mass_flow_rate": 20.0,
                "thrust": 50000.0,
            }
        ],
        mass=200.0,
        normal_force_coefficient=0.0,
        use_coriolis=False,
    )
    mover.thrust_cmd = 100.0

    engine.register_platform(Platform("rocket_burn", mover))
    engine.run(1.5)

    assert np.isclose(mover.propellant_mass, 20.0, atol=1e-2)
    assert np.isclose(mover.current_total_mass(), 1020.0, atol=1e-2)


def test_rocket_forward_acceleration_increases_as_mass_drops_under_fixed_thrust():
    pos = lla_to_ecef(0.0, 0.0, 1500.0)
    mover = RocketMover(
        pos,
        np.zeros(3),
        initial_orientation=np.eye(3),
        stages=[
            {
                "dry_mass": 400.0,
                "propellant_mass": 100.0,
                "reference_area": 1.0,
                "drag_coefficient": 0.0,
                "rotational_mass": np.diag([1200.0, 3000.0, 3000.0]),
                "angular_damping": np.array([500.0, 1000.0, 1000.0]),
                "max_thrust": 12000.0,
                "max_steering_moment": 2000.0,
                "mass_flow_rate": 10.0,
                "thrust": 12000.0,
            }
        ],
        mass=100.0,
        normal_force_coefficient=0.0,
        use_coriolis=False,
    )
    mover.thrust_cmd = 100.0

    heavy_state = mover.get_initial_state().copy()
    light_state = mover.get_initial_state().copy()
    heavy_state[mover.get_propellant_mass_slice()] = 100.0
    light_state[mover.get_propellant_mass_slice()] = 20.0

    heavy_dvel = mover.compute_state_derivative(0.0, heavy_state)[mover.get_velocity_slice()]
    light_dvel = mover.compute_state_derivative(0.0, light_state)[mover.get_velocity_slice()]

    assert light_dvel[0] > heavy_dvel[0]


def test_rocket_stage_thrust_and_mass_flow_drop_to_zero_at_burnout():
    pos = lla_to_ecef(0.0, 0.0, 1500.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = RocketMover(
        pos,
        vel,
        stages=[
            {
                "dry_mass": 700.0,
                "propellant_mass": 60.0,
                "reference_area": 1.2,
                "drag_coefficient": 0.08,
                "rotational_mass": np.diag([1400.0, 3500.0, 3500.0]),
                "angular_damping": np.array([700.0, 1300.0, 1300.0]),
                "max_thrust": 80000.0,
                "max_steering_moment": 4000.0,
                "mass_flow_rate": 12.0,
                "thrust": 60000.0,
            }
        ],
        mass=150.0,
        use_coriolis=False,
    )
    mover.thrust_cmd = 50.0

    assert np.isclose(mover.current_stage_thrust(0.0, propellant_mass=10.0), 30000.0)
    assert np.isclose(mover.current_mass_flow_rate(0.0, propellant_mass=10.0), 6.0)
    assert mover.has_active_burn(propellant_mass=10.0) is True

    assert mover.current_stage_thrust(0.0, propellant_mass=0.0) == 0.0
    assert mover.current_mass_flow_rate(0.0, propellant_mass=0.0) == 0.0
    assert mover.has_active_burn(propellant_mass=0.0) is False


def test_rocket_stage_separation_updates_stage_properties_and_propellant_state():
    engine = SimulationEngine()

    pos = lla_to_ecef(0.0, 0.0, 1500.0)
    vel = np.array([0.0, 180.0, 0.0])
    mover = RocketMover(
        pos,
        vel,
        stages=[
            {
                "dry_mass": 700.0,
                "propellant_mass": 40.0,
                "reference_area": 1.5,
                "drag_coefficient": 0.12,
                "rotational_mass": np.diag([1800.0, 5000.0, 5000.0]),
                "angular_damping": np.array([900.0, 1600.0, 1600.0]),
                "max_thrust": 90000.0,
                "max_steering_moment": 7000.0,
                "mass_flow_rate": 10.0,
                "thrust": 85000.0,
            },
            {
                "dry_mass": 300.0,
                "propellant_mass": 20.0,
                "reference_area": 0.9,
                "drag_coefficient": 0.2,
                "rotational_mass": np.diag([900.0, 2200.0, 2200.0]),
                "angular_damping": np.array([400.0, 700.0, 700.0]),
                "max_thrust": 30000.0,
                "max_steering_moment": 2500.0,
                "mass_flow_rate": 5.0,
                "thrust": 28000.0,
            },
        ],
        mass=100.0,
        use_coriolis=False,
    )
    engine.register_platform(Platform("rocket_stage_sep", mover))

    mover._set_propellant_mass_state_in_engine(0.0, engine)
    assert mover.can_separate_stage() is True

    next_stage = mover.separate_stage(engine)

    mover_slice = engine.context.get_state_slice(mover)
    propellant_state = engine.context.committed_y[mover_slice][mover.get_propellant_mass_slice()][0]

    assert next_stage is mover.stages[1]
    assert mover.active_stage_index == 1
    assert np.isclose(mover.propellant_mass, 20.0)
    assert np.isclose(propellant_state, 20.0)
    assert np.isclose(mover.area, 0.9)
    assert np.isclose(mover.cd0, 0.2)
    assert np.isclose(mover.max_thrust, 30000.0)
    assert np.isclose(mover.max_steering_moment, 2500.0)
    assert np.allclose(mover.angular_damping, [400.0, 700.0, 700.0])
    assert np.allclose(mover.rotational_mass, np.diag([900.0, 2200.0, 2200.0]))
    assert np.isclose(mover.current_total_mass(), 420.0)
    assert mover.can_separate_stage() is False


def test_rocket_equal_transverse_steering_commands_have_equal_response_magnitude():
    pos = lla_to_ecef(0.0, 0.0, 1500.0)
    mover = RocketMover(
        pos,
        np.zeros(3),
        initial_orientation=np.eye(3),
        rotational_mass=np.diag([1200.0, 3600.0, 3600.0]),
        angular_damping=np.array([0.0, 0.0, 0.0]),
        normal_force_coefficient=0.0,
        alignment_restoring_coefficient=0.0,
        use_coriolis=False,
    )
    mover.steer_cmd = 60.0

    mover.steer_direction_body = np.array([0.0, 1.0, 0.0])
    y_response = mover._angular_acceleration_body(pos, np.zeros(3), np.eye(3), np.zeros(3))

    mover.steer_direction_body = np.array([0.0, 0.0, 1.0])
    z_response = mover._angular_acceleration_body(pos, np.zeros(3), np.eye(3), np.zeros(3))

    assert np.isclose(y_response[0], 0.0, atol=1e-12)
    assert np.isclose(z_response[0], 0.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(y_response), np.linalg.norm(z_response), atol=1e-12)
    assert np.isclose(abs(y_response[1]), abs(z_response[2]), atol=1e-12)
