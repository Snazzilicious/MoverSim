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
    assert autopilot.completed is False
    assert autopilot.hold_active is False
    assert autopilot.hold_speed is None
    assert autopilot.hold_altitude is None
    assert autopilot.hold_horizontal_direction is None


def test_fixed_wing_autopilot_constructor_allows_empty_route_with_empty_target_speeds():
    autopilot = FixedWingAutopilot([], target_speed=170.0, target_speeds=[])

    assert autopilot.waypoints == []
    assert autopilot.target_speeds.shape == (0,)
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

    autopilot._enter_hold_mode(mover)

    assert autopilot.hold_active is True
    assert autopilot.hold_speed == 190.0
    assert np.isclose(autopilot.hold_altitude, 2000.0, atol=1.0)


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


def test_fixed_wing_autopilot_update_zeroes_commands_when_completed():
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
    assert mover.thrust_cmd == 0.0
    assert mover.roll_cmd == 0.0
    assert mover.pitch_cmd == 0.0
    assert mover.yaw_cmd == 0.0


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
