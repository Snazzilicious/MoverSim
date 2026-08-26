import os
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
)
from mover_sim.math.coordinates import lla_to_ecef, ecef_to_enu, ecef_to_lla
from mover_sim.math.orientation import rotate_vector_by_quaternion

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
        lines = f.readlines()
        
    # Expect: Header + initial log (0.0) + log at 0.5 + log at 1.0 + log at 1.5
    assert len(lines) >= 4
    
    # Verify header columns
    header = lines[0].strip().split(",")
    assert header[0] == "time"
    assert "test_plane_x" in header
    assert "test_plane_lat" in header
    assert "test_plane_vx" in header
    
    # Verify first row time
    first_row = lines[1].strip().split(",")
    assert np.isclose(float(first_row[0]), 0.0)


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
