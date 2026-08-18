import os
import pytest
import numpy as np
from mover_sim.core.platform import Platform
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import CSVLogger
from mover_sim.models.aircraft_mover import AircraftMover, AircraftAutopilot
from mover_sim.math.coordinates import lla_to_ecef, ecef_to_lla

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
