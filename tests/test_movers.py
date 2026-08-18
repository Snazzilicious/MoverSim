import pytest
import numpy as np
from mover_sim.core.platform import Platform
from mover_sim.core.mover import NewtonianMover
from mover_sim.core.controller import Controller
from mover_sim.core.engine import SimulationEngine
from mover_sim.models.spline_mover import WaypointMover, SplineMover

def test_waypoint_mover():
    times = [0.0, 10.0, 20.0]
    positions = [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [100.0, 200.0, 0.0]
    ]
    mover = WaypointMover(times, positions)
    
    # Test before start
    pos, vel = mover.get_state_at(-5.0)
    assert np.allclose(pos, [0.0, 0.0, 0.0])
    assert np.allclose(vel, [0.0, 0.0, 0.0])
    
    # Test at exactly waypoint 1
    pos, vel = mover.get_state_at(10.0)
    assert np.allclose(pos, [100.0, 0.0, 0.0])
    
    # Test intermediate linear position and velocity
    pos, vel = mover.get_state_at(5.0)
    assert np.allclose(pos, [50.0, 0.0, 0.0])
    assert np.allclose(vel, [10.0, 0.0, 0.0]) # 100 meters / 10 seconds
    
    # Test second segment intermediate
    pos, vel = mover.get_state_at(15.0)
    assert np.allclose(pos, [100.0, 100.0, 0.0])
    assert np.allclose(vel, [0.0, 20.0, 0.0]) # 200 meters / 10 seconds

    # Test past end
    pos, vel = mover.get_state_at(25.0)
    assert np.allclose(pos, [100.0, 200.0, 0.0])
    assert np.allclose(vel, [0.0, 0.0, 0.0])

def test_spline_mover():
    times = [0.0, 5.0, 10.0]
    positions = [
        [0.0, 0.0, 0.0],
        [10.0, 50.0, 0.0],
        [20.0, 100.0, 0.0]
    ]
    mover = SplineMover(times, positions)
    
    # Test boundary condition clamped (vel=0 at endpoints)
    pos_start, vel_start = mover.get_state_at(0.0)
    assert np.allclose(pos_start, [0.0, 0.0, 0.0])
    assert np.allclose(vel_start, [0.0, 0.0, 0.0])

    pos_mid, vel_mid = mover.get_state_at(5.0)
    assert np.allclose(pos_mid, [10.0, 50.0, 0.0])
    # Spline should have non-zero velocity in the middle
    assert not np.allclose(vel_mid, [0.0, 0.0, 0.0])

def test_newtonian_mover_integration():
    engine = SimulationEngine()
    
    # Newtonian mover with pos=[0,0,0], vel=[10, 20, 30] (constant velocity)
    mover = NewtonianMover([0.0, 0.0, 0.0], [10.0, 20.0, 30.0])
    platform = Platform("test_vehicle", mover)
    engine.register_platform(platform)
    
    # Run simulation for 10 seconds
    engine.run(10.0)
    
    # With constant velocity, pos should be [100, 200, 300] after 10s
    assert np.allclose(mover.position, [100.0, 200.0, 300.0])
    assert np.allclose(mover.velocity, [10.0, 20.0, 30.0])
    assert np.isclose(engine.t, 10.0)

def test_controller_execution():
    engine = SimulationEngine()
    
    # Mock Newtonian mover where velocity can be modified by controller
    class ControllableMover(NewtonianMover):
        def __init__(self, pos, vel):
            super().__init__(pos, vel)
            self.acceleration = np.zeros(3)
            
        def compute_derivatives(self, t, pos, vel):
            return vel, self.acceleration
            
    # Controller that doubles acceleration at each step
    class AccelerationController(Controller):
        def update(self, t, eng):
            # Apply constant acceleration
            self.platform.mover.acceleration = np.array([1.0, 0.0, 0.0])
            
    mover = ControllableMover([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    # Run controller every 1.0 seconds
    controller = AccelerationController(update_interval=1.0)
    platform = Platform("missile", mover, controller)
    engine.register_platform(platform)
    
    # Run for 2.0 seconds
    # t=0: controller initializes and sets acceleration = [1,0,0]
    # t=[0, 2.0]: integrates with accel=[1,0,0]
    # x(t) = 0.5 * a * t^2 = 0.5 * 1.0 * 4.0 = 2.0
    # v(t) = a * t = 1.0 * 2.0 = 2.0
    engine.run(2.0)
    
    assert np.allclose(mover.position, [2.0, 0.0, 0.0])
    assert np.allclose(mover.velocity, [2.0, 0.0, 0.0])
