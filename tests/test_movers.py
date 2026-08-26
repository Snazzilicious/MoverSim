import pytest
import numpy as np
from mover_sim.core.platform import Platform
from mover_sim.core.mover import TranslationalNewtonianMover
from mover_sim.core.controller import Controller
from mover_sim.core.engine import SimulationEngine
from mover_sim.models.spline_mover import WaypointMover, SplineMover


class TimeStub:
    def __init__(self, t=0.0):
        self.t = t

    def get_time(self):
        return self.t

def test_waypoint_mover():
    times = [0.0, 10.0, 20.0]
    positions = [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [100.0, 200.0, 0.0]
    ]
    mover = WaypointMover(times, positions)
    context = TimeStub()
    mover._context = context
    
    # Test before start
    context.t = -5.0
    state = mover.get_state()
    assert np.allclose(state[:3], [0.0, 0.0, 0.0])
    assert np.allclose(state[3:], [0.0, 0.0, 0.0])
    
    # Test at exactly waypoint 1
    context.t = 10.0
    state = mover.get_state()
    assert np.allclose(state[:3], [100.0, 0.0, 0.0])
    
    # Test intermediate linear position and velocity
    context.t = 5.0
    state = mover.get_state()
    assert np.allclose(state[:3], [50.0, 0.0, 0.0])
    assert np.allclose(state[3:], [10.0, 0.0, 0.0]) # 100 meters / 10 seconds
    
    # Test second segment intermediate
    context.t = 15.0
    state = mover.get_state()
    assert np.allclose(state[:3], [100.0, 100.0, 0.0])
    assert np.allclose(state[3:], [0.0, 20.0, 0.0]) # 200 meters / 10 seconds

    # Test past end
    context.t = 25.0
    state = mover.get_state()
    assert np.allclose(state[:3], [100.0, 200.0, 0.0])
    assert np.allclose(state[3:], [0.0, 0.0, 0.0])

def test_spline_mover():
    times = [0.0, 5.0, 10.0]
    positions = [
        [0.0, 0.0, 0.0],
        [10.0, 50.0, 0.0],
        [20.0, 100.0, 0.0]
    ]
    mover = SplineMover(times, positions)
    context = TimeStub()
    mover._context = context
    
    # Test boundary condition clamped (vel=0 at endpoints)
    context.t = 0.0
    state_start = mover.get_state()
    assert np.allclose(state_start[:3], [0.0, 0.0, 0.0])
    assert np.allclose(state_start[3:], [0.0, 0.0, 0.0])

    context.t = 5.0
    state_mid = mover.get_state()
    assert np.allclose(state_mid[:3], [10.0, 50.0, 0.0])
    # Spline should have non-zero velocity in the middle
    assert not np.allclose(state_mid[3:], [0.0, 0.0, 0.0])


def test_translational_analytical_movers_preserve_public_api():
    times = [0.0, 10.0, 20.0]
    waypoint_positions = [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [100.0, 200.0, 0.0],
    ]
    spline_positions = [
        [0.0, 0.0, 0.0],
        [10.0, 50.0, 0.0],
        [20.0, 100.0, 0.0],
    ]

    waypoint = WaypointMover(times, waypoint_positions)
    spline = SplineMover([0.0, 5.0, 10.0], spline_positions)

    context = TimeStub(5.0)
    waypoint._context = context
    spline._context = context

    waypoint_state = waypoint.get_state()
    spline_state = spline.get_state()

    assert waypoint.get_state_dimension() == 6
    assert np.allclose(waypoint.position, waypoint_state[:3])
    assert np.allclose(waypoint.velocity, waypoint_state[3:])

    assert spline.get_state_dimension() == 6
    assert np.allclose(spline.position, spline_state[:3])
    assert np.allclose(spline.velocity, spline_state[3:])

def test_newtonian_mover_integration():
    engine = SimulationEngine()
    
    # Newtonian mover with pos=[0,0,0], vel=[10, 20, 30] (constant velocity)
    mover = TranslationalNewtonianMover([0.0, 0.0, 0.0], [10.0, 20.0, 30.0])
    platform = Platform("test_vehicle", mover)
    engine.register_platform(platform)
    
    # Run simulation for 10 seconds
    engine.run(10.0)
    
    # With constant velocity, pos should be [100, 200, 300] after 10s
    assert np.allclose(mover.position, [100.0, 200.0, 300.0])
    assert np.allclose(mover.velocity, [10.0, 20.0, 30.0])
    assert np.isclose(engine.t, 10.0)


def test_translational_newtonian_mover_preserves_public_api():
    mover = TranslationalNewtonianMover([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    assert mover.get_state_dimension() == 6
    assert np.allclose(mover.get_state(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(mover.position, [1.0, 2.0, 3.0])
    assert np.allclose(mover.velocity, [4.0, 5.0, 6.0])

def test_controller_execution():
    engine = SimulationEngine()
    
    # Mock Newtonian mover where velocity can be modified by controller
    class ControllableMover(TranslationalNewtonianMover):
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
