import pytest
import numpy as np
from mover_sim.math.physics import (
    gravity,
    coriolis_acceleration,
    air_density,
    aerodynamic_drag_force,
    GM,
    OMEGA_E
)
from mover_sim.core.platform import Platform
from mover_sim.core.mover import TranslationalIntegratedMover
from mover_sim.core.engine import SimulationEngine

def test_gravity_magnitude_and_direction():
    # Test at Earth's surface (radius ~ 6378 km)
    pos = np.array([6378137.0, 0.0, 0.0])
    g_vec = gravity(pos)
    
    # Gravity points towards the center of the Earth: [-g, 0, 0]
    assert g_vec[0] < 0
    assert np.isclose(g_vec[1], 0.0, atol=1e-7)
    assert np.isclose(g_vec[2], 0.0, atol=1e-7)
    
    # Magnitude at equator should be approx 9.798 m/s^2 (using GM/r^2)
    expected_mag = GM / (6378137.0 ** 2)
    assert np.isclose(np.linalg.norm(g_vec), expected_mag, atol=1e-3)

def test_coriolis_acceleration():
    # Earth rotation vector in ECEF is [0, 0, OMEGA_E]
    # For a platform moving East at the Prime Meridian: velocity is [0, 100.0, 0] m/s
    vel = np.array([0.0, 100.0, 0.0])
    a_cor = coriolis_acceleration(vel)
    
    # a_cor = [2 * OMEGA_E * vy, -2 * OMEGA_E * vx, 0]
    expected_ax = 2.0 * OMEGA_E * 100.0
    assert np.isclose(a_cor[0], expected_ax, atol=1e-9)
    assert np.isclose(a_cor[1], 0.0, atol=1e-9)
    assert np.isclose(a_cor[2], 0.0, atol=1e-9)
    
    # Zero velocity should yield zero acceleration
    assert np.allclose(coriolis_acceleration(np.zeros(3)), np.zeros(3))

def test_air_density():
    # Sea-level density
    assert np.isclose(air_density(0.0), 1.225)
    
    # At scale height (8500 m)
    assert np.isclose(air_density(8500.0), 1.225 * np.exp(-1.0))
    
    # Negative altitude should be clamped to sea-level density
    assert np.isclose(air_density(-100.0), 1.225)

def test_aerodynamic_drag():
    vel = np.array([100.0, 0.0, 0.0])
    alt = 0.0 # sea level, rho = 1.225
    cd = 0.5
    area = 10.0
    
    drag = aerodynamic_drag_force(vel, alt, cd, area)
    
    # Drag acts opposite to velocity
    assert drag[0] < 0
    assert np.isclose(drag[1], 0.0, atol=1e-9)
    assert np.isclose(drag[2], 0.0, atol=1e-9)
    
    # F_drag_mag = 0.5 * 1.225 * 0.5 * 10.0 * 100^2 = 30625 N
    expected_mag = 0.5 * 1.225 * cd * area * (100.0 ** 2)
    assert np.isclose(np.linalg.norm(drag), expected_mag, atol=1e-3)
    
    # Test with wind
    # Platform velocity: [100, 0, 0], Wind velocity: [50, 0, 0] (tailwind)
    # Relative velocity: [50, 0, 0]
    # F_drag_mag = 0.5 * 1.225 * 0.5 * 10.0 * 50^2 = 7656.25 N
    drag_wind = aerodynamic_drag_force(vel, alt, cd, area, wind_ecef=np.array([50.0, 0.0, 0.0]))
    assert np.isclose(np.linalg.norm(drag_wind), 0.5 * 1.225 * cd * area * (50.0 ** 2), atol=1e-3)

def test_integrated_mover_with_gravity():
    engine = SimulationEngine()
    engine.max_step = 0.1

    class FallingRockMover(TranslationalIntegratedMover):
        def compute_derivatives(self, t, pos, vel):
            dpos, dvel = super().compute_derivatives(t, pos, vel)
            return dpos, dvel + gravity(pos)
    
    # Drop an object from 10,000 meters altitude above the equator
    # Equator surface ECEF: [6378137.0, 0.0, 0.0]
    initial_pos = np.array([6378137.0 + 10000.0, 0.0, 0.0])
    mover = FallingRockMover(initial_pos, np.zeros(3))
    platform = Platform("falling_rock", mover)
    engine.register_platform(platform)
    
    # Run for 2.0 seconds
    engine.run(2.0)
    
    # Object should fall towards the earth center (X coordinate decreases)
    assert mover.position[0] < initial_pos[0]
    assert mover.velocity[0] < 0.0  # Velocity should point inwards (negative X)
    assert np.isclose(mover.position[1], 0.0, atol=1e-7)
    assert np.isclose(mover.position[2], 0.0, atol=1e-7)


def test_integrated_mover_has_no_implicit_gravity():
    engine = SimulationEngine()
    engine.max_step = 0.1

    initial_pos = np.array([6378137.0 + 10000.0, 0.0, 0.0])
    mover = TranslationalIntegratedMover(initial_pos, np.zeros(3))
    platform = Platform("coasting_rock", mover)
    engine.register_platform(platform)

    engine.run(2.0)

    assert np.allclose(mover.position, initial_pos)
    assert np.allclose(mover.velocity, np.zeros(3))


def test_integrated_subclass_can_use_fully_custom_dynamics_without_gravity():
    engine = SimulationEngine()
    engine.max_step = 0.05

    class CustomAccelerationMover(TranslationalIntegratedMover):
        def __init__(self, pos, vel, acceleration):
            super().__init__(pos, vel)
            self.acceleration = np.asarray(acceleration, dtype=float)

        def compute_derivatives(self, t, pos, vel):
            dpos, dvel = super().compute_derivatives(t, pos, vel)
            return dpos, dvel + self.acceleration

    initial_pos = np.array([6378137.0 + 10000.0, 1000.0, -500.0])
    initial_vel = np.array([10.0, -4.0, 2.0])
    acceleration = np.array([1.5, -0.5, 0.25])
    mover = CustomAccelerationMover(initial_pos, initial_vel, acceleration)
    engine.register_platform(Platform("custom_dynamics", mover))

    duration = 2.0
    engine.run(duration)

    expected_position = initial_pos + initial_vel * duration + 0.5 * acceleration * duration**2
    expected_velocity = initial_vel + acceleration * duration

    assert np.allclose(mover.position, expected_position, atol=1e-6)
    assert np.allclose(mover.velocity, expected_velocity, atol=1e-6)
