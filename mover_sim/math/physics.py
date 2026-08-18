import numpy as np
from mover_sim.math.coordinates import ecef_to_lla

# WGS-84 physical constants
GM = 3.986004418e14      # Earth's gravitational constant (m^3/s^2)
OMEGA_E = 7.292115e-5    # Earth's rotation rate (rad/s)
RHO_0 = 1.225            # Sea-level atmospheric density (kg/m^3)
H_SCALE = 8500.0         # Atmospheric scale height (meters)

def gravity(pos_ecef):
    """
    Calculate the spherical gravity acceleration vector in ECEF.
    g = -GM * r / |r|^3
    
    Parameters:
        pos_ecef: ECEF position vector [X, Y, Z] in meters (3,) or array-like
        
    Returns:
        gravity acceleration vector [gx, gy, gz] in m/s^2 (3,)
    """
    r = np.asarray(pos_ecef, dtype=float)
    r_mag = np.linalg.norm(r)
    if r_mag < 1e-3:
        return np.zeros(3)
    return -GM * r / (r_mag ** 3)

def coriolis_acceleration(vel_ecef):
    """
    Calculate the Coriolis acceleration vector in ECEF.
    a_coriolis = -2 * (omega_e x v)
    
    Parameters:
        vel_ecef: ECEF velocity vector [Vx, Vy, Vz] in m/s (3,) or array-like
        
    Returns:
        Coriolis acceleration vector [ax, ay, az] in m/s^2 (3,)
    """
    v = np.asarray(vel_ecef, dtype=float)
    # Earth's rotation vector in ECEF: [0, 0, OMEGA_E]
    # omega x v = [-OMEGA_E * vy, OMEGA_E * vx, 0]
    # -2 * (omega x v) = [2 * OMEGA_E * vy, -2 * OMEGA_E * vx, 0]
    ax = 2.0 * OMEGA_E * v[1]
    ay = -2.0 * OMEGA_E * v[0]
    az = 0.0
    return np.array([ax, ay, az])

def air_density(alt_m):
    """
    Calculate atmospheric density at a given altitude using an exponential model.
    rho = rho_0 * exp(-alt / H)
    
    Parameters:
        alt_m: Altitude in meters above the WGS-84 ellipsoid
        
    Returns:
        Density in kg/m^3
    """
    alt = max(0.0, alt_m)
    return RHO_0 * np.exp(-alt / H_SCALE)

def aerodynamic_drag_force(vel_ecef, alt_m, cd, area, wind_ecef=None):
    """
    Calculate the aerodynamic drag force vector in ECEF.
    F_drag = -0.5 * rho * Cd * A * |v_rel| * v_rel
    
    Parameters:
        vel_ecef: ECEF velocity vector [Vx, Vy, Vz] in m/s
        alt_m: Altitude in meters above the ellipsoid
        cd: Drag coefficient (dimensionless)
        area: Reference area (m^2)
        wind_ecef: ECEF wind velocity vector [Vx, Vy, Vz] in m/s (optional)
        
    Returns:
        Drag force vector [Fx, Fy, Fz] in Newtons (3,)
    """
    v = np.asarray(vel_ecef, dtype=float)
    w = np.asarray(wind_ecef, dtype=float) if wind_ecef is not None else np.zeros(3)
    
    # Relative velocity of platform to the air
    v_rel = v - w
    v_rel_mag = np.linalg.norm(v_rel)
    
    if v_rel_mag < 1e-6:
        return np.zeros(3)
        
    rho = air_density(alt_m)
    drag_mag = 0.5 * rho * cd * area * (v_rel_mag ** 2)
    
    # Drag acts opposite to relative velocity direction
    return -drag_mag * (v_rel / v_rel_mag)
