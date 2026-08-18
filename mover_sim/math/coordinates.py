import numpy as np

# WGS-84 Ellipsoid constants
A = 6378137.0           # semi-major axis (meters)
F_INV = 298.257223563   # inverse flattening
F = 1.0 / F_INV         # flattening
B = A * (1.0 - F)       # semi-minor axis (meters): 6356752.314245
E2 = F * (2.0 - F)      # first eccentricity squared: (a^2 - b^2)/a^2
E_PRIME2 = E2 / (1.0 - E2) # second eccentricity squared: (a^2 - b^2)/b^2

def lla_to_ecef(lat_deg, lon_deg, alt_m):
    """
    Convert Geodetic coordinates (Lat, Lon, Alt) to ECEF (X, Y, Z).
    
    Parameters:
        lat_deg: Latitude in degrees (scalar or array-like)
        lon_deg: Longitude in degrees (scalar or array-like)
        alt_m: Altitude in meters (scalar or array-like)
        
    Returns:
        x, y, z in meters
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    
    N = A / np.sqrt(1.0 - E2 * sin_lat**2)
    
    x = (N + alt_m) * cos_lat * np.cos(lon)
    y = (N + alt_m) * cos_lat * np.sin(lon)
    z = (N * (1.0 - E2) + alt_m) * sin_lat
    
    return x, y, z

def ecef_to_lla(x, y, z):
    """
    Convert ECEF coordinates (X, Y, Z) to Geodetic (Lat, Lon, Alt) using Bowring's method.
    
    Parameters:
        x, y, z: ECEF coordinates in meters (scalar or array-like)
        
    Returns:
        lat_deg, lon_deg, alt_m
    """
    # Ensure inputs are numpy arrays for element-wise operations
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    p = np.sqrt(x**2 + y**2)
    
    # Handle the pole cases where p close to 0
    # We will use np.where to handle arrays cleanly
    is_polar = p < 1e-6
    
    # Pre-allocate output arrays
    lat_rad = np.zeros_like(x)
    lon_rad = np.zeros_like(x)
    alt_m = np.zeros_like(x)
    
    # Polar calculations
    lat_rad = np.where(is_polar, np.sign(z) * np.pi / 2.0, lat_rad)
    lon_rad = np.where(is_polar, 0.0, lon_rad)
    alt_m = np.where(is_polar, np.abs(z) - B, alt_m)
    
    # Non-polar calculations
    # Bowring's parameters
    theta = np.arctan2(z * A, p * B)
    
    lat_non_polar = np.arctan2(
        z + E_PRIME2 * B * np.sin(theta)**3,
        p - E2 * A * np.cos(theta)**3
    )
    lon_non_polar = np.arctan2(y, x)
    
    sin_lat = np.sin(lat_non_polar)
    cos_lat = np.cos(lat_non_polar)
    N = A / np.sqrt(1.0 - E2 * sin_lat**2)
    
    alt_non_polar = p / cos_lat - N
    
    # Combine polar and non-polar cases
    lat_rad = np.where(is_polar, lat_rad, lat_non_polar)
    lon_rad = np.where(is_polar, lon_rad, lon_non_polar)
    alt_m = np.where(is_polar, alt_m, alt_non_polar)
    
    return np.degrees(lat_rad), np.degrees(lon_rad), alt_m

def ecef_to_enu(x, y, z, lat_ref_deg, lon_ref_deg, alt_ref_m):
    """
    Convert ECEF coordinates to East-North-Up (ENU) coordinates relative to a reference point.
    
    Parameters:
        x, y, z: ECEF coordinates in meters
        lat_ref_deg, lon_ref_deg, alt_ref_m: Geodetic coordinates of the ENU origin
        
    Returns:
        e, n, u in meters
    """
    x_ref, y_ref, z_ref = lla_to_ecef(lat_ref_deg, lon_ref_deg, alt_ref_m)
    
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref
    
    phi = np.radians(lat_ref_deg)
    lam = np.radians(lon_ref_deg)
    
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)
    
    # Rotation matrix multiply
    e = -sin_lam * dx + cos_lam * dy
    n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    u = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz
    
    return e, n, u

def enu_to_ecef(e, n, u, lat_ref_deg, lon_ref_deg, alt_ref_m):
    """
    Convert East-North-Up (ENU) coordinates to ECEF coordinates.
    
    Parameters:
        e, n, u: ENU coordinates in meters
        lat_ref_deg, lon_ref_deg, alt_ref_m: Geodetic coordinates of the ENU origin
        
    Returns:
        x, y, z in ECEF meters
    """
    x_ref, y_ref, z_ref = lla_to_ecef(lat_ref_deg, lon_ref_deg, alt_ref_m)
    
    phi = np.radians(lat_ref_deg)
    lam = np.radians(lon_ref_deg)
    
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)
    
    # Transposed rotation matrix multiply
    dx = -sin_lam * e - sin_phi * cos_lam * n + cos_phi * cos_lam * u
    dy = cos_lam * e - sin_phi * sin_lam * n + cos_phi * sin_lam * u
    dz = cos_phi * n + sin_phi * u
    
    x = x_ref + dx
    y = y_ref + dy
    z = z_ref + dz
    
    return x, y, z
