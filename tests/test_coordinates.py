import pytest
import numpy as np
from mover_sim.math.coordinates import (
    lla_to_ecef,
    ecef_to_lla,
    ecef_to_enu,
    enu_to_ecef
)

# Test cases: (lat_deg, lon_deg, alt_m)
TEST_LOCATIONS = [
    (0.0, 0.0, 0.0),          # Equator, Prime Meridian, sea level
    (45.0, 45.0, 100.0),      # Mid-latitudes
    (90.0, 0.0, 0.0),         # North Pole
    (-90.0, 180.0, 2000.0),    # South Pole, high altitude
    (35.6762, 139.6503, 40.0), # Tokyo, Japan
    (51.5074, -0.1278, 15.0),  # London, UK
    (-33.8688, 151.2093, 3.0), # Sydney, Australia
]

def test_lla_to_ecef_to_lla_round_trip():
    """Verify that converting LLA -> ECEF -> LLA returns the original coordinates."""
    for lat, lon, alt in TEST_LOCATIONS:
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat_rt, lon_rt, alt_rt = ecef_to_lla(x, y, z)
        
        # Check tolerance (latitude/longitude in degrees, altitude in meters)
        # Note: longitude at the poles is singular, so we check lat/alt or handle longitude wrap
        if np.abs(lat) > 89.99:
            # Near poles, longitude can be anything, check lat and alt
            assert np.isclose(lat_rt, lat, atol=1e-7)
            assert np.isclose(alt_rt, alt, atol=1e-3)
        else:
            assert np.isclose(lat_rt, lat, atol=1e-7)
            
            # Account for longitude wrap-around (e.g. -180 to 180)
            lon_diff = (lon_rt - lon + 180) % 360 - 180
            assert np.isclose(lon_diff, 0.0, atol=1e-7)
            
            assert np.isclose(alt_rt, alt, atol=1e-3)

def test_vectorized_lla_ecef():
    """Verify that functions handle numpy arrays correctly."""
    # Filter out polar locations where longitude is singular
    non_polar_locations = [loc for loc in TEST_LOCATIONS if np.abs(loc[0]) < 89.99]
    lats = np.array([loc[0] for loc in non_polar_locations])
    lons = np.array([loc[1] for loc in non_polar_locations])
    alts = np.array([loc[2] for loc in non_polar_locations])
    
    x, y, z = lla_to_ecef(lats, lons, alts)
    assert x.shape == lats.shape
    
    lat_rt, lon_rt, alt_rt = ecef_to_lla(x, y, z)
    assert np.allclose(lat_rt, lats, atol=1e-7)
    
    # Handle longitude diff modulo 360
    lon_diff = (lon_rt - lons + 180) % 360 - 180
    assert np.allclose(lon_diff, 0.0, atol=1e-7)
    assert np.allclose(alt_rt, alts, atol=1e-3)

def test_enu_origin():
    """Verify that the ENU coordinates of the reference origin are exactly (0, 0, 0)."""
    for lat, lon, alt in TEST_LOCATIONS:
        x, y, z = lla_to_ecef(lat, lon, alt)
        e, n, u = ecef_to_enu(x, y, z, lat, lon, alt)
        assert np.isclose(e, 0.0, atol=1e-5)
        assert np.isclose(n, 0.0, atol=1e-5)
        assert np.isclose(u, 0.0, atol=1e-5)

def test_enu_to_ecef_to_enu_round_trip():
    """Verify that converting ENU -> ECEF -> ENU returns the original ENU coordinates."""
    ref_lat, ref_lon, ref_alt = 35.0, -120.0, 100.0
    
    # Test offset offsets: East, North, Up
    offsets = [
        (100.0, 0.0, 0.0),
        (0.0, -500.0, 0.0),
        (0.0, 0.0, 1000.0),
        (1000.0, 2000.0, -50.0),
    ]
    
    for e, n, u in offsets:
        x, y, z = enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt)
        e_rt, n_rt, u_rt = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)
        
        assert np.isclose(e_rt, e, atol=1e-5)
        assert np.isclose(n_rt, n, atol=1e-5)
        assert np.isclose(u_rt, u, atol=1e-5)
