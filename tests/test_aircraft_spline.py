import numpy as np

from mover_sim.math.coordinates import enu_to_ecef
from mover_sim.math.orientation import rotate_vector_by_quaternion
from mover_sim.models.spline_mover import AircraftSplineMover


class TimeStub:
    def __init__(self, t=0.0):
        self.t = t

    def get_time(self):
        return self.t


def _local_up(position):
    return position / np.linalg.norm(position)


def _attach_time(mover, t):
    context = TimeStub(t)
    mover._context = context
    return context


def _enu_positions(points, ref_lat=0.0, ref_lon=0.0, ref_alt=0.0):
    return [enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt) for e, n, u in points]


def test_aircraft_spline_mover_straight_segment_gives_level_attitude():
    times = [0.0, 5.0, 10.0]
    positions = _enu_positions([
        [0.0, 0.0, 1000.0],
        [1000.0, 0.0, 1000.0],
        [2000.0, 0.0, 1000.0],
    ])

    mover = AircraftSplineMover(times, positions)
    _attach_time(mover, 5.0)

    state = mover.get_state()
    quaternion = state[6:10]
    body_rates = state[10:13]
    forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], quaternion)
    body_up = rotate_vector_by_quaternion([0.0, 0.0, 1.0], quaternion)
    local_up = _local_up(state[:3])

    assert mover.get_state_dimension() == 13
    assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-7)
    assert abs(np.dot(forward, local_up)) < 1e-3
    assert np.dot(body_up, local_up) > 0.999
    assert np.linalg.norm(body_rates) < 1e-3


def test_aircraft_spline_mover_climbing_segment_gives_positive_pitch():
    times = [0.0, 5.0, 10.0]
    positions = _enu_positions([
        [0.0, 0.0, 1000.0],
        [1000.0, 0.0, 1250.0],
        [2000.0, 0.0, 1500.0],
    ])

    mover = AircraftSplineMover(times, positions)
    _attach_time(mover, 5.0)

    state = mover.get_state()
    quaternion = state[6:10]
    forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], quaternion)
    local_up = _local_up(state[:3])

    assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-7)
    assert np.dot(forward, local_up) > 0.05


def test_aircraft_spline_mover_turning_segment_gives_banked_attitude():
    times = [0.0, 5.0, 10.0, 15.0, 20.0]
    radius = 2000.0
    angles = np.linspace(-0.8, 0.8, len(times))
    positions = _enu_positions([
        [radius * np.cos(theta), radius * np.sin(theta), 1000.0]
        for theta in angles
    ])

    mover = AircraftSplineMover(times, positions)
    _attach_time(mover, 10.0)

    state = mover.get_state()
    quaternion = state[6:10]
    body_rates = state[10:13]
    body_right = rotate_vector_by_quaternion([0.0, 1.0, 0.0], quaternion)
    local_up = _local_up(state[:3])

    assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-7)
    assert abs(np.dot(body_right, local_up)) > 0.05
    assert np.linalg.norm(body_rates) > 1e-4
