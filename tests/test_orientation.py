import numpy as np

from mover_sim.math.orientation import (
    build_aircraft_body_axes,
    normalize_quaternion,
    quaternion_derivative_from_body_rates,
    quaternion_from_basis,
    quaternion_multiply,
    rotate_vector_by_quaternion,
)


def test_normalize_quaternion_returns_unit_quaternion():
    quaternion = normalize_quaternion([2.0, 0.0, 0.0, 0.0])
    assert np.allclose(quaternion, [1.0, 0.0, 0.0, 0.0])


def test_quaternion_multiply_respects_identity():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    quaternion = normalize_quaternion([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(quaternion_multiply(identity, quaternion), quaternion)
    assert np.allclose(quaternion_multiply(quaternion, identity), quaternion)


def test_quaternion_from_basis_returns_identity_for_world_axes():
    quaternion = quaternion_from_basis(
        forward_axis=[1.0, 0.0, 0.0],
        right_axis=[0.0, 1.0, 0.0],
        up_axis=[0.0, 0.0, 1.0],
    )
    assert np.allclose(quaternion, [1.0, 0.0, 0.0, 0.0])


def test_rotate_vector_by_quaternion_matches_basis_rotation():
    quaternion = quaternion_from_basis(
        forward_axis=[0.0, 1.0, 0.0],
        right_axis=[-1.0, 0.0, 0.0],
        up_axis=[0.0, 0.0, 1.0],
    )
    rotated = rotate_vector_by_quaternion([1.0, 0.0, 0.0], quaternion)
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-7)


def test_quaternion_derivative_from_body_rates_matches_identity_case():
    derivative = quaternion_derivative_from_body_rates(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        body_rates=[2.0, 4.0, 6.0],
    )
    assert np.allclose(derivative, [0.0, 1.0, 2.0, 3.0])


def test_build_aircraft_body_axes_straight_and_level():
    forward, right, up = build_aircraft_body_axes(
        path_tangent=[1.0, 0.0, 0.0],
        local_vertical=[0.0, 0.0, 1.0],
    )
    assert np.allclose(forward, [1.0, 0.0, 0.0])
    assert np.allclose(right, [0.0, 1.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])


def test_build_aircraft_body_axes_uses_curvature_to_bank_frame():
    forward, right, up = build_aircraft_body_axes(
        path_tangent=[1.0, 0.0, 0.0],
        local_vertical=[0.0, 0.0, 1.0],
        curvature_vector=[0.0, 1.0, 0.0],
    )
    assert np.isclose(np.linalg.norm(forward), 1.0)
    assert np.isclose(np.linalg.norm(right), 1.0)
    assert np.isclose(np.linalg.norm(up), 1.0)
    assert np.allclose(np.cross(forward, right), up, atol=1e-7)
    assert up[1] < 0.0
