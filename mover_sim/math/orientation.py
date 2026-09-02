import numpy as np


def _as_vector(vector, size, name):
    array = np.asarray(vector, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _normalize_vector(vector, name, eps=1e-12):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError(f"{name} must have non-zero magnitude")
    return vector / norm


def _project_to_plane(vector, normal):
    return vector - np.dot(vector, normal) * normal


def normalize_quaternion(quaternion, eps=1e-12):
    """Normalize a scalar-first quaternion `[w, x, y, z]`."""
    quaternion = _as_vector(quaternion, 4, "quaternion")
    norm = np.linalg.norm(quaternion)
    if norm < eps:
        raise ValueError("quaternion must have non-zero magnitude")
    return quaternion / norm


def quaternion_multiply(q1, q2):
    """Hamilton product of scalar-first quaternions `[w, x, y, z]`."""
    w1, x1, y1, z1 = _as_vector(q1, 4, "q1")
    w2, x2, y2, z2 = _as_vector(q2, 4, "q2")
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_from_basis(forward_axis, right_axis, up_axis, atol=1e-6):
    """
    Build a scalar-first quaternion from orthonormal body axes expressed in world frame.

    The basis follows the aircraft/body convention:
    - x axis: forward
    - y axis: right
    - z axis: up
    """
    forward = _as_vector(forward_axis, 3, "forward_axis")
    right = _as_vector(right_axis, 3, "right_axis")
    up = _as_vector(up_axis, 3, "up_axis")

    rotation = np.column_stack([forward, right, up])
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("basis vectors must be orthonormal")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("basis vectors must form a right-handed frame")

    trace = np.trace(rotation)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array([
            0.25 * s,
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
        ])
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        quaternion = np.array([
            (rotation[2, 1] - rotation[1, 2]) / s,
            0.25 * s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
        ])
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        quaternion = np.array([
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            0.25 * s,
            (rotation[1, 2] + rotation[2, 1]) / s,
        ])
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        quaternion = np.array([
            (rotation[1, 0] - rotation[0, 1]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
            (rotation[1, 2] + rotation[2, 1]) / s,
            0.25 * s,
        ])

    return normalize_quaternion(quaternion)


def rotate_vector_by_quaternion(vector, quaternion):
    """Rotate a 3D vector by a scalar-first quaternion."""
    vector = _as_vector(vector, 3, "vector")
    quaternion = normalize_quaternion(quaternion)
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    pure_vector = np.concatenate([[0.0], vector])
    rotated = quaternion_multiply(quaternion_multiply(quaternion, pure_vector), conjugate)
    return rotated[1:]


def quaternion_derivative_from_body_rates(quaternion, body_rates):
    """
    Compute quaternion derivative from body rates `[p, q, r]`.

    Assumes the quaternion maps body-frame vectors into world-frame vectors.
    """
    quaternion = normalize_quaternion(quaternion)
    body_rates = _as_vector(body_rates, 3, "body_rates")
    omega = np.concatenate([[0.0], body_rates])
    return 0.5 * quaternion_multiply(quaternion, omega)


def quaternion_to_euler_zyx(quaternion):
    """Convert a scalar-first quaternion to roll, pitch, yaw angles in radians.

    The returned angles follow the aerospace ZYX convention: yaw about world Z,
    pitch about intermediate Y, and roll about body X. The quaternion is assumed to
    map body-frame vectors into world-frame vectors.
    """

    w, x, y, z = normalize_quaternion(quaternion)

    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)

    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    return np.array([roll, pitch, yaw])


def build_aircraft_body_axes(path_tangent, local_vertical, curvature_vector=None):
    """
    Build a right-handed aircraft body frame from path geometry.

    Parameters:
        path_tangent: Velocity or path tangent direction in world frame.
        local_vertical: Local up direction in world frame.
        curvature_vector: Optional curvature/turn-direction vector in world frame.

    Returns:
        `(forward, right, up)` unit vectors in world coordinates.
    """
    forward = _normalize_vector(_as_vector(path_tangent, 3, "path_tangent"), "path_tangent")
    local_up = _normalize_vector(_as_vector(local_vertical, 3, "local_vertical"), "local_vertical")

    up_reference = _project_to_plane(local_up, forward)

    if curvature_vector is not None:
        curvature_vector = _as_vector(curvature_vector, 3, "curvature_vector")
        curvature_lateral = _project_to_plane(curvature_vector, forward)
        curvature_norm = np.linalg.norm(curvature_lateral)
        if curvature_norm > 1e-12:
            # Tilt the aircraft up axis away from the turn-center direction.
            up_reference = up_reference - curvature_lateral / curvature_norm

    if np.linalg.norm(up_reference) < 1e-12:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, fallback)) > 0.95:
            fallback = np.array([1.0, 0.0, 0.0])
        up_reference = _project_to_plane(fallback, forward)

    up = _normalize_vector(up_reference, "up_reference")
    right = _normalize_vector(np.cross(up, forward), "right_axis")
    up = _normalize_vector(np.cross(forward, right), "up_axis")
    return forward, right, up


def renormalize_basis( basis ):
    """Returns the nearest unitary matrix to the provided matrix
    """
    u,_,vh = svd( basis )
    return u @ vh