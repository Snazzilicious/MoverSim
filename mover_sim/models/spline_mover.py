import numpy as np
from scipy.interpolate import CubicSpline
from mover_sim.core.mover import AnalyticalMover, TranslationalAnalyticalMover, TranslationalMover
from mover_sim.math.orientation import (
    build_aircraft_body_axes,
    normalize_quaternion,
    quaternion_from_basis,
    quaternion_multiply,
)

class SplineMover(TranslationalAnalyticalMover):
    """
    An analytical mover that follows a smooth path defined by a Cubic Spline.
    """
    def __init__(self, times, positions):
        """
        Parameters:
            times: Array-like of shape (N,) specifying the times at each control point.
            positions: Array-like of shape (N, 3) specifying the ECEF positions at each time.
        """
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        if len(self.times) != len(self.positions):
            raise ValueError("times and positions must have the same length")
        if len(self.times) < 2:
            raise ValueError("At least 2 points are required for spline interpolation")
            
        super().__init__()
        
        # Fit cubic spline (clamped boundary conditions to start/stop cleanly)
        self.spline = CubicSpline(self.times, self.positions, bc_type='clamped')
        self.spline_deriv = self.spline.derivative()

    def get_state(self):
        t = self.t
        if self.t <= self.times[0]:
            return np.concatenate([self.positions[0].copy(), np.zeros(3)])
        if t >= self.times[-1]:
            return np.concatenate([self.positions[-1].copy(), np.zeros(3)])

        pos = self.spline(t)
        vel = self.spline_deriv(t)
        return np.concatenate([pos, vel])


class WaypointMover(TranslationalAnalyticalMover):
    """
    An analytical mover that follows a set of waypoints using linear interpolation.
    """
    def __init__(self, times, positions):
        """
        Parameters:
            times: Array-like of shape (N,) specifying the arrival times at each waypoint.
            positions: Array-like of shape (N, 3) specifying ECEF coordinates at each waypoint.
        """
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        if len(self.times) != len(self.positions):
            raise ValueError("times and positions must have the same length")
        if len(self.times) < 1:
            raise ValueError("At least 1 waypoint is required")
            
        super().__init__()

    def get_state(self):
        t = self.t
        if len(self.times) == 1 or t <= self.times[0]:
            return np.concatenate([self.positions[0].copy(), np.zeros(3)])
        if t >= self.times[-1]:
            return np.concatenate([self.positions[-1].copy(), np.zeros(3)])
             
        # Binary search to find which waypoint interval we are in
        idx = np.searchsorted(self.times, t, side='right') - 1
        t0, t1 = self.times[idx], self.times[idx + 1]
        p0, p1 = self.positions[idx], self.positions[idx + 1]
        
        dt = t1 - t0
        if dt <= 0:
            return np.concatenate([p0.copy(), np.zeros(3)])
             
        # Linear interpolation
        tau = (t - t0) / dt
        pos = p0 + tau * (p1 - p0)
        vel = (p1 - p0) / dt
        return np.concatenate([pos, vel])


class AircraftSplineMover(TranslationalMover, AnalyticalMover):
    """Analytical mover that follows a spline path with derived aircraft attitude state."""

    def __init__(self, times, positions):
        """
        Parameters:
            times: Array-like of shape (N,) specifying the times at each control point.
            positions: Array-like of shape (N, 3) specifying the ECEF positions at each time.

        The mover derives translational velocity, attitude quaternion, and body rates from
        the spline geometry.
        """
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        if len(self.times) != len(self.positions):
            raise ValueError("times and positions must have the same length")
        if len(self.times) < 2:
            raise ValueError("At least 2 points are required for spline interpolation")

        dt = np.diff(self.times)
        if np.any(dt <= 0.0):
            raise ValueError("times must be strictly increasing")

        super().__init__()

        self.spline = CubicSpline(self.times, self.positions, bc_type="clamped")
        self.spline_deriv = self.spline.derivative()
        self.spline_second_deriv = self.spline_derivative_order(2)

        min_dt = np.min(dt)
        self._orientation_dt = max(1e-3, min_dt * 1e-2)

    def spline_derivative_order(self, order):
        return self.spline.derivative(order)

    def get_state_dimension(self):
        return 13

    def get_orientation_slice(self):
        return slice(6, 10)

    def get_body_rate_slice(self):
        return slice(10, 13)

    @property
    def orientation(self):
        return self.get_state()[self.get_orientation_slice()]

    @property
    def body_rates(self):
        return self.get_state()[self.get_body_rate_slice()]

    def get_state(self):
        t = self.t
        if t <= self.times[0]:
            pos = self.positions[0].copy()
            vel = np.zeros(3)
            quat = self._orientation_at_time(self.times[0])
            body_rates = np.zeros(3)
            return np.concatenate([pos, vel, quat, body_rates])

        if t >= self.times[-1]:
            pos = self.positions[-1].copy()
            vel = np.zeros(3)
            quat = self._orientation_at_time(self.times[-1])
            body_rates = np.zeros(3)
            return np.concatenate([pos, vel, quat, body_rates])

        pos = self.spline(t)
        vel = self.spline_deriv(t)
        quat = self._orientation_at_time(t)
        body_rates = self._body_rates_at_time(t, quat)
        return np.concatenate([pos, vel, quat, body_rates])

    def _clamp_time(self, t):
        return float(np.clip(t, self.times[0], self.times[-1]))

    def _align_quaternion_sign(self, quaternion, reference):
        quaternion = normalize_quaternion(quaternion)
        if np.dot(quaternion, reference) < 0.0:
            quaternion = -quaternion
        return quaternion

    def _reference_forward(self, t):
        candidates = []
        for offset in [0.0, self._orientation_dt, -self._orientation_dt, 2.0 * self._orientation_dt, -2.0 * self._orientation_dt]:
            t_eval = self._clamp_time(t + offset)
            candidates.append(self.spline_deriv(t_eval))

        candidates.append(self.positions[-1] - self.positions[0])

        for candidate in candidates:
            norm = np.linalg.norm(candidate)
            if norm > 1e-8:
                return candidate / norm

        raise ValueError("unable to determine spline tangent direction")

    def _orientation_at_time(self, t):
        t_eval = self._clamp_time(t)
        pos = self.spline(t_eval)
        vel = self.spline_deriv(t_eval)
        acc = self.spline_second_deriv(t_eval)

        speed = np.linalg.norm(vel)
        forward = vel / speed if speed > 1e-8 else self._reference_forward(t_eval)

        pos_norm = np.linalg.norm(pos)
        local_vertical = pos / pos_norm if pos_norm > 1e-8 else np.array([0.0, 0.0, 1.0])

        lateral_acc = acc - np.dot(acc, forward) * forward
        curvature_vector = lateral_acc if np.linalg.norm(lateral_acc) > 1e-8 else None

        forward_axis, right_axis, up_axis = build_aircraft_body_axes(
            forward,
            local_vertical,
            curvature_vector=curvature_vector,
        )
        return quaternion_from_basis(forward_axis, right_axis, up_axis)

    def _body_rates_at_time(self, t, quaternion):
        dt = self._orientation_dt
        t0 = self.times[0]
        t1 = self.times[-1]

        if t - dt >= t0 and t + dt <= t1:
            q_prev = self._align_quaternion_sign(self._orientation_at_time(t - dt), quaternion)
            q_next = self._align_quaternion_sign(self._orientation_at_time(t + dt), quaternion)
            q_dot = (q_next - q_prev) / (2.0 * dt)
        elif t + dt <= t1:
            q_next = self._align_quaternion_sign(self._orientation_at_time(t + dt), quaternion)
            q_dot = (q_next - quaternion) / dt
        elif t - dt >= t0:
            q_prev = self._align_quaternion_sign(self._orientation_at_time(t - dt), quaternion)
            q_dot = (quaternion - q_prev) / dt
        else:
            return np.zeros(3)

        q_conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
        omega_quaternion = 2.0 * quaternion_multiply(q_conjugate, q_dot)
        return omega_quaternion[1:]
