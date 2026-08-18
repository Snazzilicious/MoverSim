import numpy as np
from scipy.interpolate import CubicSpline
from mover_sim.core.mover import AnalyticalMover

class SplineMover(AnalyticalMover):
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
            
        super().__init__(self.positions[0])
        
        # Fit cubic spline (clamped boundary conditions to start/stop cleanly)
        self.spline = CubicSpline(self.times, self.positions, bc_type='clamped')
        self.spline_deriv = self.spline.derivative()

    def get_state_at(self, t):
        if t <= self.times[0]:
            return self.positions[0].copy(), np.zeros(3)
        elif t >= self.times[-1]:
            return self.positions[-1].copy(), np.zeros(3)
        else:
            pos = self.spline(t)
            vel = self.spline_deriv(t)
            return pos, vel


class WaypointMover(AnalyticalMover):
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
            
        super().__init__(self.positions[0])

    def get_state_at(self, t):
        if len(self.times) == 1 or t <= self.times[0]:
            return self.positions[0].copy(), np.zeros(3)
        elif t >= self.times[-1]:
            return self.positions[-1].copy(), np.zeros(3)
            
        # Binary search to find which waypoint interval we are in
        idx = np.searchsorted(self.times, t) - 1
        t0, t1 = self.times[idx], self.times[idx + 1]
        p0, p1 = self.positions[idx], self.positions[idx + 1]
        
        dt = t1 - t0
        if dt <= 0:
            return p0.copy(), np.zeros(3)
            
        # Linear interpolation
        tau = (t - t0) / dt
        pos = p0 + tau * (p1 - p0)
        vel = (p1 - p0) / dt
        return pos, vel
