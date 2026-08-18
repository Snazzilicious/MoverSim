import numpy as np

class Mover:
    """
    Abstract base class for all movers. A Mover manages the kinematic and 
    dynamic state of a platform (position and velocity in global coordinates).
    """
    def __init__(self, initial_position, initial_velocity=None):
        """
        Parameters:
            initial_position: ECEF coordinates [X, Y, Z] in meters (array-like).
            initial_velocity: ECEF velocity [Vx, Vy, Vz] in meters/second (array-like).
        """
        self.platform = None  # Linked when added to Platform
        self._position = np.asarray(initial_position, dtype=float)
        self._velocity = np.asarray(initial_velocity, dtype=float) if initial_velocity is not None else np.zeros(3)

    @property
    def position(self):
        """Get the current position [X, Y, Z] in ECEF meters."""
        return self._position

    @property
    def velocity(self):
        """Get the current velocity [Vx, Vy, Vz] in ECEF meters/second."""
        return self._velocity

    def update(self, t):
        """
        Update the state of the mover to simulation time t.
        Only used for Analytical Movers.
        """
        pass


class AnalyticalMover(Mover):
    """
    Base class for movers whose motion is governed by explicit analytical functions of time.
    Bypasses the numerical ODE solver.
    """
    def update(self, t):
        pos, vel = self.get_state_at(t)
        self._position = np.asarray(pos, dtype=float)
        self._velocity = np.asarray(vel, dtype=float)

    def get_state_at(self, t):
        """
        Evaluate the position and velocity at time t.
        Must be implemented by subclasses.
        
        Returns:
            position (array-like of shape (3,)), velocity (array-like of shape (3,))
        """
        raise NotImplementedError


class NewtonianMover(Mover):
    """
    Base class for movers whose motion is integrated numerically using an ODE solver.
    """
    def __init__(self, initial_position, initial_velocity=None, enable_gravity=False, enable_coriolis=False):
        """
        Parameters:
            initial_position: ECEF coordinates [X, Y, Z] in meters.
            initial_velocity: ECEF velocity [Vx, Vy, Vz] in m/s.
            enable_gravity: If True, applies standard gravitational forces in derivative calculations.
            enable_coriolis: If True, applies Coriolis forces in derivative calculations.
        """
        super().__init__(initial_position, initial_velocity)
        self.enable_gravity = enable_gravity
        self.enable_coriolis = enable_coriolis

    def get_state_dimension(self):
        """
        Returns the number of state variables (typically 6: X, Y, Z, Vx, Vy, Vz).
        """
        return 6

    def get_state(self):
        """
        Get the current state vector slice to pack into the global solver vector.
        """
        return np.concatenate([self._position, self._velocity])

    def set_state(self, state_slice):
        """
        Set the current state from a slice of the global solver vector.
        """
        self._position = np.asarray(state_slice[0:3], dtype=float)
        self._velocity = np.asarray(state_slice[3:6], dtype=float)

    def compute_derivatives(self, t, pos, vel):
        """
        Compute derivatives of the state variables (dPos/dt, dVel/dt).
        
        Parameters:
            t: Current time (seconds).
            pos: Current position vector in ECEF (3,).
            vel: Current velocity vector in ECEF (3,).
            
        Returns:
            dpos: Position derivatives [Vx, Vy, Vz] (3,).
            dvel: Velocity derivatives (accelerations) [Ax, Ay, Az] (3,).
        """
        dpos = vel
        dvel = np.zeros(3)
        
        if self.enable_gravity:
            from mover_sim.math.physics import gravity
            dvel += gravity(pos)
            
        if self.enable_coriolis:
            from mover_sim.math.physics import coriolis_acceleration
            dvel += coriolis_acceleration(vel)
            
        return dpos, dvel
