import numpy as np

class Mover:
    """
    Base class for all movers.

    The shared contract is state-centric: a mover exposes an initial state vector and a
    state dimension, while concrete subclasses define how to interpret and evolve that
    state over time.
    """
    def __init__(self, initial_state):
        """
        Parameters:
            initial_state: Initial mover state vector.
        """
        self.platform = None  # Linked when added to Platform
        self._context = None  # Injected by the engine at registration
        self._initial_state = np.asarray(initial_state, dtype=float).copy()
        if self._initial_state.ndim != 1:
            raise ValueError("initial_state must be a 1D state vector")

    def get_initial_state(self):
        """
        Return the initial state vector used to seed the simulation.
        """
        return self._initial_state.copy()

    def get_state_dimension(self):
        """Return the number of scalar values in this mover's state vector."""
        return self._initial_state.size

    def get_state(self):
        """Return the mover's current state vector."""
        raise NotImplementedError

    def compute_state_derivative(self, t, state):
        """Return the derivative of the supplied state vector at time t."""
        raise NotImplementedError

    @property
    def t(self):
        if self._context is not None:
            return self._context.get_time()
        return 0.0
    


class AnalyticalMover(Mover):
    """
    Base class for movers whose motion is governed by explicit analytical functions of time.
    Bypasses the numerical ODE solver.
    """

    def get_state(self):
        """
        Evaluate and return the mover state at the current simulation time.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class NewtonianMover(Mover):
    """
    Base class for movers whose motion is integrated numerically using an ODE solver.

    State is owned by the SimulationContext after registration. Subclasses implement
    compute_state_derivative() and may hold references to other movers to call get_state()
    on them; the context ensures all movers see a consistent substep snapshot.
    """
    def __init__(self, initial_state, enable_gravity=False, enable_coriolis=False):
        """
        Parameters:
            initial_state: Initial mover state vector.
            enable_gravity: If True, applies standard gravitational forces in derivative calculations.
            enable_coriolis: If True, applies Coriolis forces in derivative calculations.
        """
        super().__init__(initial_state)
        self.enable_gravity = enable_gravity
        self.enable_coriolis = enable_coriolis

    def get_state(self):
        """
        Return the mover state vector.

        Routes through the SimulationContext so that:
          - During ode_fun evaluation: returns the current RK45 substep values,
            consistent with all other movers being evaluated in the same call.
          - At all other times: returns the last committed (post-step) values,
            which is what events and observers should see.

        Falls back to the constructor-supplied initial state if called before the mover
        has been registered with an engine.
        """
        if self._context is not None:
            return self._context.get_state(self)
        return self.get_initial_state()

    def compute_state_derivative(self, t, state):
        """Return the state derivative at time t for the supplied state vector."""
        raise NotImplementedError


class TranslationalMover(Mover):
    """Compatibility layer for movers that interpret state as position and velocity."""

    @staticmethod
    def build_translational_state(initial_position, initial_velocity=None):
        initial_position = np.asarray(initial_position, dtype=float)
        initial_velocity = (
            np.asarray(initial_velocity, dtype=float)
            if initial_velocity is not None
            else np.zeros(3)
        )
        return np.concatenate([initial_position, initial_velocity])

    def get_position_slice(self):
        """Return the state slice corresponding to translational position."""
        return slice(0, 3)

    def get_velocity_slice(self):
        """Return the state slice corresponding to translational velocity."""
        return slice(3, 6)

    @property
    def position(self):
        """Get the current position [X, Y, Z] in ECEF meters."""
        return self.get_state()[self.get_position_slice()]

    @property
    def velocity(self):
        """Get the current velocity [Vx, Vy, Vz] in ECEF meters/second."""
        return self.get_state()[self.get_velocity_slice()]


class TranslationalAnalyticalMover(TranslationalMover, AnalyticalMover):
    """Analytical mover with the legacy translational state convention."""

    def __init__(self, initial_position, initial_velocity=None):
        super().__init__(self.build_translational_state(initial_position, initial_velocity))


class TranslationalNewtonianMover(TranslationalMover, NewtonianMover):
    """Newtonian mover with the legacy translational state convention."""

    def __init__(self, initial_position, initial_velocity=None, enable_gravity=False, enable_coriolis=False):
        super().__init__(
            self.build_translational_state(initial_position, initial_velocity),
            enable_gravity=enable_gravity,
            enable_coriolis=enable_coriolis,
        )

    def compute_state_derivative(self, t, state):
        """Map the translational state vector to the legacy derivative pair API."""
        pos = state[self.get_position_slice()]
        vel = state[self.get_velocity_slice()]
        dpos, dvel = self.compute_derivatives(t, pos, vel)
        return np.concatenate([dpos, dvel])

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
