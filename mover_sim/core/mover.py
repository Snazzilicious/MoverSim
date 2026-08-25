import numpy as np

class Mover:
    """
    Abstract base class for all movers. A Mover defines the kinematic and dynamic
    behaviour of a platform.

    For AnalyticalMovers, _position and _velocity are the live state, updated each
    step via update(t).  For NewtonianMovers they serve only as pre-registration
    temporaries; after register_platform() is called the SimulationContext becomes
    the sole owner of state and _position/_velocity are no longer used.
    """
    def __init__(self, initial_position, initial_velocity=None):
        """
        Parameters:
            initial_position: ECEF coordinates [X, Y, Z] in meters (array-like).
            initial_velocity: ECEF velocity [Vx, Vy, Vz] in meters/second (array-like).
        """
        self.platform = None  # Linked when added to Platform
        self._context = None  # Injected by the engine at registration
        self._initial_position = np.asarray(initial_position, dtype=float)
        self._initial_velocity = np.asarray(initial_velocity, dtype=float) if initial_velocity is not None else np.zeros(3)

    def get_initial_state(self):
        """
        Return the 6-element initial state vector [x, y, z, vx, vy, vz].

        Called once by the engine at registration time to seed the SimulationContext.
        After that, initial_position and initial_velocity are no longer the live state source.
        """
        return np.concatenate([self._initial_position, self._initial_velocity])

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
    
    @property
    def position(self):
        """Get the current position [X, Y, Z] in ECEF meters."""
        return self.get_state()[:3]

    @property
    def velocity(self):
        """Get the current velocity [Vx, Vy, Vz] in ECEF meters/second."""
        return self.get_state()[3:]

    def get_state(self):
        """
        Evaluate and return the 6-element [x, y, z, vx, vy, vz] state at the current
        simulation time.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class NewtonianMover(Mover):
    """
    Base class for movers whose motion is integrated numerically using an ODE solver.

    State is owned by the SimulationContext after registration. Subclasses implement
    compute_derivatives() and may hold references to other movers to call get_state()
    on them; the context ensures all movers see a consistent substep snapshot.
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
    
    @property
    def position(self):
        """Get the current position [X, Y, Z] in ECEF meters."""
        return self.get_state()[:3]

    @property
    def velocity(self):
        """Get the current velocity [Vx, Vy, Vz] in ECEF meters/second."""
        return self.get_state()[3:]

    def get_state(self):
        """
        Return the 6-element [x, y, z, vx, vy, vz] state vector.

        Routes through the SimulationContext so that:
          - During ode_fun evaluation: returns the current RK45 substep values,
            consistent with all other movers being evaluated in the same call.
          - At all other times: returns the last committed (post-step) values,
            which is what events and observers should see.

        Falls back to the constructor-supplied initial_position/initial_velocity if called
        before the mover has been registered with an engine.
        """
        if self._context is not None:
            return self._context.get_state(self)
        return self.get_initial_state()

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
