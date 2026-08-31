import heapq
import itertools
import numpy as np
from scipy.integrate import RK45
from mover_sim.core.broker import EventBroker
from mover_sim.core.mover import IntegratedMover

class Event:
    """
    Represents a discrete simulation event.
    """
    def __init__(self, time, callback, name=None, interval=None):
        """
        Parameters:
            time: Absolute simulation time (float) when the event should execute.
            callback: Callable executed when the event triggers. Signature: callback(engine).
            name: Optional descriptive name.
            interval: Optional recurrence interval (float). If provided, the event repeats.
        """
        self.time = time
        self.callback = callback
        self.name = name or callback.__name__
        self.interval = interval

    def __repr__(self):
        return f"Event(name={self.name}, time={self.time}, interval={self.interval})"


class EventScheduler:
    """
    A priority queue event scheduler for discrete events.
    """
    def __init__(self):
        self._events = []
        self._counter = itertools.count()  # Tie-breaker for events scheduled at the same time

    def schedule(self, time, callback, name=None, interval=None):
        """
        Schedule a new event.
        """
        event = Event(time, callback, name, interval)
        heapq.heappush(self._events, (time, next(self._counter), event))
        return event

    def peek_next_time(self):
        """
        Return the time of the next scheduled event, or None if queue is empty.
        """
        if self._events:
            return self._events[0][0]
        return None

    def pop_next(self):
        """
        Pop and return the next scheduled Event object.
        """
        if self._events:
            return heapq.heappop(self._events)[2]
        return None

    def is_empty(self):
        return len(self._events) == 0

    def clear(self):
        self._events.clear()


class SimulationContext:
    """
    Owns all `IntegratedMover` state and provides context-aware access via get_state().

    State is maintained in two arrays:
        committed_y:    The last accepted RK45 state, updated after each solver.step().
                        This is the state seen by events and observers.
        _integration_y: The current ode_fun substep state. Set only during derivative
                        evaluation and cleared immediately after via a try/finally block.

    get_state() automatically routes to the appropriate array based on whether integration
    is active, so all movers observe a consistent snapshot for the same substep.
    """

    def __init__(self):
        self.t = 0.0
        self.committed_y = np.empty(0, dtype=float)
        self._integration_y = None
        self._integration_t = None
        self._integrating = False
        self._index_map = {}  # {IntegratedMover: state slice in committed_y}

    def register(self, mover, initial_state):
        """
        Allocate a slot in the state vector for mover and seed it with initial_state.
        Called by the engine when a platform containing an IntegratedMover is registered.

        Parameters:
            mover:         An IntegratedMover instance.
            initial_state: Array-like containing the mover's initial state vector.
        """
        initial_state = np.asarray(initial_state, dtype=float)
        expected_size = mover.get_state_dimension()
        if initial_state.shape != (expected_size,):
            raise ValueError(
                f"initial_state must have shape ({expected_size},), got {initial_state.shape}"
            )

        start = len(self.committed_y)
        state_slice = slice(start, start + expected_size)
        self._index_map[mover] = state_slice
        self.committed_y = np.concatenate([self.committed_y, initial_state])

    def get_state_slice(self, mover):
        """Return the slice occupied by mover in the shared state vector."""
        return self._index_map[mover]

    def get_state(self, mover):
        """
        Return the full state vector for mover.

        During ode_fun evaluation (_integrating is True), returns the current substep
        state so that coupled movers see values consistent with the calling mover's
        own pos/vel. At all other times, returns the last committed (post-step) state.
        """
        y = self._integration_y if self._integrating else self.committed_y
        state_slice = self._index_map[mover]
        return y[state_slice].copy()

    def get_time(self):
        """
        Return the current simulation time.

        During ode_fun evaluation this is the current RK45 substep time so analytical
        movers queried from an integrated mover observe a time-consistent snapshot.
        Otherwise it is the last committed simulation time.
        """
        return self._integration_t if self._integrating else self.t

    def _enter_integration(self, y, t):
        """
        Point the context at the current substep state vector y and time t.
        Called at the top of ode_fun before any derivative calls.
        """
        self._integration_y = y
        self._integration_t = t
        self._integrating = True

    def _exit_integration(self):
        """
        Clear the substep reference so get_state() reverts to committed_y.
        Called in the finally block of ode_fun to guarantee cleanup even on error.
        """
        self._integrating = False
        self._integration_y = None
        self._integration_t = None

    def commit(self, y, t):
        """
        Record an accepted solver state. Called by the engine after each solver.step().

        Parameters:
            y: The accepted state vector from the solver.
            t: The accepted simulation time.
        """
        self.committed_y = y.copy()
        self.t = t


'''
### Comments
* Plotting
    * On Globe
    * Entire trajectory
    * Make movie
* More examples (need to spec these better)
    * Rocket w/ stage separation
* Future features (need not be added yet, but ideally not precluded)
    * despawn platform (e.g. if crashes into ground or something)
    * collision evaluator
        * including with the ground
    * line of sight evaluator
'''
class SimulationEngine:
    """
    The core Simulation Engine that coordinates time advancement, continuous integration, 
    and discrete event dispatching.
    """
    def __init__(self):
        self.t = 0.0
        self.scheduler = EventScheduler()
        self.broker = EventBroker()
        self.context = SimulationContext()
        self.platforms = {}
        self.max_step = 1.0  # Default maximum step size for continuous integration
        self.running = False

    def schedule(self, time, callback, name=None, interval=None):
        """
        Helper to schedule an event in the simulation.
        """
        return self.scheduler.schedule(time, callback, name, interval)

    def register_platform(self, platform):
        """
        Register a platform in the simulation.

        For IntegratedMover platforms, allocates a slot in the SimulationContext state
        vector using the mover's current state as the initial value, then injects the
        context into the mover so it can call get_state() on itself or any other
        registered mover.
        """
        self.platforms[platform.id] = platform
        platform.mover._context = self.context
        if isinstance(platform.mover, IntegratedMover):
            # Seed the context with this mover's initial state and give it a context
            # reference.
            self.context.register(platform.mover, platform.mover.get_initial_state())
        # If simulation is already running, initialize its controller if present and
        # publish that the platform joined the live simulation.
        if self.running:
            if platform.controller:
                platform.controller.initialize(self)
            self.broker.publish("platform_registered", platform)

    def stop(self):
        """
        Stop the simulation execution early.
        """
        self.running = False

    def step_continuous(self, t_target):
        """
        Advance continuous state to t_target.
        Integrates `IntegratedMover` instances using scipy.integrate.RK45.
        AnalyticalMovers derive state from the current context time on demand.
        """
        # 1. Collect active integrated movers from the context index map
        active_movers = list(self.context._index_map.items())

        # Skip RK45 if there are no integrated movers or the time step is too small.
        # In both cases analytical movers observe the new time through the shared context.
        if not active_movers or t_target - self.t < 1e-9:
            self.t = t_target
            self.context.t = t_target
            self.broker.publish("position_updated", self.t)
            return

        # 2. Seed the solver from the context's committed state vector
        y0 = self.context.committed_y.copy()

        # 3. Define the combined derivative function.
        #    _enter_integration points the context at the current substep y so that any
        #    mover calling get_state() on a peer sees values consistent with its own
        #    state slice. _exit_integration is called in a finally block to guarantee cleanup.
        def ode_fun(t, y):
            self.context._enter_integration(y, t)
            try:
                dy = np.zeros_like(y)
                for mover, state_slice in active_movers:
                    state = y[state_slice]
                    derivative = np.asarray(mover.compute_state_derivative(t, state), dtype=float)
                    if derivative.shape != state.shape:
                        raise ValueError(
                            f"{mover.__class__.__name__}.compute_state_derivative() returned shape "
                            f"{derivative.shape}, expected {state.shape}"
                        )
                    dy[state_slice] = derivative
                return dy
            finally:
                self.context._exit_integration()

        # Determine integration constraints
        max_step_limit = min(self.max_step, t_target - self.t) if self.max_step else (t_target - self.t)

        # Initialize RK45 integration solver
        solver = RK45(ode_fun, self.t, y0, t_bound=t_target, max_step=max_step_limit)

        # 4. Integrate state up to t_target
        while self.running and solver.t < t_target and solver.status == "running":
            solver.step()
            # Commit the accepted step to the context (replaces individual set_state calls)
            self.context.commit(solver.y, solver.t)
            self.t = solver.t
            self.broker.publish("position_updated", self.t)

        # Ensure we are exactly at the target time if we are still running
        if self.running and np.abs(self.t - t_target) > 1e-9:
            self.t = t_target
            self.context.t = t_target
            self.broker.publish("position_updated", self.t)

    def run(self, t_end):
        """
        Run the simulation from the current time up to t_end.
        """
        self.running = True
        
        # Initialize all platform controllers before beginning
        for platform in list(self.platforms.values()):
            if platform.controller:
                platform.controller.initialize(self)

        self.broker.publish("sim_start", self.t)
        
        while self.running and self.t < t_end:
            t_next_event = self.scheduler.peek_next_time()
            
            # Determine the target time for this step
            if t_next_event is not None and t_next_event <= t_end:
                t_target = t_next_event
            else:
                t_target = t_end
                
            # 1. Advance continuous dynamics to t_target
            if t_target > self.t:
                self.step_continuous(t_target)
                
            # 2. Execute any discrete events scheduled at this time
            while self.running:
                t_next = self.scheduler.peek_next_time()
                if t_next is None or t_next > self.t:
                    break
                
                # Pop and execute event
                event = self.scheduler.pop_next()
                event.callback(self)
                
                # Reschedule if it is a recurring event
                if event.interval is not None and event.interval > 0:
                    self.schedule(self.t + event.interval, event.callback, event.name, event.interval)
                    
        self.broker.publish("sim_end", self.t)
        self.running = False
