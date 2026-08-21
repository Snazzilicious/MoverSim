import heapq
import itertools
import numpy as np
from scipy.integrate import RK45
from mover_sim.core.broker import EventBroker
from mover_sim.core.mover import NewtonianMover, AnalyticalMover

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


'''
### Comments
* Would like state to be consistent across all movers when they compute their derivatives,
  in case their derivatives depend on the state of others.
    * Potential work around is to always model coupled movers as a single system, but this
      may be less intuitive to users
    * Need a mechanism for movers to query state of other platforms within ode_fun (temporary state),
      and for events & observers to query state when they are activated (committed state).
* Seems like there is or can be considerable overlap between mover and controller in current design.
  This causes confusion and potentially indicates one is not necessary.
    * If mover should not have any discontinuous logic, or change the 'mode' of the platform,
      then we should enforce that somehow
        * e.g. remove `update` and `set_state` methods
* Should clarify purpose and responsiblity of each class for users' sake
    * Engine gets computes and hosts state at each time requested by events
    * Platform interfaces individual entities logic and phsyics with engine
    * Mover either returns the state at a given time,
      or returns the acceleration of the platform's state given the time and the whole simulation state
        * Should not irreversibly alter the 'mode' of platform, b/c that could break RK45
    * Controller may alter the 'mode' of the platform in irreversible ways (e.g. stage separation)
    * Events drive simulation and denote time intervals of continuous evolution
* Does `_solver_reset_flag` ever get used?
* if-blocks in `step_continuous` appear to have identical bodies and therfore can be combined
* Engine could add EndSimulation event to itself instead of having an explicit t_end
* Will need 6 DOF state at some point
    * An AircraftSpline mover should roll and tilt with the bends in its path
* Future features (need not be added yet, but ideally not precluded)
    * despawn platform (e.g. if crashes into ground or something)
    * collision evaluator
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
        self.platforms = {}
        self.max_step = 1.0  # Default maximum step size for continuous integration
        self.running = False
        self._solver_reset_flag = False

    def schedule(self, time, callback, name=None, interval=None):
        """
        Helper to schedule an event in the simulation.
        """
        return self.scheduler.schedule(time, callback, name, interval)

    def register_platform(self, platform):
        """
        Register a platform in the simulation.
        """
        self.platforms[platform.id] = platform
        # If simulation is already running, initialize its controller
        if self.running and platform.controller:
            platform.controller.initialize(self)
        self.broker.publish("platform_registered", platform)

    def flag_solver_reset(self):
        """
        Flags that a non-smooth change has occurred (e.g. force change),
        requiring the continuous solver to be reset at the next step.
        """
        self._solver_reset_flag = True

    def stop(self):
        """
        Stop the simulation execution early.
        """
        self.running = False

    def step_continuous(self, t_target):
        """
        Advance continuous state to t_target.
        Integrates NewtonianMovers using scipy.integrate.RK45.
        Updates AnalyticalMovers directly to the target/stepped time.
        """
        # 1. Identify active Newtonian movers
        active_movers = []
        y_list = []
        idx = 0
        for platform in self.platforms.values():
            if isinstance(platform.mover, NewtonianMover):
                active_movers.append((platform.mover, idx))
                y_list.append(platform.mover.get_state())
                idx += 6

        # If no Newtonian movers exist, just update time and analytical movers
        if not active_movers:
            self.t = t_target
            for platform in self.platforms.values():
                if isinstance(platform.mover, AnalyticalMover):
                    platform.mover.update(self.t)
            self.broker.publish("position_updated", self.t)
            return

        # Safety: if time step is too small, just advance analytically to avoid RK45 errors
        if t_target - self.t < 1e-9:
            self.t = t_target
            for platform in self.platforms.values():
                if isinstance(platform.mover, AnalyticalMover):
                    platform.mover.update(self.t)
            self.broker.publish("position_updated", self.t)
            return

        y0 = np.concatenate(y_list)

        # 2. Define the combined derivative function
        def ode_fun(t, y):
            dy = np.zeros_like(y)
            for mover, start in active_movers:
                pos = y[start : start + 3]
                vel = y[start + 3 : start + 6]
                dpos, dvel = mover.compute_derivatives(t, pos, vel)
                dy[start : start + 3] = dpos
                dy[start + 3 : start + 6] = dvel
            return dy

        # Determine integration constraints
        max_step_limit = min(self.max_step, t_target - self.t) if self.max_step else (t_target - self.t)
        
        # Initialize RK45 integration solver
        solver = RK45(ode_fun, self.t, y0, t_bound=t_target, max_step=max_step_limit)

        # 3. Integrate state up to t_target
        while self.running and solver.t < t_target and solver.status == "running":
            solver.step()
            self.t = solver.t

            # Write integrated state back to Newtonian movers
            for mover, start in active_movers:
                mover.set_state(solver.y[start : start + 6])
            # Update analytical movers
            for platform in self.platforms.values():
                if isinstance(platform.mover, AnalyticalMover):
                    platform.mover.update(self.t)

            self.broker.publish("position_updated", self.t)

        # Ensure we are exactly at the target time if we are still running
        if self.running and np.abs(self.t - t_target) > 1e-9:
            self.t = t_target
            for platform in self.platforms.values():
                if isinstance(platform.mover, AnalyticalMover):
                    platform.mover.update(self.t)
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
