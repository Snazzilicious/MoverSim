import heapq
import itertools
from mover_sim.core.broker import EventBroker

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
        self.broker.publish("platform_registered", platform)

    def flag_solver_reset(self):
        """
        Flags that a non-smooth change has occurred (e.g. force change),
        requiring the continuous solver to be reset at the next step.
        """
        self._solver_reset_flag = True

    def step_continuous(self, t_target):
        """
        Advance continuous state to t_target.
        Note: Continuous physics (ODE solver integration) will be implemented in Phase 3.
        For now, simply advance the simulation clock.
        """
        self.t = t_target
        # Publish position/state updates to subscribers
        self.broker.publish("position_updated", self.t)

    def run(self, t_end):
        """
        Run the simulation from the current time up to t_end.
        """
        self.broker.publish("sim_start", self.t)
        
        while self.t < t_end:
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
            while True:
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
