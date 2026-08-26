import pytest
import numpy as np
from mover_sim.core.broker import EventBroker
from mover_sim.core.engine import EventScheduler, SimulationEngine
from mover_sim.core.mover import AnalyticalMover, Mover, NewtonianMover
from mover_sim.core.platform import Platform


def test_mover_base_contract_supports_arbitrary_initial_state():
    class GenericMover(Mover):
        def get_state(self):
            return self.get_initial_state()

    mover = GenericMover([1.0, 2.0, 3.0, 4.0])

    assert mover.get_state_dimension() == 4
    assert np.allclose(mover.get_initial_state(), [1.0, 2.0, 3.0, 4.0])

    state = mover.get_initial_state()
    state[0] = 99.0
    assert np.allclose(mover.get_initial_state(), [1.0, 2.0, 3.0, 4.0])

def test_event_broker():
    broker = EventBroker()
    received_data = []

    def callback(data):
        received_data.append(data)

    broker.subscribe("test_topic", callback)
    broker.publish("test_topic", "hello")
    assert received_data == ["hello"]

    # Test unsubscribe
    broker.unsubscribe("test_topic", callback)
    broker.publish("test_topic", "world")
    assert received_data == ["hello"]  # Should not have changed

def test_event_scheduler_ordering():
    scheduler = EventScheduler()
    order = []

    scheduler.schedule(2.0, lambda eng: order.append("second"), "event2")
    scheduler.schedule(1.0, lambda eng: order.append("first"), "event1")
    scheduler.schedule(3.0, lambda eng: order.append("third"), "event3")

    # Verify peek time is 1.0
    assert scheduler.peek_next_time() == 1.0

    # Pop and verify order
    ev1 = scheduler.pop_next()
    assert ev1.name == "event1"
    assert scheduler.peek_next_time() == 2.0
    
    ev2 = scheduler.pop_next()
    assert ev2.name == "event2"
    
    ev3 = scheduler.pop_next()
    assert ev3.name == "event3"
    assert scheduler.peek_next_time() is None

def test_simulation_engine_run():
    engine = SimulationEngine()
    execution_times = []

    def one_time_event(eng):
        execution_times.append(("one_time", eng.t))

    def recurring_event(eng):
        execution_times.append(("recurring", eng.t))

    engine.schedule(1.5, one_time_event, "OneTime")
    engine.schedule(1.0, recurring_event, "Recurring", interval=1.0)

    # We will subscribe to engine broker events
    broker_events = []
    engine.broker.subscribe("sim_start", lambda t: broker_events.append(("start", t)))
    engine.broker.subscribe("sim_end", lambda t: broker_events.append(("end", t)))
    engine.broker.subscribe("position_updated", lambda t: broker_events.append(("update", t)))

    # Run simulation from 0.0 to 3.5
    engine.run(3.5)

    # Verify final time
    assert engine.t == 3.5

    # Verify broker events
    # sim_start at 0.0
    # steps to 1.0 (recurring event) -> update
    # steps to 1.5 (one-time event) -> update
    # steps to 2.0 (recurring event) -> update
    # steps to 3.0 (recurring event) -> update
    # steps to 3.5 (sim end) -> update
    # sim_end at 3.5
    start_events = [e for e in broker_events if e[0] == "start"]
    end_events = [e for e in broker_events if e[0] == "end"]
    update_events = [e for e in broker_events if e[0] == "update"]

    assert len(start_events) == 1
    assert start_events[0][1] == 0.0
    assert len(end_events) == 1
    assert end_events[0][1] == 3.5
    assert len(update_events) > 0

    # Verify event execution times
    # Recurring should fire at 1.0, 2.0, 3.0
    # One-time should fire at 1.5
    expected_execution = [
        ("recurring", 1.0),
        ("one_time", 1.5),
        ("recurring", 2.0),
        ("recurring", 3.0),
    ]
    assert execution_times == expected_execution

def test_simulation_context_returns_state_vector_and_substep_time():
    engine = SimulationEngine()

    class TrackingMover(NewtonianMover):
        def __init__(self, pos, vel):
            super().__init__(pos, vel)
            self.observed_state = None
            self.observed_time = None

        def compute_derivatives(self, t, pos, vel):
            self.observed_state = self.get_state()
            self.observed_time = self.t
            return vel, np.zeros(3)

    mover = TrackingMover([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    engine.register_platform(Platform("tracker", mover))

    engine.run(0.1)

    assert mover.observed_state is not None
    assert mover.observed_state.shape == (6,)
    assert np.allclose(mover.observed_state[3:], [1.0, 2.0, 3.0])
    assert mover.observed_time is not None
    assert mover.observed_time >= 0.0
    assert mover.observed_time <= engine.t
    assert np.allclose(mover.observed_state[:3], mover.observed_state[3:] * mover.observed_time)


def test_newtonian_context_get_state_uses_mover_state_dimension():
    engine = SimulationEngine()

    class ExtendedStateMover(NewtonianMover):
        def __init__(self):
            super().__init__([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
            self._extra_initial_state = np.array([7.0, 8.0])

        def get_initial_state(self):
            return np.concatenate([super().get_initial_state(), self._extra_initial_state])

        def get_state_dimension(self):
            return 8

    mover = ExtendedStateMover()
    engine.register_platform(Platform("extended", mover))

    assert np.allclose(mover.get_state(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

def test_analytical_mover_uses_substep_time_during_newtonian_integration():
    engine = SimulationEngine()
    engine.max_step = 0.1

    class ClockedAnalyticalMover(AnalyticalMover):
        def __init__(self):
            super().__init__([0.0, 0.0, 0.0])

        def get_state(self):
            t = self.t
            return np.array([t, -t, 2.0 * t, 1.0, -1.0, 2.0])

    class ObserverMover(NewtonianMover):
        def __init__(self, target):
            super().__init__([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self.target = target
            self.observed_target_state = None
            self.observed_time = None
            self.derivative_time = None

        def compute_derivatives(self, t, pos, vel):
            self.observed_target_state = self.target.get_state()
            self.observed_time = self.target.t
            self.derivative_time = t
            return vel, np.zeros(3)

    analytical = ClockedAnalyticalMover()
    observer = ObserverMover(analytical)

    engine.register_platform(Platform("target", analytical))
    engine.register_platform(Platform("observer", observer))

    engine.run(0.3)

    assert observer.observed_target_state is not None
    assert observer.observed_time is not None
    assert observer.derivative_time is not None
    assert np.isclose(observer.observed_time, observer.derivative_time)
    assert np.allclose(
        observer.observed_target_state,
        [observer.derivative_time, -observer.derivative_time, 2.0 * observer.derivative_time, 1.0, -1.0, 2.0],
    )

def test_analytical_only_run_publishes_states_at_context_time():
    engine = SimulationEngine()

    class ClockedAnalyticalMover(AnalyticalMover):
        def __init__(self):
            super().__init__([0.0, 0.0, 0.0])

        def get_state(self):
            t = self.t
            return np.array([10.0 + t, 20.0 - t, 30.0 + 2.0 * t, 1.0, -1.0, 2.0])

    mover = ClockedAnalyticalMover()
    engine.register_platform(Platform("analytical", mover))

    observations = []

    def record_state(t):
        observations.append((t, mover.get_state().copy()))

    engine.broker.subscribe("position_updated", record_state)
    engine.schedule(0.5, lambda eng: None, "Checkpoint")

    engine.run(1.25)

    assert [t for t, _ in observations] == [0.5, 1.25]
    for t, state in observations:
        assert np.allclose(state[:3], [10.0 + t, 20.0 - t, 30.0 + 2.0 * t])
        assert np.allclose(state[3:], [1.0, -1.0, 2.0])
    assert np.allclose(mover.get_state()[:3], [11.25, 18.75, 32.5])
