import pytest
from mover_sim.core.broker import EventBroker
from mover_sim.core.engine import EventScheduler, SimulationEngine

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
