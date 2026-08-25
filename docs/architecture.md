# Architecture

## Overview

MoverSim combines:
- a discrete event scheduler
- a continuous RK45 integration loop for Newtonian movers
- analytical movers that derive state directly from simulation time
- a publish-subscribe broker for observers and notifications

The engine is script-driven and imposes minimal scenario structure beyond object
registration and callback signatures.

## Main Components

### `SimulationEngine`

Responsibilities:
- maintain current simulation time
- register platforms
- inject shared simulation context into movers
- integrate all Newtonian movers in one combined RK45 system
- execute scheduled discrete events
- publish lifecycle and telemetry events

Key methods:
- `register_platform(platform)`
- `schedule(time, callback, name=None, interval=None)`
- `step_continuous(t_target)`
- `run(t_end)`
- `stop()`

### `SimulationContext`

`SimulationContext` is the shared state owner used by movers.

It stores:
- `committed_y`: committed Newtonian state after the last accepted solver step
- `_integration_y`: temporary RK45 substep state during derivative evaluation
- `t`: committed simulation time
- `_integration_t`: temporary RK45 substep time during derivative evaluation
- `_index_map`: mapping from `NewtonianMover` to its slice in the combined state vector

It provides:
- `get_state(mover)` -> 6-element state vector
- `get_time()` -> current substep time during integration, committed time otherwise

This gives coupled movers a time-consistent view during RK45 derivative evaluation.

### `EventScheduler`

The scheduler is a priority queue of `Event` objects.

Current capabilities:
- schedule one-time events
- schedule recurring events
- peek at the next event time
- pop the next event for execution

Current limitation:
- no event cancellation API

### `EventBroker`

The broker is a minimal pub-sub bus.

It supports:
- `subscribe(topic, callback)`
- `unsubscribe(topic, callback)`
- `publish(topic, *args, **kwargs)`

The broker does not impose payload schemas. Callbacks receive whatever the publisher
passes.

### `Platform`

A `Platform` groups:
- a stable identifier
- one mover
- an optional controller
- optional user metadata

### Movers

#### Shared Contract

All movers expose state as:

```text
[x, y, z, vx, vy, vz]
```

They also expose:
- `position`
- `velocity`
- `t`

#### `NewtonianMover`

`NewtonianMover` participates in the combined RK45 system and implements:

```python
compute_derivatives(t, pos, vel)
```

After registration, live Newtonian state is owned by `SimulationContext`, not by the
mover instance itself.

#### `AnalyticalMover`

`AnalyticalMover` bypasses RK45 and computes state directly from simulation time via
`get_state()`.

Because `Mover.t` comes from `SimulationContext`, analytical movers also observe RK45
substep time when queried during Newtonian derivative evaluation.

### `Controller`

Controllers are event-driven logic loops.

Behavior:
- optionally schedule recurring callbacks using `update_interval`
- run `update(t, engine)`
- read state and update mover control inputs or engine behavior

There is no solver-reset mechanism in the current engine design.

### `CSVLogger`

`CSVLogger` subscribes to:
- `sim_start`
- `position_updated`
- `sim_end`

It writes:
- time
- ECEF position
- geodetic coordinates
- ECEF velocity

Known limitation:
- the CSV header is fixed at simulation start, so dynamically registered platforms are
  omitted from the file

## Execution Model

### Main Loop

1. Determine the next target time from the next event or `t_end`.
2. Advance continuous state to that target time.
3. Execute all discrete events due at the current time.
4. Repeat until stopped or `t_end` is reached.

### Continuous Advancement

If Newtonian movers are present:
- build the RK45 system from `SimulationContext.committed_y`
- expose substep state and time through `SimulationContext` inside `ode_fun`
- after each accepted solver step, commit the new state and publish `position_updated`

If no Newtonian movers are present:
- advance committed time directly
- analytical movers observe the new time through `SimulationContext`
- publish `position_updated`

### Mixed Newtonian and Analytical Access

During derivative evaluation:
- Newtonian movers read each other through `SimulationContext.get_state()`
- analytical movers compute against `SimulationContext.get_time()`

This allows a Newtonian mover to observe:
- another Newtonian mover at the same RK45 substep
- an analytical mover at the same RK45 substep time

## Event Topics in Current Use

Examples from the current codebase:
- `sim_start`
- `position_updated`
- `sim_end`
- `waypoint_reached`
- `platform_registered`
- `intercept`

For `position_updated`, the current payload is the simulation time.

## Package Layout

```text
mover_sim/
|
+-- mover_sim/
|   +-- core/
|   |   +-- broker.py
|   |   +-- controller.py
|   |   +-- engine.py
|   |   +-- mover.py
|   |   +-- observer.py
|   |   +-- platform.py
|   +-- math/
|   |   +-- coordinates.py
|   |   +-- physics.py
|   +-- models/
|       +-- aircraft_mover.py
|       +-- spline_mover.py
|
+-- examples/
+-- tests/
```

## Current Limitations

- `CSVLogger` does not expand columns for dynamically registered platforms.
- `EventScheduler` does not support cancellation.
- Mover state is currently fixed at 6 DOF translational state: position and velocity.
