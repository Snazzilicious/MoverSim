# Architecture

## Overview

MoverSim combines:
- a discrete event scheduler
- a continuous RK45 integration loop for integrated movers
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
- integrate all integrated movers in one combined RK45 system
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
- `committed_y`: committed integrated state after the last accepted solver step
- `_integration_y`: temporary RK45 substep state during derivative evaluation
- `t`: committed simulation time
- `_integration_t`: temporary RK45 substep time during derivative evaluation
- `_index_map`: mapping from `IntegratedMover` to its slice in the combined state vector

It provides:
- `get_state(mover)` -> mover-owned state slice
- `get_state_slice(mover)` -> slice in the combined solver vector
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

All movers expose state through `get_state()` and `get_state_dimension()`.

The core does not require a fixed state layout.

Translational movers conventionally expose:

```text
[x, y, z, vx, vy, vz]
```

Translational compatibility subclasses also expose:
- `position`
- `velocity`

All movers expose:
- `t`

#### `IntegratedMover`

`IntegratedMover` participates in the combined RK45 system and implements:

```python
compute_state_derivative(t, state)
```

After registration, live integrated state is owned by `SimulationContext`, not by the
mover instance itself.

`IntegratedMover` is force-model agnostic. Gravity, Coriolis, drag, thrust, and similar
terms belong in concrete subclasses.

For translational compatibility, `TranslationalIntegratedMover` adapts the generic state
API back to:

```python
compute_derivatives(t, pos, vel)
```

#### `AnalyticalMover`

`AnalyticalMover` bypasses RK45 and computes state directly from simulation time via
`get_state()`.

Because `Mover.t` comes from `SimulationContext`, analytical movers also observe RK45
substep time when queried during integrated derivative evaluation.

For translational compatibility, `TranslationalAnalyticalMover` exposes the conventional
position/velocity view.

#### Built-in Model Families

Current movers include:
- `SplineMover` and `WaypointMover`: translational analytical movers
- `AircraftSplineMover`: analytical aircraft mover with 13-element state
- `AircraftMover`: translational point-mass aircraft mover
- `Aircraft6DOFMover`: rigid-body aircraft mover with 13-element state

Rigid-body aircraft state layout:

```text
[x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
```

### `Controller`

Controllers are event-driven logic loops.

Behavior:
- optionally schedule recurring callbacks using `update_interval`
- run `update(t, engine)`
- read state and update mover control inputs or engine behavior

There is no solver-reset mechanism in the current engine design.

### Trajectory Loggers

The logging system is layered:
- `BaseTrajectoryLogger`: shared sampling, buffering, broker subscription, and normalized record extraction
- `CSVLogger`: flat long-row export sink
- `HDF5Logger`: structured HDF5 archive sink

`BaseTrajectoryLogger` subscribes to:
- `sim_start`
- `position_updated`
- `sim_end`
- optional event topics such as:
  - `platform_registered`
  - `waypoint_reached`
  - `intercept`

It builds normalized per-platform records containing:
- `time`
- `platform_id`
- `state`
- `state_dim`
- optional derived fields:
  - `position`
  - `velocity`
  - `lla`
  - `orientation`
  - `body_rates`

#### `CSVLogger`

`CSVLogger` writes one row per platform sample.

The output is a stable long/table format rather than a simulation-wide wide row. It is
intended as an analysis/export format.

Typical columns include:
- `time`
- `platform_id`
- `state_dim`
- translational fields when available
- orientation and body-rate fields when available
- `state_json` for the full arbitrary-dimensional state

Optional event logging is written to a separate CSV file.

#### `HDF5Logger`

`HDF5Logger` is the structured archive format.

It stores:
- `/trajectories/<platform_id>/time`
- `/trajectories/<platform_id>/state`
- optional derived datasets:
  - `position`
  - `velocity`
  - `lla`
  - `orientation`
  - `body_rates`
- `/events/*` datasets when event logging is enabled
- `/metadata` attributes describing the file schema

Per-platform trajectory groups are created lazily on first sample. Datasets are appendable,
chunked, and optionally compressed.

## Execution Model

### Main Loop

1. Determine the next target time from the next event or `t_end`.
2. Advance continuous state to that target time.
3. Execute all discrete events due at the current time.
4. Repeat until stopped or `t_end` is reached.

### Continuous Advancement

If integrated movers are present:
- build the RK45 system from `SimulationContext.committed_y`
- expose substep state and time through `SimulationContext` inside `ode_fun`
- after each accepted solver step, commit the new state and publish `position_updated`

Each integrated mover contributes its own state slice, so the combined RK45 vector can mix
different state dimensions in one solve.

If no integrated movers are present:
- advance committed time directly
- analytical movers observe the new time through `SimulationContext`
- publish `position_updated`

### Mixed Integrated and Analytical Access

During derivative evaluation:
- integrated movers read each other through `SimulationContext.get_state()`
- analytical movers compute against `SimulationContext.get_time()`

This allows an integrated mover to observe:
- another integrated mover at the same RK45 substep
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
|   |   +-- orientation.py
|   |   +-- physics.py
|   +-- models/
|       +-- aircraft_mover.py
|       +-- spline_mover.py
|
+-- examples/
+-- tests/
```

## Current Limitations

- `EventScheduler` does not support cancellation.
- `CSVLogger` stores arbitrary full state in serialized `state_json` rather than expanding generic per-element columns.
- `HDF5Logger` currently assumes `h5py` is installed at runtime.
