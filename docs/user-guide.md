# User Guide

## Overview

MoverSim is a script-driven simulation library. There is no scenario file format or GUI.
You create simulation objects directly in Python and run them with `SimulationEngine`.

Typical flow:
1. Create a `SimulationEngine`.
2. Create one or more `Platform` objects.
3. Attach a `Mover` to each platform.
4. Optionally attach a `Controller`.
5. Optionally attach observers such as `CSVLogger` or `HDF5Logger`.
6. Register platforms with the engine.
7. Schedule any discrete events.
8. Run the engine to a target time.

## Core Objects

### `SimulationEngine`

The engine owns simulation time, event dispatch, and RK45 integration for
`IntegratedMover` instances.

The engine state model is arbitrary-dimensional: each integrated mover contributes its own
state slice to the shared solver vector.

Main entry points:
- `register_platform(platform)`
- `schedule(time, callback, name=None, interval=None)`
- `run(t_end)`
- `stop()`

### `Platform`

A platform is the named entity in the simulation.

Each platform contains:
- `id`
- `mover`
- optional `controller`
- optional `properties`

### Movers

All movers expose state through `get_state()` and `get_state_dimension()`.

The core does not require a fixed state layout. A mover can expose any state length that
fits its model.

Translational movers conventionally use a 6-element state vector:

```text
[x, y, z, vx, vy, vz]
```

Convenience properties such as `mover.position` and `mover.velocity` are provided by the
translational mover subclasses, not by the generic base classes.

Common mover families:
- `IntegratedMover`: generic continuous-state base for RK45-integrated movers
- `AnalyticalMover`: generic time-driven base for analytical movers
- `TranslationalIntegratedMover`: position/velocity compatibility mover
- `TranslationalAnalyticalMover`: analytical position/velocity compatibility mover
- `Mover.t`: current simulation time view, including RK45 substep time during integration

#### `IntegratedMover`

Use `IntegratedMover` when motion should be integrated from derivatives.

Subclasses implement:

```python
def compute_state_derivative(self, t, state):
    return dstate
```

`IntegratedMover` is a generic base. It does not inject gravity, Coriolis, or any other
force model on its own.

If your model is translational, use `TranslationalIntegratedMover` and implement:

```python
def compute_derivatives(self, t, pos, vel):
    return dpos, dvel
```

Concrete subclasses add the forces they need explicitly.

#### `AnalyticalMover`

Use `AnalyticalMover` when state is known directly as a function of time.

Subclasses implement:

```python
def get_state(self):
    return state
```

Current built-in analytical movers:
- `SplineMover`
- `WaypointMover`
- `AircraftSplineMover`

Use `TranslationalAnalyticalMover` when the state should still be interpreted as position
and velocity.

### Controllers

Controllers are scheduled logic loops attached to platforms.

They can:
- read current platform state
- update mover control inputs
- publish events
- stop the simulation

Subclasses implement:

```python
def update(self, t, engine):
    ...
```

If `update_interval` is set, the controller runs recurrently.

### Observers

Observers subscribe to engine events through `EventBroker`.

Built-in trajectory loggers:
- `CSVLogger`
- `HDF5Logger`

`CSVLogger` writes one row per platform sample in a flat long/table format.

Typical CSV columns include:
- `time`
- `platform_id`
- `state_dim`
- translational fields when available:
  - `x`, `y`, `z`
  - `lat`, `lon`, `alt`
  - `vx`, `vy`, `vz`
- orientation fields when available:
  - `qw`, `qx`, `qy`, `qz`
- body-rate fields when available:
  - `p`, `q`, `r`
- `state_json` containing the full arbitrary-dimensional state

`HDF5Logger` writes structured per-platform datasets and is the preferred high-fidelity
archive format for mixed mover types and arbitrary-dimensional state.

Typical HDF5 structure:
- `/trajectories/<platform_id>/time`
- `/trajectories/<platform_id>/state`
- optional derived datasets such as:
  - `position`
  - `velocity`
  - `lla`
  - `orientation`
  - `body_rates`
- `/events/*` when event logging is enabled
- `/metadata` attributes describing the file

Recommended use:
- use `CSVLogger` for simple tabular export and quick analysis
- use `HDF5Logger` when you want full structured state history, mixed state dimensions, or event capture

## Events

Schedule one-time or recurring callbacks with:

```python
engine.schedule(10.0, callback, "MyEvent")
engine.schedule(1.0, callback, "Tick", interval=1.0)
```

Callback signature:

```python
def callback(engine):
    ...
```

Common broker topics:
- `sim_start`
- `position_updated`
- `sim_end`
- `waypoint_reached`
- `platform_registered`
- `intercept`

## Coordinate and Physics Helpers

Coordinate helpers in `mover_sim.math.coordinates`:
- `lla_to_ecef`
- `ecef_to_lla`
- `ecef_to_enu`
- `enu_to_ecef`

Physics helpers in `mover_sim.math.physics`:
- `gravity`
- `coriolis_acceleration`
- `air_density`
- `aerodynamic_drag_force`

Orientation helpers in `mover_sim.math.orientation`:
- `normalize_quaternion`
- `quaternion_multiply`
- `quaternion_from_basis`
- `rotate_vector_by_quaternion`
- `quaternion_derivative_from_body_rates`

## Worked Example

### Constant-Velocity Integrated Platform

```python
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.core.mover import TranslationalIntegratedMover

engine = SimulationEngine()
vehicle = Platform("vehicle", TranslationalIntegratedMover([0, 0, 0], [10, 20, 30]))

engine.register_platform(vehicle)
engine.run(10.0)

print(vehicle.mover.position)  # [100, 200, 300]
```

### Aircraft Models

Built-in aircraft paths now include:
- `AircraftMover`: translational point-mass aircraft model
- `AircraftSplineMover`: analytical aircraft mover with a 13-element state
- `Aircraft6DOFMover`: rigid-body aircraft mover with a 13-element state

Rigid-body aircraft state layout:

```text
[x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
```

### Logging

```python
import h5py

from mover_sim.core.observer import CSVLogger, HDF5Logger

logger = CSVLogger(engine, "output/telemetry.csv", log_interval=0.5)
with h5py.File("output/telemetry.h5", "w") as h5:
    h5_logger = HDF5Logger(engine, h5.create_group("run_001"), sample_interval=0.5)
    engine.run(30.0)
```

`CSVLogger` accepts optional external event logging:

```python
CSVLogger(
    engine,
    "output/telemetry.csv",
    log_interval=0.5,
    include_events=True,
    events_filepath="output/telemetry.events.csv",
)
```

### Example Scenarios

Reference scripts:
- `examples/scenario_a.py`: waypoint flight pattern around SFO
- `examples/scenario_b.py`: interceptor aircraft and dynamically spawned missile
- `examples/plot_trajectories.py`: plotting helper for generated CSV files

## Testing

Run the full suite:

```bash
pytest
```
