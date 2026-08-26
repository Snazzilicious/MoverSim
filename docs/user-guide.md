# User Guide

## Overview

MoverSim is a script-driven simulation library. There is no scenario file format or GUI.
You create simulation objects directly in Python and run them with `SimulationEngine`.

Typical flow:
1. Create a `SimulationEngine`.
2. Create one or more `Platform` objects.
3. Attach a `Mover` to each platform.
4. Optionally attach a `Controller`.
5. Optionally attach observers such as `CSVLogger`.
6. Register platforms with the engine.
7. Schedule any discrete events.
8. Run the engine to a target time.

## Core Objects

### `SimulationEngine`

The engine owns simulation time, event dispatch, and RK45 integration of Newtonian
movers.

The engine state model is arbitrary-dimensional: each Newtonian mover contributes its own
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
- `NewtonianMover`: generic continuous-state base for RK45-integrated movers
- `AnalyticalMover`: generic time-driven base for analytical movers
- `TranslationalNewtonianMover`: position/velocity compatibility mover
- `TranslationalAnalyticalMover`: analytical position/velocity compatibility mover
- `Mover.t`: current simulation time view, including RK45 substep time during integration

#### `NewtonianMover`

Use `NewtonianMover` when motion should be integrated from derivatives.

Subclasses implement:

```python
def compute_state_derivative(self, t, state):
    return dstate
```

`NewtonianMover` is a generic base. It does not inject gravity, Coriolis, or any other
force model on its own.

If your model is translational, use `TranslationalNewtonianMover` and implement:

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

Built-in observer:
- `CSVLogger`

`CSVLogger` writes time, ECEF position, geodetic coordinates, and velocity for each
platform.

Current limitation:
- CSV columns are fixed at simulation start, so platforms registered later are not added
  to the output file.

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

### Constant-Velocity Newtonian Platform

```python
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.core.mover import TranslationalNewtonianMover

engine = SimulationEngine()
vehicle = Platform("vehicle", TranslationalNewtonianMover([0, 0, 0], [10, 20, 30]))

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

### CSV Logging

```python
from mover_sim.core.observer import CSVLogger

logger = CSVLogger(engine, "output/telemetry.csv", log_interval=0.5)
engine.run(30.0)
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
