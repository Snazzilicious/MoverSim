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

Movers expose state as a 6-element vector:

```text
[x, y, z, vx, vy, vz]
```

Convenience properties:
- `mover.position`
- `mover.velocity`
- `mover.t`

#### `NewtonianMover`

Use `NewtonianMover` when motion should be integrated from derivatives.

Subclasses implement:

```python
def compute_derivatives(self, t, pos, vel):
    return dpos, dvel
```

The base class can optionally add gravity and Coriolis acceleration.

#### `AnalyticalMover`

Use `AnalyticalMover` when position and velocity are known directly as a function of
time.

Subclasses implement:

```python
def get_state(self):
    return [x, y, z, vx, vy, vz]
```

Current built-in analytical movers:
- `SplineMover`
- `WaypointMover`

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

## Worked Example

### Constant-Velocity Newtonian Platform

```python
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.core.mover import NewtonianMover

engine = SimulationEngine()
vehicle = Platform("vehicle", NewtonianMover([0, 0, 0], [10, 20, 30]))

engine.register_platform(vehicle)
engine.run(10.0)

print(vehicle.mover.position)  # [100, 200, 300]
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
