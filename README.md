# MoverSim

A script-driven Python simulator for moving platforms in a global coordinate frame.

## What It Is

MoverSim provides:
- a simulation engine with discrete events and continuous RK45 integration
- Newtonian movers for numerically integrated motion
- analytical movers for state defined directly as a function of time
- generic arbitrary-dimensional state support in the simulation core
- translational compatibility movers for position/velocity-based models
- controllers for guidance and behavior
- observers such as CSV telemetry logging
- coordinate, physics, and orientation helpers for ECEF, ENU, gravity, Coriolis, drag, and quaternions

Users build scenarios directly in Python by constructing platforms, movers, controllers,
and observers, then handing them to `SimulationEngine`.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Minimal Example

```python
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.core.mover import TranslationalNewtonianMover

engine = SimulationEngine()
mover = TranslationalNewtonianMover([0.0, 0.0, 0.0], [100.0, 0.0, 0.0])
platform = Platform("vehicle", mover)

engine.register_platform(platform)
engine.run(10.0)

print(platform.mover.position)
print(platform.mover.velocity)
```

## Concepts

- `Platform`: named entity containing a mover and optional controller
- `NewtonianMover`: generic context-owned continuous-state base class
- `AnalyticalMover`: generic time-driven state base class
- `TranslationalNewtonianMover`: position/velocity compatibility mover with a 6-element state
- `TranslationalAnalyticalMover`: analytical position/velocity compatibility mover
- `Controller`: scheduled logic that reads state and updates mover inputs or engine behavior
- `EventBroker`: pub-sub bus for lifecycle and telemetry events

State model:
- the engine integrates arbitrary-length Newtonian state vectors
- movers define how to interpret their own state layout
- translational movers conventionally use `[x, y, z, vx, vy, vz]`
- rigid-body aircraft movers use 13 state elements: position 3, velocity 3, quaternion 4, body rates 3

Built-in aircraft movers:
- `AircraftMover`: translational point-mass aircraft model
- `AircraftSplineMover`: analytical aircraft path follower with derived quaternion attitude
- `Aircraft6DOFMover`: rigid-body aircraft model with quaternion attitude and body rates

## Examples

- `python examples/scenario_a.py`
- `python examples/scenario_b.py`
- `python examples/plot_trajectories.py`

## Documentation

- User guide: [`docs/user-guide.md`](docs/user-guide.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)

## Tests

```bash
pytest
```
