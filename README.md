# MoverSim

A script-driven Python simulator for moving platforms in a global coordinate frame.

## What It Is

MoverSim provides:
- a simulation engine with discrete events and continuous RK45 integration
- Newtonian movers for numerically integrated motion
- analytical movers for state defined directly as a function of time
- controllers for guidance and behavior
- observers such as CSV telemetry logging
- coordinate and physics helpers for ECEF, ENU, gravity, Coriolis, and drag

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
from mover_sim.core.mover import NewtonianMover

engine = SimulationEngine()
mover = NewtonianMover([0.0, 0.0, 0.0], [100.0, 0.0, 0.0])
platform = Platform("vehicle", mover)

engine.register_platform(platform)
engine.run(10.0)

print(platform.mover.position)
print(platform.mover.velocity)
```

## Concepts

- `Platform`: named entity containing a mover and optional controller
- `NewtonianMover`: integrated by the engine with a 6-element state vector
- `AnalyticalMover`: computes `[x, y, z, vx, vy, vz]` directly from simulation time
- `Controller`: scheduled logic that reads state and updates mover inputs or engine behavior
- `EventBroker`: pub-sub bus for lifecycle and telemetry events

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
