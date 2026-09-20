# MoverSim

A script-driven Python simulator for moving platforms in a global coordinate frame.

## What It Is

MoverSim provides:
- a simulation engine with discrete events and continuous RK45 integration
- Integrated movers for numerically integrated motion
- analytical movers for state defined directly as a function of time
- generic arbitrary-dimensional state support in the simulation core
- translational compatibility movers for position/velocity-based models
- controllers for guidance and behavior
- observers such as CSV and HDF5 trajectory logging
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
from mover_sim.core.mover import TranslationalIntegratedMover

engine = SimulationEngine()
mover = TranslationalIntegratedMover([0.0, 0.0, 0.0], [100.0, 0.0, 0.0])
platform = Platform("vehicle", mover)

engine.register_platform(platform)
engine.run(10.0)

print(platform.mover.position)
print(platform.mover.velocity)
```

## Concepts

- `Platform`: named entity containing a mover and optional controller
- `IntegratedMover`: generic context-owned continuous-state base class
- `AnalyticalMover`: generic time-driven state base class
- `TranslationalIntegratedMover`: position/velocity compatibility mover with a 6-element state
- `TranslationalAnalyticalMover`: analytical position/velocity compatibility mover
- `Controller`: scheduled logic that reads state and updates mover inputs or engine behavior
- `EventBroker`: pub-sub bus for lifecycle and telemetry events

State model:
- the engine integrates arbitrary-length state vectors for integrated movers
- movers define how to interpret their own state layout
- translational movers conventionally use `[x, y, z, vx, vy, vz]`
- rigid-body aircraft and rocket movers use 18+ element states with rotation matrix attitude and body angular rates (`FixedWingMover`, `RocketMover`)

Built-in aircraft and rocket models:
- `AircraftMover`: translational point-mass aircraft model
- `AircraftSplineMover`: analytical aircraft path follower with derived quaternion attitude
- `FixedWingMover`: rigid-body aircraft mover with rotation matrix attitude and body rates (`FixedWingAutopilot`)
- `RocketMover`: rigid-body rocket mover with multi-stage support, propellant mass, and aerodynamics (`RocketController`)

Logging:
- `CSVLogger`: flat long-row export format, one row per platform sample
- `HDF5Logger`: structured archive format with per-platform datasets and optional event tables

Recommended use:
- use CSV for quick inspection, spreadsheets, and lightweight analysis pipelines
- use HDF5 for full-fidelity archival of mixed-dimension trajectories and events

## Logging Example

```python
from mover_sim.core.observer import CSVLogger, HDF5Logger

CSVLogger(engine, "output/telemetry.csv", log_interval=0.5)
HDF5Logger(engine, "output/telemetry.h5", sample_interval=0.5)

engine.run(30.0)
```

## Examples

- `python examples/scenario_a.py`
- `python examples/scenario_b.py`

## Documentation

- User guide: [`docs/user-guide.md`](docs/user-guide.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)

## Tests

```bash
pytest
```
