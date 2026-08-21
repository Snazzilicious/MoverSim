# MoverSim

A general-purpose simulator for moving platforms in a global coordinate frame.

---

## Usage Model

MoverSim is **script-driven**. There is no configuration file format, GUI, or built-in
scenario loader. Instead, users write Python scripts that directly construct and wire
together simulation objects — platforms, movers, controllers, and observers — and then
hand them to the engine to run.

This means:
- How movers are constructed and how they reference each other is entirely up to the user
  and the needs of their specific simulation.
- There is no framework convention for dependency injection, component registration
  order, or coupling patterns beyond what the engine requires at registration time.
- The engine provides the integration loop, event dispatch, and state management; the
  user provides the physics and logic.

See the `examples/` directory for reference scripts.

---

## Architecture Overview

| Component | Role |
|---|---|
| `SimulationEngine` | Owns the simulation loop, advances time, runs the ODE solver, dispatches events |
| `SimulationContext` | Owns all Newtonian mover state (committed and in-substep); routes `get_state()` correctly |
| `Platform` | Named entity; holds a `Mover` and an optional `Controller` |
| `NewtonianMover` | Purely behavioral; provides `compute_derivatives`; reads state via shared context |
| `AnalyticalMover` | Computes state as an explicit function of time; bypasses the ODE solver |
| `Controller` | Runs on a fixed schedule; may alter platform mode discontinuously |
| `EventBroker` | Publish-subscribe bus for telemetry, logging, and inter-component notification |

For full design details see [`MoverDesign.md`](MoverDesign.md) and
[`ConsistentStateRefactorPlan.md`](ConsistentStateRefactorPlan.md).
