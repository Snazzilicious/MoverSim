# Consistent State Refactor Plan

## Objective

Refactor the engine and mover system so that all movers see a **consistent, synchronized
state** when computing their derivatives during RK45 integration. This removes the current
limitation where a mover cannot observe another mover's state at the same integration
substep.

A secondary goal is to clarify the design by consolidating all simulation state into a
single owner (`SimulationContext`) and making movers purely behavioral (no internal state
members).

---

## Background

In the current design, each `NewtonianMover` owns its own `_position` and `_velocity`
fields. During integration, the engine slices the global `y` vector and passes only a
mover's *own* slice to `compute_derivatives`. A mover that is coupled to another (e.g. a
homing missile tracking a target) has no access to the other mover's substep state — it
can only see committed state, which lags behind the current RK45 evaluation point.

---

## Proposed Design

### New Class: `SimulationContext`

A single shared object owned by the engine. It is the **sole owner of all Newtonian mover
state**, both committed (between integration steps) and temporary (within `ode_fun`).

```
SimulationContext
  .t               float       — current committed simulation time
  .committed_y     np.ndarray  — state of all Newtonian movers after last accepted RK45 step
  ._integration_y  np.ndarray  — current ode_fun substep state (None when not integrating)
  ._integrating    bool        — True only during ode_fun execution
  ._index_map      dict        — {NewtonianMover -> start_index in y}
```

**Key methods:**

| Method | Description |
|---|---|
| `register(mover, initial_state)` | Appends mover's initial 6-element state to `committed_y`; records its start index. |
| `get_state(mover) -> (pos, vel)` | Returns substep state during integration, committed state otherwise. |
| `_enter_integration(y)` | Called at the top of `ode_fun`; points `_integration_y` at the current substep `y`. |
| `_exit_integration()` | Called at the bottom of `ode_fun`; clears `_integration_y`. |
| `commit(y, t)` | Called by engine after each accepted `solver.step()`; copies `y` to `committed_y`. |

`_enter_integration` / `_exit_integration` / `commit` are **private to the engine** — user
code never calls them.

---

### Changes to `NewtonianMover`

- **Remove**: `_position`, `_velocity`, `get_state()` (returning concatenated array),
  `set_state()`.
- **Add**: `_context` reference (injected by the engine at registration).
- **Add**: `get_initial_state() -> np.ndarray` — returns the 6-element `[pos, vel]` vector
  used to seed `committed_y`. Called once by the engine at registration time.
- **Keep**: `compute_derivatives(t, pos, vel)` — signature unchanged.

```python
# Example coupled mover — no platform-ID strings required
class HomingMover(NewtonianMover):
    def __init__(self, initial_pos, initial_vel, target_mover):
        super().__init__(initial_pos, initial_vel)
        self._target = target_mover  # direct reference, wired by user script

    def compute_derivatives(self, t, pos, vel):
        target_pos, target_vel = self._target.get_state()  # reads substep state
        ...
```

The `get_state()` call on `target_mover` routes through the shared `SimulationContext`,
returning the substep `y` slice — consistent with the calling mover's own `pos`/`vel`.

---

### Changes to `AnalyticalMover`

- **Remove**: `_position`, `_velocity`, `update(t)`.
- `get_state()` calls `get_state_at(context.t)` (or `t` passed directly) — deterministic,
  so there is no consistency issue. No entry in `committed_y` is needed.

---

### Changes to `SimulationEngine`

#### `register_platform`

```
1. Call mover.get_initial_state() to get the initial [pos, vel] vector.
2. Call context.register(mover, initial_state) to allocate a slot in committed_y.
3. Inject context into mover._context.
```

#### `step_continuous`

```
1. Build active_movers list and y0 from context.committed_y (no manual concatenation).
2. Define ode_fun:
     a. context._enter_integration(y)
     b. call compute_derivatives for each mover
     c. context._exit_integration()
3. After each solver.step():
     context.commit(solver.y, solver.t)
     self.t = solver.t
     publish "position_updated"
4. Remove all set_state() calls — engine no longer writes back to individual movers.
```

#### Remove

- `_solver_reset_flag` — unused per existing comment; drop it.
- The two identical early-return if-blocks in `step_continuous` — merge into one.

---

## Migration Notes

- The `Mover` base class `__init__` currently stores `initial_position` and
  `initial_velocity` as `_position`/`_velocity`. After the refactor, these are still
  needed until `register_platform` is called (to answer `get_initial_state()`). They can
  be kept as **constructor-only temporaries** or stored as-is — they just won't be used as
  the live state source after registration.
- `platform.mover.position` and `platform.mover.velocity` convenience properties can be
  preserved: implement them to call `self.get_state()` so they continue to work for event
  callbacks and observers (which run after `_exit_integration`, so they always see
  committed state).
- Existing `AnalyticalMover` subclasses (e.g. `WaypointMover`) need minor updates: remove
  reliance on `self._position`/`self._velocity`; `get_state()` should call
  `get_state_at(self._context.t)`.

---

## File-by-File Work

| File | Changes |
|---|---|
| `core/engine.py` | Add `SimulationContext` class; update `register_platform`, `step_continuous`, `run`; remove `_solver_reset_flag`; merge duplicate early-return blocks |
| `core/mover.py` | Remove state members from `Mover`/`NewtonianMover`; add `get_initial_state()`; update `get_state()` to use context; update `AnalyticalMover.get_state()` |
| `tests/test_engine.py` | Update/add tests covering: single mover integration, cross-mover state access within `compute_derivatives`, committed-vs-substep state distinction |
| `tests/test_movers.py` | Update mover unit tests to use context injection |
| `MoverDesign.md` | Update class descriptions for `SimulationContext`, revised `Mover`, revised engine responsibilities |

---

## What This Does Not Change

- `EventScheduler`, `EventBroker`, `Platform`, `Controller` — no changes needed.
- The RK45 integration loop structure — only the state read/write paths change.
- Public API for scheduling events, registering platforms, and writing controllers.
- The script-driven usage pattern — users wire coupled movers however suits their
  simulation; the engine imposes no convention on this.

---

## Acceptance Criteria

1. A `HomingMover` can call `self._target.get_state()` inside `compute_derivatives` and
   receive the target's state at the **same RK45 substep**, not a lagging committed value.
2. Events and observers calling `mover.get_state()` (or `mover.position`/`mover.velocity`)
   always receive **committed state**.
3. All existing tests pass without changes to test logic (only setup wiring updated).
4. `_solver_reset_flag` is removed with no test regressions.
