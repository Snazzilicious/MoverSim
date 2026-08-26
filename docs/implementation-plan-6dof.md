# 6 DOF State Support Implementation Plan

## Findings

The repo is currently built around a fixed translational 6-element state contract:

- `mover_sim/core/mover.py` assumes state is always `[x, y, z, vx, vy, vz]`.
- `mover_sim/core/engine.py` packs every Newtonian mover into fixed 6-value slices and calls `compute_derivatives(t, pos, vel)`.
- `mover_sim/models/spline_mover.py` only models 3D position and translational velocity.
- `mover_sim/models/aircraft_mover.py` is a point-mass aircraft model. It infers attitude from velocity and bank command, but does not store or integrate orientation.
- `mover_sim/core/observer.py` logs only translational telemetry.
- Tests and docs codify the 6-value contract throughout.

Baseline status: `pytest` is currently green, `22 passed`.

## Recommendation

Make the engine and mover base classes fully state-centric.

That means:

- the engine only knows about full state vectors and state derivatives,
- base mover abstractions do not assume `position` or `velocity` exist,
- user-defined mover subclasses are responsible for assigning semantic meaning to state slices when needed.

That gives you a clean path to:

- keep the simulation core dimension-agnostic,
- support existing translational movers through subclass conventions,
- add analytical movers with orientation state,
- add Newtonian aircraft movers with rigid-body attitude dynamics,
- avoid baking any aircraft- or kinematics-specific assumptions into the core.

For aircraft orientation, use quaternions rather than Euler angles.

Reason:

- no gimbal lock,
- stable for pitch/roll/yaw integration,
- clean for spline-derived and rigid-body movers.

For a rigid 6-DOF aircraft, the practical state is not 12 elements but 13:

- `position` 3
- `velocity` 3
- `orientation quaternion` 4
- `body rates` 3

## Implementation Plan

1. Generalize the mover state contract.
- Refactor `Mover` to expose a full initial state vector, not just concatenated position and velocity.
- Define the base contract around generic state operations, such as:
  - `get_initial_state()`
  - `get_state_dimension()`
  - `set_state(state)`
  - `compute_state_derivative(t, state)` for continuous movers
- Remove `position` and `velocity` assumptions from the base class.
- Keep translational semantics as an opt-in convention implemented by subclasses that need them, not as a requirement of `Mover` itself.
- If convenience accessors are still desirable, place them on translational subclasses rather than on the generic base class.

2. Generalize integration in the engine.
- Change `SimulationContext` to store `(start, size)` or `slice` per mover instead of only a start index.
- Change `SimulationContext.get_state()` to return the mover's full state slice.
- Change `SimulationEngine.step_continuous()` so it only passes full state slices and never hard-splits them into `pos` and `vel`.
- Standardize the engine/mover boundary on a generic derivative API such as `compute_state_derivative(t, state)`.
- Preserve compatibility by introducing a translational/Newtonian subclass layer that can still interpret portions of state as position/velocity where appropriate.

3. Introduce a translational subclass layer for compatibility.
- Add a subclass such as `TranslationalMover` or equivalent that reintroduces position/velocity conventions for existing point-mass movers.
- Move helpers like `position`, `velocity`, and any translational derivative packing into that subclass layer instead of `Mover`.
- Update existing movers that truly are translational to inherit from that layer.

4. Keep analytical movers generic, not aircraft-specific.
- Allow analytical movers to return arbitrary-length states.
- Leave `SplineMover` and `WaypointMover` translational unless there is a strong reason to change their public behavior.
- Add a new `AircraftSplineMover` rather than overloading the generic spline mover.

5. Add orientation/math support.
- Create quaternion helpers in a new math module, likely under `mover_sim/math/`.
- Needed primitives:
  - normalize quaternion
  - quaternion multiply
  - quaternion from orthonormal basis
  - rotate vector by quaternion
  - quaternion derivative from body rates
- Add frame helpers for building aircraft body axes from path tangent, curvature, and local vertical.

6. Refactor `NewtonianMover` to be a thin dynamics base, not a force-policy class.
- Remove built-in assumptions about gravity and Coriolis from `NewtonianMover`.
- Make `NewtonianMover` responsible only for generic continuous-state integration patterns needed by Newtonian-style subclasses.
- Require subclasses to explicitly add whichever forces they need, including:
  - gravity
  - Coriolis / rotating-frame terms
  - aerodynamic forces
  - thrust or externally modeled accelerations
- This keeps the base class reusable for inertial, rotating-frame, terrestrial, and fully custom dynamics models.

7. Implement `AircraftSplineMover`.
- Recommended analytical state:
  - `[x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]`
- Use spline first derivative for forward/tangent direction.
- Use spline second derivative to estimate curvature and required lateral/normal acceleration.
- Derive:
  - yaw from tangent heading,
  - pitch from climb/descent angle,
  - roll from curvature / lateral acceleration relative to gravity.
- Handle degenerate cases explicitly:
  - near-zero speed,
  - zero curvature,
  - endpoints where derivatives go to zero.
- If angular rates are needed in state, derive them from orientation change over time.

8. Implement a rigid-body Newtonian aircraft mover.
- Add a new mover, preferably `Aircraft6DOFMover` or `RigidAircraftMover`.
- Do not mutate the current `AircraftMover` first; keep it as the existing point-mass model until the new path is proven.
- State:
  - position, velocity, quaternion, body rates
- Dynamics:
  - translational forces in world frame
  - rotational moments in body frame
  - inertia tensor
  - quaternion kinematics
- Have the subclass add gravity and any rotating-frame corrections explicitly rather than inheriting them from `NewtonianMover`.
- Control inputs should become moment/throttle oriented, not just `bank_angle_cmd` and scalar lift.
- A minimal first version can use simplified aerodynamic coefficients and commanded pitch/roll/yaw moments before building a full aero model.

9. Add aircraft-specific controllers separately.
- Keep the current `AircraftAutopilot` for the existing point-mass mover.
- Add a new controller for the rigid-body mover that maps guidance error into:
  - throttle
  - roll moment / roll rate command
  - pitch moment / pitch rate command
  - yaw moment / yaw rate or sideslip coordination
- This avoids destabilizing current examples/tests.

10. Update observers and telemetry.
- Keep current `CSVLogger` translational columns for compatibility.
- Extend it to append optional orientation/rate columns when present.
- Suggested optional columns:
  - quaternion components
  - body rates
  - possibly derived Euler angles for readability only

11. Expand tests in phases.
- Core tests:
  - mixed movers with different state dimensions
  - context slice correctness
  - generic state derivative packing/unpacking
- Analytical aircraft spline tests:
  - straight segment gives near-zero roll/pitch
  - climbing path gives pitch
  - coordinated turn path gives roll
- Newtonian base/subclass tests:
  - base class does not inject gravity implicitly
  - subclasses can add gravity explicitly
  - subclasses can omit gravity for fully custom dynamics
- Newtonian rigid-body tests:
  - roll command changes bank
  - pitch command changes flight path
  - yaw/roll coupling changes trajectory
  - quaternion remains normalized
- Regression tests:
  - current `NewtonianMover`, `WaypointMover`, `SplineMover`, and existing aircraft tests remain green

12. Update docs after core refactor is stable.
- Update `README.md`, `docs/user-guide.md`, and `docs/architecture.md`.
- Document that:
  - engine state is arbitrary-dimensional,
  - base movers are state-centric,
  - translational views are provided by translational subclasses rather than the generic core,
  - rigid-body aircraft use 13 state elements,
  - analytical and Newtonian movers can now expose additional state.

## Suggested Delivery Order

1. Core arbitrary-dimensional state refactor in `mover.py` and `engine.py`.
2. Introduce translational compatibility subclasses and keep existing tests green.
3. Refactor `NewtonianMover` so force terms are subclass-defined.
4. Quaternion/math utilities.
5. `AircraftSplineMover`.
6. `Aircraft6DOFMover`.
7. Rigid-body aircraft controller.
8. Telemetry/doc updates.

## Key Design Choice

The important boundary is:

- make the engine generic about state length,
- keep movers responsible for interpreting their own state layout,
- keep force-model choices in concrete subclasses rather than in shared base dynamics classes.

That is the smallest refactor that unlocks both requested use cases without turning every subsystem into an aircraft dynamics framework or preserving hidden assumptions about translational state semantics.

## Risk Notes

- Replacing the current `AircraftMover` directly would likely break existing behavior and tests.
- Using Euler angles as integrated state will create avoidable singularity issues.
- Making observers fully generic on day one is not necessary; optional extra columns is a safer first step.
- Removing gravity/Coriolis from `NewtonianMover` will require a deliberate compatibility pass so existing subclasses continue to produce the same trajectories.

If implemented in that order, the core refactor is moderate, while the rigid-body aircraft model is the largest and highest-risk part.
