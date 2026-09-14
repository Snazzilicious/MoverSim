# RocketMover Implementation Plan

Goal: replace the `RocketMover` brainstorming stub in `mover_sim/models/aircraft_mover.py` with a production implementation that follows `FixedWingMover` structurally while adding rocket-specific mass depletion, stage separation helpers, and cylindrically symmetric rotational dynamics.

## Design Constraints

- Keep the matrix-attitude architecture used by `FixedWingMover`.
- Add an integrated propellant-mass state for the active stage.
- Treat stage separation as a discrete event or committed-state rewrite triggered externally by a controller.
- Model cylindrical symmetry rather than separate pitch, roll, and yaw aerodynamics.
- Use a steering-vector command rather than separate roll, pitch, and yaw commands.

## Step 1: Replace the Stub Skeleton

Objective: remove the incompatible brainstorming code and create the real class structure.

Tasks:
- Replace the current `RocketMover` stub entirely.
- Mirror the overall class structure of `FixedWingMover`.
- Add a nested `OrientationCorrectionEvent` matching the `FixedWingMover` pattern.
- Reuse the same state organization style: position, velocity, orientation matrix, body angular velocity, plus active propellant mass.

Completion check:
- `RocketMover` has a valid constructor, state layout, and slice helpers.
- No placeholder methods or undefined helper references remain.

## Step 2: Define the Rocket State and Public Interface

Objective: lock down the state layout and command interface before implementing dynamics.

State layout:
- `[x, y, z, vx, vy, vz, o11, ..., o33, wx, wy, wz, m_prop]`

Tasks:
- Add `get_orientation_slice`, `get_omega_slice`, and `get_propellant_mass_slice` helpers.
- Add properties for `orientation`, `omega_body`, and active propellant mass.
- Replace roll/pitch/yaw commands with:
  - `thrust_cmd` as a 0-100 percent command
  - `steer_cmd` as a 0-100 percent command
  - `steer_direction_body` as a body-frame unit vector perpendicularized during use

Completion check:
- The class exposes a stable interface that a future controller can drive directly.

## Step 3: Add Stage Definitions and Bookkeeping

Objective: support active-stage mass, aerodynamic, and rotational properties.

Tasks:
- Define the required stage fields.
- Validate stage dictionaries in the constructor or helper functions.
- Store `self.stages` and `self.active_stage_index`.
- Track attached dry mass outside the active stage so total mass can be derived correctly.
- Provide helper methods such as:
  - `_active_stage()`
  - `current_total_mass(propellant_mass=None)`
  - `current_mass_flow_rate(t, propellant_mass=None)`
  - `current_stage_thrust(t, propellant_mass=None)`
  - `has_active_burn(propellant_mass=None)`

Notes:
- The mover should be responsible for computing mass and force properties from stage data.
- The controller should be responsible for deciding when to call stage-separation helpers.

Completion check:
- The mover can answer what stage is active, how much mass remains, and whether thrust is still available.

## Step 4: Add Stage-Separation Helper Functions

Objective: let a controller trigger clean stage transitions without embedding phase logic in the mover derivative.

Tasks:
- Add a helper that determines whether separation is possible.
- Add a helper that advances to the next stage.
- Update attached dry mass, aerodynamic coefficients, reference area, and rotational parameters on separation.
- Reset the committed active propellant-mass state to the next stage's initial propellant mass.
- Ensure separation is safe both before registration and after registration with an engine context.

Recommended helper shape:
- `can_separate_stage()`
- `separate_stage()` or `advance_to_next_stage()`

Completion check:
- A controller can trigger stage separation with one call and the mover switches to the next stage cleanly.

## Step 5: Implement Symmetric Force Models

Objective: replace fixed-wing-specific aerodynamics with rocket-appropriate symmetric forces.

Tasks:
- Add `_thrust_force_world(...)` using the body forward axis.
- Add `_drag_force_world(...)` using the current stage drag properties.
- Add a symmetric normal-force model driven by the component of air-relative velocity perpendicular to the forward axis.
- Avoid separate alpha/beta/bank logic.

Modeling guidance:
- Drag should oppose velocity.
- Normal aerodynamic force should oppose misalignment between body axis and airflow.
- The force model should be invariant to rotation about the rocket longitudinal axis.

Completion check:
- Translational forces depend on airspeed and axial misalignment, not named aircraft axes.

## Step 6: Implement Symmetric Rotational Dynamics

Objective: keep full rigid-body attitude dynamics while enforcing cylindrical symmetry.

Tasks:
- Use a full 3x3 rotational-mass matrix or equivalent, but default it so transverse moments are equal.
- Use equal transverse angular damping by default.
- Add a steering moment computed from `steer_cmd` and `steer_direction_body`.
- Remove any roll-specific restoring logic unless there is a deliberate spin or roll-control feature.

Modeling guidance:
- Steering moment should act perpendicular to the body forward axis.
- Angular response should be the same for equal commands in any transverse direction.

Completion check:
- The rotational model preserves symmetry around the rocket axis.

## Step 7: Implement `compute_state_derivative`

Objective: connect the full dynamics into one integrated state derivative.

Tasks:
- Unpack position, velocity, orientation, body angular velocity, and propellant mass.
- Project orientation back onto `SO(3)` before use.
- Compute:
  - `dpos = vel`
  - `dorientation = orientation @ skew(omega_body)`
  - optional Earth-frame correction terms matching `FixedWingMover`
  - translational acceleration from gravity, optional centrifugal/Coriolis terms, and rocket forces divided by current total mass
  - rotational acceleration from steering moment and damping
  - `dm_prop/dt = -mass_flow_rate`
- Return the concatenated derivative.

Important rule:
- Stage separation itself should not happen inside `compute_state_derivative`.

Completion check:
- The mover can be numerically integrated without any external patching except explicit stage-separation events.

## Step 8: Add Orientation Correction Support

Objective: keep the matrix attitude numerically well-conditioned during long runs.

Tasks:
- Implement `OrientationCorrectionEvent` like `FixedWingMover`.
- Add `add_orientation_correction_event` helper.
- Project the committed orientation slice onto `SO(3)`.

Completion check:
- The rocket mover can schedule the same sort of periodic orientation cleanup as `FixedWingMover`.

## Step 9: Add Unit Tests for the Mover Core

Objective: lock down the design before adding guidance logic.

Tests to add:
- Constructor/state-shape test.
- Orientation correction event test.
- Active propellant mass decreases during burn.
- Total acceleration changes as propellant mass decreases under fixed thrust.
- `current_stage_thrust` and `current_mass_flow_rate` behave correctly at burnout.
- Stage separation updates stage properties and active propellant state.
- Symmetry test: equal transverse steering commands in different directions produce equivalent angular response magnitudes.

Completion check:
- The mover behavior is covered independently of any autopilot.

## Step 10: Add a Controller Later, Separately

Objective: keep guidance concerns separate from core rigid-body physics.

Tasks:
- Add a controller only after the mover and tests are stable.
- Let the controller manage burnout timing, steering commands, and stage separation calls.
- Reuse the mover helper methods rather than duplicating mass or stage logic.

Completion check:
- Guidance work is clearly separated from mover implementation work.

## Resume Checklist

If work stops and later resumes, continue in this order:

1. Step 1 and Step 2: class skeleton, state layout, and command interface.
2. Step 3 and Step 4: stage bookkeeping and separation helpers.
3. Step 5 through Step 8: force model, rotational model, derivative, orientation correction.
4. Step 9: tests.
5. Step 10: controller work only after mover validation is complete.

## Suggested First Coding Slice

Start with the smallest end-to-end vertical slice:

1. Replace the stub with the final state layout and constructor.
2. Implement slice helpers and properties.
3. Implement stage bookkeeping helpers.
4. Add a basic thrust + drag + gravity derivative with propellant mass depletion.
5. Run tests and then expand into steering and stage separation.
