## Objective

Add a simple, effective controller for `RocketMover` focused on ballistic-missile use cases, while simplifying `RocketMover` by removing `thrust_profile` support.

## Scope

- Keep the dynamics model simple and tunable rather than high-fidelity.
- Use constant thrust per stage.
- Add a phase-based controller for boost, pitch-over, ascent, separation, and coast.
- Preserve the existing `RocketMover` control interface:
  - `thrust_cmd`
  - `steer_cmd`
  - `steer_direction_body`

## Stop / Resume Strategy

Each step below is intended to leave the repo in a coherent state. Work can stop after any completed step and resume from the next unchecked step.

## Implementation Steps

### Step 1: Simplify `RocketMover` stage schema

Goal:
- Remove `thrust_profile` support from `RocketMover` stage validation and thrust lookup.

Changes:
- Update stage validation in `mover_sim/models/aircraft_mover.py` so each stage requires constant `thrust`.
- Remove `_validate_thrust_profile(...)`.
- Remove any branching that accepts `thrust_profile`.
- Simplify `current_stage_thrust(...)` to return constant thrust for the active stage, scaled by `thrust_cmd`.

Checkpoint:
- `RocketMover` no longer references `thrust_profile`.

Validation:
- Search the file for `thrust_profile` and confirm there are no remaining references in `RocketMover`.

### Step 2: Align ballistic scenario helpers with the simplified stage model

Goal:
- Remove `thrust_profile` assumptions from `mover_sim/scenario_ballistic_missile.py`.

Changes:
- Update stage validation helpers in the scenario file to require constant `thrust` only.
- Remove scenario-side code that interpolates or reads `thrust_profile`.
- Keep stage timing based on `burn_duration` or `mass_flow_rate`, whichever path already fits the simplified model best.

Checkpoint:
- The ballistic scenario uses the same constant-thrust assumption as `RocketMover`.

Validation:
- Search the repo for `thrust_profile` and confirm any remaining references are intentional or remove them.

### Step 3: Define the reusable controller shape

Goal:
- Introduce a simple controller structure that matches `RocketMover` rather than aircraft autopilots.

Changes:
- Add a new controller class near `RocketMover` in `mover_sim/models/aircraft_mover.py` or in a nearby module if separation is cleaner.
- Keep the controller phase-based, with a small explicit state machine.
- Initial recommended phases:
  - `BOOST_VERTICAL`
  - `PITCH_OVER`
  - `POWERED_ASCENT`
  - `STAGE_SEPARATION`
  - `BALLISTIC_COAST`
  - `IMPACT`

Checkpoint:
- Controller class exists with phase constants, constructor parameters, and an `update(...)` skeleton.

Validation:
- Controller initializes cleanly and type-checks against `RocketMover`.

### Step 4: Add simple pointing guidance

Goal:
- Implement a minimal but effective steering law.

Changes:
- Add helper(s) that compute a desired forward direction in world coordinates.
- Compare current forward axis to desired forward axis.
- Convert the resulting pointing error into:
  - `steer_direction_body`
  - `steer_cmd`
- Add basic rate damping using `mover.omega_body` so steering does not chatter.

Recommended behavior:
- `BOOST_VERTICAL`: point close to local up.
- `PITCH_OVER`: gradually rotate toward downrange azimuth.
- `POWERED_ASCENT`: hold a scheduled pitch angle.
- `BALLISTIC_COAST`: set `thrust_cmd = 0` and reduce steering to zero or near-zero.

Checkpoint:
- Controller can command stable boost and pitch-over behavior using only the existing rocket control inputs.

Validation:
- Run a short simulation and inspect that steering commands stay bounded and phase transitions occur.

### Step 5: Add simple ascent-program inputs

Goal:
- Keep the controller tunable without overcomplicating it.

Recommended inputs:
- `target_position_ecef`
- `launch_azimuth` (optional if derived from launch/target geometry)
- `vertical_rise_time` or `vertical_rise_altitude`
- `pitch_over_duration`
- `target_ascent_pitch`
- `update_interval`
- `steer_kp`
- `steer_kd`
- `separation_delay`

Changes:
- Add only the smallest set needed for the first ballistic use case.
- Prefer time-based scheduling first; add altitude-based triggers only where useful.

Checkpoint:
- Controller constructor parameters are clear, minimal, and sufficient to tune a missile shot.

Validation:
- Instantiate the controller from a scenario without ad hoc hardcoded values inside `update(...)`.

### Step 6: Implement burnout and stage separation flow

Goal:
- Use `RocketMover`'s built-in stage support rather than duplicating mass changes in the controller.

Changes:
- Detect burnout from active propellant depletion or zero available mass flow.
- Set `thrust_cmd = 0` during separation delay.
- Call `mover.advance_to_next_stage(engine)` when separation should occur.
- Transition to `BALLISTIC_COAST` when no additional stage remains.

Checkpoint:
- The controller owns the timing and phase transition, while the mover owns the stage state mutation.

Validation:
- Verify stage index changes, propellant resets for the next stage, and thrust resumes only when appropriate.

### Step 7: Add simple event publishing

Goal:
- Keep scenario observability without introducing heavy orchestration.

Recommended events:
- `stage_burnout`
- `stage_separation`
- `ballistic_coast_start`
- `ground_impact`

Changes:
- Publish events from the controller when phase transitions occur.
- Avoid duplicate publishes by tracking per-event flags or phase-transition guards.

Checkpoint:
- Major mission milestones are externally visible.

Validation:
- Confirm each event fires once in a representative run.

### Step 8: Add impact / end-of-flight handling - Next step

Goal:
- Ensure the controller terminates cleanly once the missile hits the ground.

Changes:
- Detect ground impact from altitude.
- Zero commands on impact.
- Publish impact event.
- Stop the engine if that matches the existing scenario behavior.

Checkpoint:
- No further boost or steering commands are issued after impact.

Validation:
- Run until impact and confirm the final phase is stable.

### Step 9: Verify with a minimal ballistic scenario

Goal:
- Prove the simplified controller is usable end-to-end.

Changes:
- Update an existing ballistic scenario or add a minimal configuration path using `RocketMover` plus the new controller.
- Use one-stage first, then two-stage if needed.

Checkpoint:
- A ballistic missile can launch, boost, separate stages if configured, coast, and impact.

Validation:
- Run the relevant scenario/test command and inspect trajectory sanity at a high level.

### Step 10: Optional cleanup after behavior is working

Goal:
- Remove dead code and tighten naming once the first version is verified.

Changes:
- Delete any superseded helpers from the older ballistic implementation if they are no longer used.
- Tighten docstrings around the simplified stage model.
- Keep the public API small.

Checkpoint:
- No leftover `thrust_profile` code paths remain.

Validation:
- Final repo search for old fields, duplicate guidance logic, and stale comments.

## Recommended Execution Order

1. Step 1
2. Step 2
3. Step 3
4. Step 4
5. Step 5
6. Step 6
7. Step 7
8. Step 8
9. Step 9
10. Step 10

## First Practical Milestone

The first usable milestone is reached after Step 6:
- constant-thrust stages,
- simple ballistic controller,
- boost/pitch-over/ascent/coast,
- working stage separation.

At that point, event polish and scenario cleanup can continue incrementally.
