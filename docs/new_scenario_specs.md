
### Requirements for new scenarios

We would like to model the following scenarios to generate datasets for AI model training.

Each scenario should be parameterized and wrapped in its own function so we can run parameter sweeps.

### Common requirements

- Use `HDF5Logger` for trajectory export.
- Use ECEF and SI units for simulation inputs and outputs unless otherwise noted below.
- Leave `sample_interval` as a scenario function parameter.
- Each scenario function should accept at least:
  - scenario-specific parameters
  - `t_end`
  - `sample_interval`
  - `output_path`
- Use a fixed-duration run model. Each scenario must accept an explicit `t_end` and run until that time unless the implementation documents a scenario-specific early stop condition.

### Common state/output contract

- All movers in these scenarios must export full attitude, position, and velocity trajectory data.
- Use the following 13-element state convention for every mover:
  - `[x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]`
- This requirement applies to all movers, including:
  - primary vehicles
  - motherships
  - released missiles
  - ballistic stages
  - spent stages after separation
- If a mover is not actively controlled during a phase, it must still provide attitude and body-rate history.

### Common input conventions

- `position` values are ECEF Cartesian coordinates in meters.
- `velocity` values are ECEF Cartesian velocities in meters/second.
- `heading` inputs are interpreted as local ENU azimuth angles at the relevant reference position:
  - for initial vehicle headings, use the local tangent plane at the initial position
  - for air-launched missile release headings, use the local tangent plane at the release position
- `altitude` values are meters above the WGS-84 ellipsoid.
- All times are in seconds.

### Scenario 1: Surface-Launched Cruise Missile

- Required input parameters:
  - `initial_position_ecef`
  - `cruise_speed`
  - `cruise_altitude`
  - `cruise_heading`
  - `boost_duration`
  - `boost_acceleration`
  - `launch_pitch_angle`
- Behavior:
  - Model a high-level timed rocket-assisted launch/boost phase.
  - After boost completion, transition to an aircraft-like climb/cruise profile.
  - Cruise heading is commanded using the local ENU azimuth convention defined above.
  - Cruise altitude is the commanded post-transition altitude.
- Movers to log:
  - the cruise missile
- Notes:
  - The launch/boost phase is parameterized at a high level rather than by full booster mass/thrust properties.

#### Scenario 1 Design Outline

- Scenario function:
  - `run_surface_launched_cruise_missile_scenario(...)`
- Recommended function signature:
  - `run_surface_launched_cruise_missile_scenario(initial_position_ecef, cruise_speed, cruise_altitude, cruise_heading, boost_duration, boost_acceleration, launch_pitch_angle, t_end, sample_interval, output_path)`
- Primary mover implementation:
  - use a dedicated `SurfaceLaunchedCruiseMissileMover` built on the common 13-state rigid-body contract
  - use `Aircraft6DOFMover` behavior as the baseline for state layout, attitude propagation, and world-frame force integration
- Controller implementation:
  - attach a dedicated controller such as `SurfaceLaunchedCruiseMissileController`
  - the controller should be phase-based and update thrust and attitude commands according to the active flight phase

- Flight phases:
  - `boost`
  - `transition_to_cruise`
  - `cruise`
  - `impact_terminal`

- Initial conditions:
  - initialize position from `initial_position_ecef`
  - initialize orientation from local ENU heading plus `launch_pitch_angle`
  - initialize body rates to zero unless the implementation later needs a small nonzero trim value
  - initialize translational velocity to exactly zero at scenario start
  - let the boost phase generate all forward motion from rest

- Boost phase design:
  - active for `0 <= t < boost_duration`
  - apply forward thrust consistent with `boost_acceleration`
  - command the missile to hold the launch attitude defined by `launch_pitch_angle` and the commanded heading during early boost
  - allow gravity, drag, and Coriolis to act during the boost phase
  - use time only as the boost-to-transition criterion

- Transition-to-cruise phase design:
  - begins exactly at `t = boost_duration`
  - hand off from boost attitude holding to aircraft-like guidance
  - command the missile to align with the target cruise heading in the local ENU tangent plane
  - command climb or descent toward `cruise_altitude`
  - accelerate or decelerate toward `cruise_speed`
  - this phase may be implemented as a short internal control mode even if it is not separately parameterized in the scenario API

- Cruise phase design:
  - hold a straight commanded course rather than navigating to a waypoint
  - regulate toward:
    - `cruise_heading`
    - `cruise_altitude`
    - `cruise_speed`
  - continue exporting full 13-state telemetry through `t_end`

- Guidance/control approach:
  - compute heading errors in the local ENU frame at the current missile position
  - convert heading and altitude objectives into roll, pitch, yaw, and thrust commands compatible with the 13-state mover
  - prefer a simple phase-based autopilot rather than a waypoint follower for the first implementation
  - keep the design minimal: one mover class and one controller class are sufficient for the first version

- Ground impact handling:
  - if the missile reaches the ground before `t_end`, treat that as scenario termination
  - publish an impact event if event logging is enabled
  - no frozen post-impact state is required for Scenario 1

- Logging/output design:
  - register one platform for the missile
  - attach `HDF5Logger(engine, output_path, sample_interval=sample_interval, include_events=True)`
  - record at minimum:
    - full state history
    - derived position/velocity datasets
    - LLA datasets
    - orientation and body-rate datasets
    - optional event records for phase changes and impact

- Suggested event topics:
  - `boost_start`
  - `boost_end`
  - `cruise_transition_start`
  - `cruise_established`
  - `ground_impact`

- Minimal implementation sequence:
  - create the scenario function and parameter validation
  - implement `SurfaceLaunchedCruiseMissileMover` with 13-state dynamics
  - implement `SurfaceLaunchedCruiseMissileController` with time-based phase switching
  - add event publication for phase changes and impact
  - attach `HDF5Logger`
  - add one example script and focused tests for phase transitions, heading interpretation, and output shape

#### Scenario 1 Implementation Checklist

1. Create a new scenario module and add `run_surface_launched_cruise_missile_scenario(...)` with the required parameters plus `t_end`, `sample_interval`, and `output_path`.
2. Add input validation for `initial_position_ecef`, `cruise_speed`, `cruise_altitude`, `cruise_heading`, `boost_duration`, `boost_acceleration`, `launch_pitch_angle`, `t_end`, and `sample_interval`.
3. Choose and document the initial translational speed policy for launch:
   - use exactly zero translational speed at scenario start
4. Implement a helper that converts `cruise_heading` and `launch_pitch_angle` from the local ENU frame at `initial_position_ecef` into an initial ECEF orientation quaternion.
5. Implement `SurfaceLaunchedCruiseMissileMover` using the same 13-state layout as `Aircraft6DOFMover`.
6. Reuse or adapt the existing rigid-body world-frame force model structure so the mover includes gravity, drag, Coriolis, quaternion propagation, and body-rate propagation.
7. Add mover parameters or internal constants needed for Scenario 1 flight dynamics, including mass, drag/reference area, maximum thrust behavior, and any angular damping values.
8. Implement boost thrust behavior so the mover can apply forward acceleration consistent with `boost_acceleration` during the boost phase.
9. Implement `SurfaceLaunchedCruiseMissileController` as a phase-based controller attached to the missile platform.
10. Add controller phase state for:
    - `boost`
    - `transition_to_cruise`
    - `cruise`
    - `impact_terminal`
11. Implement boost-phase control logic that holds the commanded launch heading and `launch_pitch_angle` until `t >= boost_duration`.
12. Publish a `boost_start` event when the scenario begins and a `boost_end` event when the boost phase completes.
13. Implement the boost-to-transition handoff strictly as a time-based transition at `t = boost_duration`.
14. Implement transition-phase control logic that turns the missile toward `cruise_heading`, drives altitude toward `cruise_altitude`, and adjusts thrust toward `cruise_speed`.
15. Publish a `cruise_transition_start` event when transition guidance begins.
16. Define a simple internal criterion for when cruise is considered established and publish `cruise_established` once that criterion is first met.
17. Implement cruise-phase control logic that holds straight flight on the commanded heading while regulating toward `cruise_altitude` and `cruise_speed`.
18. Add ground-impact detection using current position/altitude so the scenario can detect contact with the Earth before `t_end`.
19. On ground impact, publish `ground_impact` and terminate the scenario rather than freezing the state.
20. Build the scenario assembly flow:
    - create `SimulationEngine`
    - construct the missile mover and controller
    - wrap them in a `Platform`
    - register the platform with the engine
21. Attach `HDF5Logger(engine, output_path, sample_interval=sample_interval, include_events=True)`.
22. Ensure the output contains the required datasets for time, full state, position, velocity, LLA, orientation, and body rates.
23. Add a focused unit test for initial heading/pitch conversion from local ENU inputs to the initial quaternion.
24. Add a focused unit test that verifies the mover exposes the required 13-element state layout.
25. Add a controller test that verifies the missile remains in boost before `boost_duration` and transitions out of boost at the correct time.
26. Add a controller or integration test that verifies cruise guidance reduces heading and altitude error after transition.
27. Add a scenario-level test that verifies ground impact causes scenario termination and emits `ground_impact`.
28. Add a logger test that verifies the HDF5 output for this scenario contains orientation and body-rate datasets.
29. Add an example script that runs Scenario 1 with representative parameters and writes an HDF5 file.
30. Run the relevant test subset and confirm the scenario executes end-to-end with valid output.

### Scenario 2: Air-Launched Cruise Missile

- Required input parameters:
  - `mothership_initial_position_ecef`
  - `mothership_cruise_speed`
  - `mothership_cruise_altitude`
  - `mothership_cruise_heading`
  - `mothership_rtb_position_ecef`
  - `missile_launch_time`
  - `missile_cruise_speed`
  - `missile_cruise_altitude`
  - `missile_cruise_heading`
  - `missile_drop_duration`
- Behavior:
  - Derive missile release position and release state from the mothership state at `missile_launch_time`.
  - After release, the missile should freefall for `missile_drop_duration`.
  - After the freefall phase, the missile should ignite and transition to its commanded cruise behavior.
  - After weapon release, the mothership should execute RTB behavior toward `mothership_rtb_position_ecef`.
  - Heading inputs use the local ENU azimuth convention defined above.
- Movers to log:
  - the mothership
  - the released missile

### Scenario 3: Ballistic Missile

- Required input parameters:
  - `initial_position_ecef`
  - `target_position_ecef`
  - `peak_altitude`
  - `stages`
- `stages` must support a configurable one-stage or two-stage stack.
- Each stage definition must provide full physical stage parameters, including at minimum:
  - `dry_mass`
  - `propellant_mass`
  - `burn_duration`
  - `thrust` or an equivalent thrust-time profile
  - `drag_coefficient`
  - `reference_area`
  - `separation_delay`
- Behavior:
  - Model powered ascent, stage separation, coast/ballistic flight, and terminal descent.
  - Spent stages must separate into their own movers and fall back under gravity/aerodynamic effects.
  - Spent stages should continue to be tracked until ground impact.
  - After a spent stage hits the ground, freeze its state rather than removing it from the scenario.
- Movers to log:
  - the active missile body
  - each separated spent stage
