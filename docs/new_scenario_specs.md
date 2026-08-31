
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

#### Scenario 2 Design Outline

- Scenario function:
  - `run_air_launched_cruise_missile_scenario(...)`
- Recommended function signature:
  - `run_air_launched_cruise_missile_scenario(mothership_initial_position_ecef, mothership_cruise_speed, mothership_cruise_altitude, mothership_cruise_heading, mothership_rtb_position_ecef, missile_launch_time, missile_cruise_speed, missile_cruise_altitude, missile_cruise_heading, missile_drop_duration, t_end, sample_interval, output_path)`

- Primary mover implementation:
  - represent both the mothership and the released missile with dedicated 13-state rigid-body movers
  - use `Aircraft6DOFMover` behavior as the baseline for state layout, attitude propagation, and world-frame force integration
  - the mothership and missile may share common helper logic, but should remain separate mover/controller pairs

- Controller implementation:
  - attach a dedicated controller such as `AirLaunchedCruiseMissileMothershipController` to the mothership
  - attach a dedicated controller such as `AirLaunchedCruiseMissileController` to the released missile after spawn
  - use engine scheduling to trigger missile release at `missile_launch_time`

- Flight phases:
  - mothership:
    - `pre_release_cruise`
    - `post_release_rtb`
    - `rtb_hold`
  - missile:
    - `drop`
    - `ignite_transition`
    - `cruise`
    - `impact_terminal`

- Initial conditions:
  - initialize the mothership at `mothership_initial_position_ecef`
  - initialize the mothership orientation from local ENU heading plus a level cruise pitch assumption unless later parameterized
  - initialize the mothership translational velocity from `mothership_cruise_speed` and `mothership_cruise_heading`
  - initialize mothership body rates to zero unless later trim behavior is needed
  - do not create the missile platform at scenario start

- Release event design:
  - schedule a release event at `missile_launch_time`
  - at release time, derive the missile initial position, velocity, and attitude directly from the current mothership state
  - the missile release state should match the mothership state at release rather than applying a separate offset or separation impulse in the first implementation
  - dynamically register the missile platform with the engine at release time

- Mothership pre-release design:
  - command the mothership to hold its cruise heading, cruise altitude, and cruise speed before release
  - use local ENU heading guidance at the current aircraft position
  - continue exporting full 13-state telemetry before and after release

- Missile drop-phase design:
  - active for `missile_launch_time <= t < missile_launch_time + missile_drop_duration`
  - after release, the missile should freefall for `missile_drop_duration`
  - during the drop phase, keep the missile attitude consistent with the release state or simple attitude hold behavior rather than adding a detailed separation aerodynamics model
  - allow gravity, drag, and Coriolis to act during the drop phase
  - no powered thrust should be applied during the drop phase

- Missile ignite-transition design:
  - begins exactly at the end of the drop phase
  - transition from unpowered drop to powered missile guidance
  - command the missile toward `missile_cruise_heading`, `missile_cruise_altitude`, and `missile_cruise_speed`
  - this phase may be implemented as a short internal capture mode before cruise is declared established

- Missile cruise-phase design:
  - hold a straight commanded course rather than navigating to a waypoint in the first implementation
  - regulate toward:
    - `missile_cruise_heading`
    - `missile_cruise_altitude`
    - `missile_cruise_speed`
  - continue exporting full 13-state telemetry through `t_end` or until missile impact terminates the scenario

- Mothership RTB design:
  - after weapon release, switch the mothership from pre-release cruise to direct RTB guidance toward `mothership_rtb_position_ecef`
  - use a simple direct-to-point guidance law for the first implementation rather than a multi-waypoint return route
  - after the mothership reaches the RTB point, transition to a hold-near-base behavior and continue logging through the fixed-duration run

- Guidance/control approach:
  - compute heading errors in the local ENU frame at the current mover position
  - use simple phase-based autopilots for both mothership and missile rather than waypoint-following route planners in the first implementation
  - keep the design minimal: one mover/controller pair for the mothership and one mover/controller pair for the missile
  - share helper logic for heading, altitude, speed, and initial orientation conversion where practical

- Impact handling:
  - if the released missile reaches the ground before `t_end`, publish a missile impact event and terminate the full scenario
  - no frozen post-impact missile state is required for Scenario 2
  - if the mothership reaches RTB before `t_end`, it should hold near base and continue logging rather than ending the scenario

- Logging/output design:
  - register one platform for the mothership at scenario start
  - dynamically register the missile platform at release time
  - attach `HDF5Logger(engine, output_path, sample_interval=sample_interval, include_events=True)`
  - record at minimum:
    - full state history for both movers
    - derived position/velocity datasets
    - LLA datasets
    - orientation and body-rate datasets
    - event records for release, drop completion, ignition, RTB transition, RTB arrival, cruise establishment, and impact

- Suggested event topics:
  - `platform_registered`
  - `missile_release`
  - `missile_drop_start`
  - `missile_drop_end`
  - `missile_ignite`
  - `missile_cruise_established`
  - `mothership_rtb_start`
  - `mothership_rtb_arrival`
  - `missile_ground_impact`

- Minimal implementation sequence:
  - create the scenario function and parameter validation
  - implement a mothership mover/controller pair for pre-release cruise and RTB
  - implement a missile mover/controller pair for drop, ignition transition, and cruise
  - schedule the release event at `missile_launch_time`
  - spawn and register the missile from the current mothership state at release
  - add event publication for release, drop completion, ignition, RTB transition/arrival, cruise establishment, and impact
  - attach `HDF5Logger` with the Scenario 2 event topics
  - add one example script and focused tests for dynamic missile registration, drop-to-ignite timing, RTB behavior, and mixed-platform HDF5 output

#### Scenario 2 Implementation Checklist

1. Create a new scenario module and add `run_air_launched_cruise_missile_scenario(...)` with the required parameters plus `t_end`, `sample_interval`, and `output_path`.
2. Add input validation for `mothership_initial_position_ecef`, `mothership_cruise_speed`, `mothership_cruise_altitude`, `mothership_cruise_heading`, `mothership_rtb_position_ecef`, `missile_launch_time`, `missile_cruise_speed`, `missile_cruise_altitude`, `missile_cruise_heading`, `missile_drop_duration`, `t_end`, and `sample_interval`.
3. Implement or reuse a helper that converts local ENU heading inputs at a mover position into an initial ECEF orientation quaternion.
4. Implement or reuse a helper that converts commanded heading and speed into an initial ECEF velocity vector for the mothership.
5. Implement `AirLaunchedCruiseMissileMothershipMover` using the standard 13-state layout.
6. Reuse or adapt the existing rigid-body world-frame force model structure so the mothership mover includes gravity, drag, Coriolis, quaternion propagation, and body-rate propagation.
7. Add mothership-specific mover parameters or internal constants, including mass, drag/reference area, maximum thrust behavior, and any angular damping values.
8. Implement `AirLaunchedCruiseMissileMover` for the released missile using the same 13-state layout.
9. Reuse or adapt the same rigid-body force-model structure for the missile mover.
10. Add missile-specific mover parameters or internal constants, including mass, drag/reference area, powered-thrust behavior, and any angular damping values.
11. Implement `AirLaunchedCruiseMissileMothershipController` as a phase-based controller attached to the mothership platform.
12. Add mothership controller phase state for:
    - `pre_release_cruise`
    - `post_release_rtb`
    - `rtb_hold`
13. Implement mothership pre-release cruise logic that holds `mothership_cruise_heading`, `mothership_cruise_altitude`, and `mothership_cruise_speed`.
14. Implement `AirLaunchedCruiseMissileController` as a phase-based controller attached to the released missile platform.
15. Add missile controller phase state for:
    - `drop`
    - `ignite_transition`
    - `cruise`
    - `impact_terminal`
16. Implement a release-event callback scheduled at `missile_launch_time`.
17. In the release callback, derive the missile initial position, velocity, attitude, and body rates directly from the current mothership state.
18. Construct the missile mover and controller from the derived release state.
19. Dynamically wrap the missile in a `Platform` and register it with the engine at release time.
20. Publish `missile_release` when the missile platform is spawned.
21. Publish `missile_drop_start` when the missile enters the drop phase.
22. Implement missile drop-phase logic for `missile_drop_duration` with gravity, drag, and Coriolis active and no powered thrust.
23. Implement the drop-to-ignite handoff strictly at the end of `missile_drop_duration`.
24. Publish `missile_drop_end` and `missile_ignite` when powered missile guidance begins.
25. Implement missile ignite-transition guidance that turns the missile toward `missile_cruise_heading`, drives altitude toward `missile_cruise_altitude`, and adjusts thrust toward `missile_cruise_speed`.
26. Define a simple internal criterion for when missile cruise is established and publish `missile_cruise_established` once that criterion is first met.
27. Implement missile cruise-phase control logic that holds straight flight on the commanded heading while regulating toward `missile_cruise_altitude` and `missile_cruise_speed`.
28. Implement mothership release-to-RTB handoff so the mothership leaves pre-release cruise immediately after weapon release.
29. Publish `mothership_rtb_start` when the mothership begins RTB behavior.
30. Implement mothership RTB guidance toward `mothership_rtb_position_ecef` using a simple direct-to-point guidance law.
31. Define an RTB arrival criterion for the mothership and publish `mothership_rtb_arrival` once the base point is reached.
32. Implement mothership `rtb_hold` behavior so it remains near base and continues logging through `t_end`.
33. Add ground-impact detection for the released missile using current position/altitude.
34. On missile ground impact, publish `missile_ground_impact` and terminate the full scenario.
35. Build the scenario assembly flow:
    - create `SimulationEngine`
    - construct the mothership mover and controller
    - wrap them in a `Platform`
    - register the mothership platform with the engine
    - schedule the missile release event
36. Attach `HDF5Logger(engine, output_path, sample_interval=sample_interval, include_events=True)` and pass the Scenario 2 event topic list explicitly.
37. Ensure the output contains the required datasets for time, full state, position, velocity, LLA, orientation, and body rates for both the mothership and missile.
38. Add a focused unit test for mothership initial heading-to-orientation and heading-to-velocity conversion.
39. Add a focused unit test that verifies both the mothership and missile movers expose the required 13-element state layout.
40. Add a scenario or controller test that verifies the missile is not present before `missile_launch_time` and is dynamically registered at release.
41. Add a controller test that verifies the missile remains in `drop` before `missile_drop_duration` expires and transitions to powered guidance at the correct time.
42. Add a controller or integration test that verifies the missile’s transition/cruise guidance reduces heading and altitude error after ignition.
43. Add a controller or integration test that verifies the mothership switches to RTB behavior after release and reaches the RTB hold state.
44. Add a scenario-level test that verifies missile ground impact terminates the full scenario and emits `missile_ground_impact`.
45. Add a logger test that verifies mixed-platform HDF5 output contains separate trajectory groups for the mothership and missile and includes orientation/body-rate datasets for both.
46. Add a logger test that verifies the Scenario 2 event topics are present in the HDF5 event table.
47. Add an example script that runs Scenario 2 with representative parameters and writes an HDF5 file.
48. Run the relevant test subset and confirm the scenario executes end-to-end with valid output.

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
