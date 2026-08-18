# MoverSim Development Plan

This plan breaks down the construction of the Mover Simulation package into sequential, testable milestones. Each milestone culminates in tests to ensure correctness before moving forward.

---

## Phase 1: Project Setup & Coordinate Math (Milestone 1)
*Goal: Establish directory structure, tooling, and correct coordinate transformation functions.*

1.  **Initialize Project Structure**:
    *   Set up the Python package structure under `/home/ian/MoverSim`.
    *   Create base files: `setup.py` / `pyproject.toml`, `.gitignore`, and `requirements.txt`.
2.  **Coordinate Transformations Module (`mover_sim/math/coordinates.py`)**:
    *   Implement conversions between Geodetic (LLA) $\leftrightarrow$ Earth-Centered Earth-Fixed (ECEF) using the WGS-84 ellipsoid parameters.
    *   Implement local coordinate frames: ECEF $\leftrightarrow$ East-North-Up (ENU) given a tangent point origin.
3.  **Unit Tests (`tests/test_coordinates.py`)**:
    *   Verify coordinates round-trip correctly (LLA $\to$ ECEF $\to$ LLA).
    *   Verify ECEF $\leftrightarrow$ ENU against known test vectors.

---

## Phase 2: Event System & Simulation Engine Skeleton (Milestone 2)
*Goal: Build the event dispatch system and the core simulation step loop.*

1.  **Event Broker (`mover_sim/core/broker.py`)**:
    *   Implement the Publish-Subscribe pattern.
    *   Allow objects to subscribe to custom topics (e.g. state updates).
2.  **Event Scheduler (`mover_sim/core/engine.py`)**:
    *   Create a priority queue based event scheduler for one-time and recurring events.
3.  **Simulation Engine Core (`mover_sim/core/engine.py` - `SimulationEngine`)**:
    *   Implement basic simulation time stepping (`t` progression).
    *   Integrate event checking: advance time to the next scheduled event, pause, execute event, then continue.
4.  **Unit Tests (`tests/test_engine.py`)**:
    *   Test basic discrete event scheduling and execution timing.
    *   Test publisher-subscriber event receipt.

---

## Phase 3: Platform and Mover Interfaces (Milestone 3)
*Goal: Introduce platform entities, analytical movers, and the continuous ODE integration system.*

1.  **Entity Classes (`mover_sim/core/platform.py`, `mover_sim/core/mover.py`)**:
    *   Define `Platform`, `Mover`, and `Controller` base classes.
2.  **Analytical & Spline Movers (`mover_sim/models/spline_mover.py`)**:
    *   Implement `AnalyticalMover` (time-based positions).
    *   Implement a `WaypointMover` that follows a set of waypoints by interpolating time/distance.
3.  **Newtonian Movers & ODE Solver Integration**:
    *   Implement `NewtonianMover` that registers state vectors (position and velocity) with the `SimulationEngine`.
    *   Update `SimulationEngine` to assemble the ODE state vector dynamically, call SciPy's `RK45` to integrate continuous motion, and trigger solver resets when events alter dynamics.
4.  **Unit Tests (`tests/test_movers.py`)**:
    *   Verify `WaypointMover` visits waypoints on schedule.
    *   Test simple constant-velocity Newtonian integration using `RK45`.

---

## Phase 4: Physics & Environment Models (Milestone 4)
*Goal: Model real-world physical forces acting on Newtonian platforms.*

1.  **Physics Module (`mover_sim/math/physics.py`)**:
    *   Implement global gravity vector computation (WGS-84 gravity model J2 or spherical gravity).
    *   Implement Coriolis force acceleration based on Earth's angular velocity in ECEF.
    *   Implement aerodynamic drag (wind, altitude-dependent air density, drag coefficient, and reference area).
2.  **Integrate Forces into Newtonian Movers**:
    *   Extend `NewtonianMover` to compute net forces acting on the platform.
3.  **Unit Tests**:
    *   Verify gravity and Coriolis force calculations match expected analytical values at different coordinates and velocities.

---

## Phase 5: Controllers & Templates (Milestone 5)
*Goal: Add complex aircraft behaviors, guidance controllers, and observers.*

1.  **Logical Controller Base & Implementations (`mover_sim/core/controller.py`)**:
    *   Implement controller scheduling hook (e.g. regular updates at a fixed interval like 50Hz).
    *   Create a simple waypoint navigation autopilot/guidance controller.
2.  **Aircraft Mover Template (`mover_sim/models/aircraft_mover.py`)**:
    *   Implement a 6-DOF aircraft mover model resolving thrust, lift, drag, gravity, and control surface effects.
3.  **Telemetry Observers**:
    *   Implement a file logger observer that writes platform trajectories to CSV files.

---

## Phase 6: End-to-End Simulation & Verification (Milestone 6)
*Goal: Run full-scenario simulations, visualize trajectories, and finalize the package.*

1.  **Create Integration Scenarios**:
    *   **Scenario A**: Aircraft flying a waypoint pattern around an airport using ENU coordinates.
    *   **Scenario B**: Dynamic spawn event (e.g., aircraft spawning and launching a tracking missile).
2.  **Visualization & Diagnostics**:
    *   Write scripts to plot 3D paths on a globe or relative to an origin using Matplotlib.
3.  **Final Cleanup**:
    *   Add docstrings, package details, and final documentation.