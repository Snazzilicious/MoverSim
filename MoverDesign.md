# Mover Simulation Design Specification

This document details the high-level design, class structure, and data flows for the general-purpose moving platforms simulator.

---

## 1. System Architecture

The simulation uses a hybrid **event-driven and step-based** architecture, managing both continuous physics state integration (via SciPy's `RK45` solver) and discrete system events.

```mermaid
graph TD
    Engine[Simulation Engine] --> Scheduler[Event Scheduler]
    Engine --> Solver[Central ODE Solver]
    Engine --> Broker[Event Broker]
    
    Platform[Platform] --> Mover[Mover Component]
    Platform --> Controller[Logical Controller]
    
    Mover --> NewtonianMover[NewtonianMover]
    Mover --> AnalyticalMover[AnalyticalMover]
    
    NewtonianMover -.-> Solver
    Controller -.-> Scheduler
    Broker -.-> Observer[Observers/Loggers]
```

---

## 2. Core Modules and Class Structure

### 2.1 Core Package (`mover_sim.core`)

#### `SimulationEngine`
Coordinates the simulation loop, advances time, manages the ODE solver, and handles events.
*   **Key Responsibilities**:
    *   Maintain current simulation time $t$.
    *   Dynamically assemble state vector $Y$ and derivative function $F(t, Y)$ from active `NewtonianMover` instances.
    *   Integrate continuous state using `scipy.integrate.RK45` up to the next scheduled event time.
    *   Execute events at their scheduled times and handle state/solver resets when force or state updates are non-smooth.
    *   Dispatch simulation lifecycle events to the central `EventBroker`.

#### `EventScheduler`
Maintains a priority queue of scheduled events sorted by time.
*   **Key Responsibilities**:
    *   Insert, retrieve, and cancel events.
    *   Support both user-defined events (one-time or recurring) and internal solver boundary events.

#### `EventBroker`
A Publish-Subscribe broker that decouples platforms, solvers, and observers.
*   **Key Responsibilities**:
    *   Allow observers to subscribe to specific topics (e.g., `"position_update"`, `"collision"`, `"spawn"`).
    *   Provide a `publish(topic, data)` method for sending telemetry and event payloads.

#### `Platform`
Represent a physical entity in the simulation.
*   **Attributes**:
    *   `id`: Unique identifier string.
    *   `mover`: Reference to a `Mover` object.
    *   `controller`: Reference to an optional `Controller` object.
    *   `properties`: Key-value store for physical characteristics (e.g., mass, wing area).

#### `Mover` (Base Class)
Abstract base class defining how a platform updates its position/state.
*   **Subclasses**:
    *   `AnalyticalMover`: Bypasses the ODE solver, calculating state analytically: $x(t) = f(t)$.
    *   `NewtonianMover`: Registers a slice of the global ODE state vector and provides derivatives $\dot{y} = f(t, y, \text{forces})$.

#### `Controller` (Base Class)
Base class for logical control loops (guidance, path following, autopilots).
*   **Key Responsibilities**:
    *   Registers recurring update events at a target frequency (e.g., 50 Hz).
    *   During execution, reads platform state, computes guidance commands (e.g., target thrust, steering angle), and applies them as forces or control variables to the `Mover`.

---

### 2.2 Math & Physics Layer (`mover_sim.math`)

#### `CoordinateTransforms`
Utility class for conversions using the WGS-84 ellipsoidal model:
*   `ecef_to_lla(x, y, z) -> (lat, lon, alt)`
*   `lla_to_ecef(lat, lon, alt) -> (x, y, z)`
*   `ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref) -> (e, n, u)`
*   `enu_to_ecef(e, n, u, lat_ref, lon_ref, alt_ref) -> (x, y, z)`

#### `GlobalPhysics`
Calculates environmental accelerations/forces in ECEF/ECI coordinates:
*   `gravity(ecef_pos) -> gravity_accel`
*   `coriolis(velocity) -> coriolis_accel`
*   `aerodynamic_drag(velocity, air_density, cd, area) -> drag_force`

---

## 3. Key Workflows & Execution Sequences

### 3.1 Main Simulation Step

```mermaid
sequenceDiagram
    participant Engine as SimulationEngine
    participant Scheduler as EventScheduler
    participant Solver as RK45 Solver
    participant Broker as EventBroker

    rect rgb(240, 240, 240)
    Note over Engine, Broker: Simulation Step Loop
    Engine->>Scheduler: get_next_event_time()
    Scheduler-->>Engine: t_next_event
    
    alt t + max_step < t_next_event
        Note over Engine: Integrate to t + max_step
    else
        Note over Engine: Integrate to t_next_event
    end

    Engine->>Solver: step_to(t_target)
    Solver-->>Engine: state_updated
    Engine->>Broker: publish("position_updated", states)
    
    alt reached t_next_event
        Engine->>Scheduler: pop_next_event()
        Scheduler-->>Engine: event
        Engine->>Engine: execute(event)
        Note over Engine: If event modified force/state, reset ODE Solver state
    end
    end
```

### 3.2 Event-Scheduled Controller Execution

```mermaid
sequenceDiagram
    participant Engine as SimulationEngine
    participant Scheduler as EventScheduler
    participant Controller as LogicalController
    participant Mover as NewtonianMover

    Engine->>Scheduler: execute(ControllerEvent)
    Scheduler->>Controller: update(t)
    Controller->>Mover: set_control_inputs(thrust, steering)
    Note over Mover: Input changed (non-smooth derivative)
    Mover->>Engine: flag_solver_reset()
    Controller->>Scheduler: schedule_next(t + dt)
```

---

## 4. Proposed Package Structure

```text
mover_sim/
│
├── mover_sim/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py          # SimulationEngine & EventScheduler
│   │   ├── broker.py          # EventBroker (Pub/Sub)
│   │   ├── platform.py        # Platform base class
│   │   ├── mover.py           # Mover (Analytical, Newtonian)
│   │   └── controller.py      # Logical controller base
│   │
│   ├── math/
│   │   ├── __init__.py
│   │   ├── coordinates.py     # Coordinate transforms (WGS-84, ENU, ECEF)
│   │   └── physics.py         # Global force models (Gravity, Coriolis, Drag)
│   │
│   └── models/
│       ├── __init__.py
│       ├── spline_mover.py    # Waypoint & spline followers
│       └── aircraft_mover.py  # Aircraft model (forces: thrust, lift, drag)
│
└── tests/                     # Unit test suite
    ├── __init__.py
    ├── test_coordinates.py
    ├── test_engine.py
    └── test_movers.py
```