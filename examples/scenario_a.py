import os
import numpy as np
from mover_sim.core.platform import Platform
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import CSVLogger
from mover_sim.models.aircraft_mover import AircraftMover, AircraftAutopilot
from mover_sim.math.coordinates import lla_to_ecef, enu_to_ecef

def run_scenario_a():
    print("=== Running Scenario A: Waypoint Flight Pattern around SFO ===")
    
    # 1. Initialize simulation engine
    engine = SimulationEngine()
    engine.max_step = 0.5  # Integrate in half-second steps
    
    # SFO Airport tangent plane origin
    sfo_lat, sfo_lon, sfo_alt = 37.6193, -122.3750, 4.0
    
    # Convert ENU waypoints to global ECEF
    # Creating a rectangular airfield pattern around SFO
    enu_waypoints = [
        [0.0, 5000.0, 1000.0],       # 5km North, 1000m alt (Takeoff / Climb)
        [5000.0, 5000.0, 1500.0],    # Crosswind turn: 5km East, 5km North, 1500m alt
        [5000.0, -5000.0, 1500.0],   # Downwind leg: 5km East, 5km South, 1500m alt
        [-2000.0, -5000.0, 1000.0],  # Base leg: 2km West, 5km South, 1000m alt
        [0.0, 0.0, 10.0]             # Final approach: Back to SFO center
    ]
    
    ecef_waypoints = [
        enu_to_ecef(e, n, u, sfo_lat, sfo_lon, sfo_alt)
        for e, n, u in enu_waypoints
    ]
    
    # 2. Define aircraft and initial state (starting on runway heading North at 120 m/s)
    # Velocity directed North: in local ENU is [0.0, 120.0, 0.0]
    pos0 = enu_to_ecef(0.0, 0.0, 10.0, sfo_lat, sfo_lon, sfo_alt)
    
    # Calculate ECEF velocity vector
    pos0_ref = enu_to_ecef(0.0, 120.0, 10.0, sfo_lat, sfo_lon, sfo_alt)
    vel0 = np.array(pos0_ref) - np.array(pos0)
    # normalize to exactly 120 m/s
    vel0 = (vel0 / np.linalg.norm(vel0)) * 120.0
    
    mover = AircraftMover(pos0, vel0, mass=12000.0, cd0=0.02, t_max=100000.0)
    autopilot = AircraftAutopilot(ecef_waypoints, target_speed=150.0, waypoint_radius=400.0, update_interval=0.1)
    aircraft = Platform("Airliner", mover, autopilot)
    engine.register_platform(aircraft)
    
    # 3. Setup event listeners and CSV logger
    def on_waypoint(platform, idx):
        print(f"[{engine.t:6.1f}s] {platform.id} reached Waypoint {idx} -> Heading to WP {idx+1}")
        
    engine.broker.subscribe("waypoint_reached", on_waypoint)
    
    os.makedirs("output", exist_ok=True)
    csv_path = "output/scenario_a_trajectory.csv"
    logger = CSVLogger(engine, csv_path, log_interval=1.0)
    
    # 4. Run simulation for 250 seconds
    print("Starting simulation run...")
    engine.run(250.0)
    print(f"Simulation ended at t = {engine.t:.1f}s. Telemetry written to {csv_path}")

if __name__ == "__main__":
    run_scenario_a()
