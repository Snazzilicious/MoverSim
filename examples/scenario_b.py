import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.core.platform import Platform
from mover_sim.core.mover import NewtonianMover
from mover_sim.core.controller import Controller
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import CSVLogger
from mover_sim.models.aircraft_mover import AircraftMover, AircraftAutopilot
from mover_sim.models.spline_mover import WaypointMover
from mover_sim.math.physics import aerodynamic_drag_force
from mover_sim.math.coordinates import ecef_to_lla, enu_to_ecef

# Define Missile classes for dynamic spawn
class MissileMover(NewtonianMover):
    """
    Newtonian mover for a rocket-propelled missile.
    """
    def __init__(self, initial_position, initial_velocity):
        # Enable gravity and Coriolis
        super().__init__(initial_position, initial_velocity, enable_gravity=True, enable_coriolis=True)
        self.mass = 80.0         # Mass (kg)
        self.thrust = 12000.0     # Thrust force (Newtons) - high thrust
        self.cd = 0.6            # Drag coefficient (higher to limit speed and reduce turning radius)
        self.area = 0.08         # Reference area (m^2)
        self.thrust_dir = np.zeros(3)  # Direction vector of thrust (unit vector)

    def compute_derivatives(self, t, pos, vel):
        dpos, dvel_base = super().compute_derivatives(t, pos, vel)
        
        # Compute aerodynamic drag
        lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
        drag = aerodynamic_drag_force(vel, alt, self.cd, self.area)
        
        # Thrust force in ECEF
        thrust_force = self.thrust * self.thrust_dir
        
        # Net acceleration
        accel = (drag + thrust_force) / self.mass
        
        return dpos, dvel_base + accel


class MissileGuidance(Controller):
    """
    Guidance controller for a missile to intercept a target platform.
    """
    def __init__(self, target_platform, update_interval=0.01):
        super().__init__(update_interval=update_interval)
        self.target = target_platform
        self.hit = False

    def update(self, t, engine):
        mover = self.platform.mover
        pos = mover.position
        tgt_pos = self.target.mover.position
        
        # Compute relative position and distance
        rel_pos = tgt_pos - pos
        dist = np.linalg.norm(rel_pos)
        
        # Check for target intercept
        if dist < 20.0 and not self.hit:
            self.hit = True
            print(f"[{t:6.1f}s] *** INTERCEPT! Missile hit {self.target.id} at a distance of {dist:.2f} meters! ***")
            engine.broker.publish("intercept", self.platform, self.target, dist)
            # Stop simulation
            engine.stop()
            return
            
        # Point thrust directly towards the target (pure pursuit)
        if dist > 1e-3:
            mover.thrust_dir = rel_pos / dist
        else:
            mover.thrust_dir = np.zeros(3)


def run_scenario_b():
    print("=== Running Scenario B: Missile Intercept of Drone ===")
    
    engine = SimulationEngine()
    engine.max_step = 0.05  # High-fidelity step size
    
    # Tactical reference origin (Equator)
    ref_lat, ref_lon, ref_alt = 0.0, 0.0, 0.0
    
    # 1. Setup target (slow drone flying East at 40 m/s at 1500m altitude)
    # Starts at ENU: [-2000.0, 3000.0, 1500.0]
    # Ends at ENU:   [8000.0, 3000.0, 1500.0]
    drone_t = [0.0, 250.0]
    drone_enu_wps = [
        [-2000.0, 3000.0, 1500.0],
        [8000.0, 3000.0, 1500.0]
    ]
    drone_ecef_wps = [enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt) for e, n, u in drone_enu_wps]
    
    drone_mover = WaypointMover(drone_t, drone_ecef_wps)
    drone = Platform("Drone", drone_mover)
    engine.register_platform(drone)
    
    # 2. Setup interceptor aircraft (F18 flying towards the drone path)
    # Starts at ENU: [-2000.0, 0.0, 1200.0]
    f18_pos0 = enu_to_ecef(-2000.0, 0.0, 1200.0, ref_lat, ref_lon, ref_alt)
    
    # F18 flies towards an intercept point in front of the drone: [0.0, 3000.0, 1500.0]
    f18_wp = enu_to_ecef(0.0, 3000.0, 1500.0, ref_lat, ref_lon, ref_alt)
    
    # Calculate initial velocity pointing to the waypoint
    f18_vel0 = (np.array(f18_wp) - np.array(f18_pos0))
    f18_vel0 = (f18_vel0 / np.linalg.norm(f18_vel0)) * 150.0
    
    mover = AircraftMover(f18_pos0, f18_vel0, mass=10000.0, cd0=0.02, t_max=80000.0)
    autopilot = AircraftAutopilot([f18_wp], target_speed=160.0, waypoint_radius=200.0)
    f18 = Platform("F18", mover, autopilot)
    engine.register_platform(f18)
    
    # 3. Setup dynamic missile launch event at t = 1.0 seconds (F18 is pointed towards target line)
    def launch_missile(eng):
        print(f"[{eng.t:6.1f}s] F18 launches intercept missile!")
        
        # Get F18 current state
        f18_pos = f18.mover.position
        f18_vel = f18.mover.velocity
        
        # Spawn missile slightly ahead of F18 with a boost velocity
        f18_dir = f18_vel / np.linalg.norm(f18_vel)
        missile_pos = f18_pos + f18_dir * 10.0  # 10 meters ahead
        missile_vel = f18_vel + f18_dir * 120.0  # 120 m/s relative boost
        
        missile_mover = MissileMover(missile_pos, missile_vel)
        missile_guidance = MissileGuidance(drone)
        missile = Platform("Missile", missile_mover, missile_guidance)
        
        # Dynamically register the spawned missile to the engine
        eng.register_platform(missile)
         
    engine.schedule(1.0, launch_missile, "MissileLaunch")
    
    # Setup CSV logger
    os.makedirs("output", exist_ok=True)
    csv_path = "output/scenario_b_trajectory.csv"
    logger = CSVLogger(engine, csv_path, log_interval=0.05)
    
    print("Starting simulation run...")
    engine.run(30.0)
    print(f"Simulation ended at t = {engine.t:.1f}s. Telemetry written to {csv_path}")

if __name__ == "__main__":
    run_scenario_b()
