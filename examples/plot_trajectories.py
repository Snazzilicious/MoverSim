import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib is not installed. Plotting will be skipped. Run 'pip install matplotlib' to enable.")

from mover_sim.math.coordinates import ecef_to_enu

def plot_scenario_a():
    csv_path = "output/scenario_a_trajectory.csv"
    if not os.path.exists(csv_path):
        print(f"Scenario A output file {csv_path} not found. Run scenario_a.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # SFO Airport reference coordinate
    ref_lat, ref_lon, ref_alt = 37.6193, -122.3750, 4.0
    
    # Extract airliner coordinates and convert to ENU
    e_coords, n_coords, u_coords = [], [], []
    for _, row in df.iterrows():
        e, n, u = ecef_to_enu(
            row["Airliner_x"], row["Airliner_y"], row["Airliner_z"],
            ref_lat, ref_lon, ref_alt
        )
        e_coords.append(e)
        n_coords.append(n)
        u_coords.append(u)
        
    df["East"] = e_coords
    df["North"] = n_coords
    df["Up"] = u_coords
    
    if MATPLOTLIB_AVAILABLE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 2D Flight Path (ENU)
        ax1.plot(df["East"] / 1000.0, df["North"] / 1000.0, 'b-', label="Airliner Trajectory")
        
        # Plot waypoints
        enu_waypoints = [
            [0.0, 5.0],    # WP0: 5km North
            [5.0, 5.0],    # WP1: 5km East, 5km North
            [5.0, -5.0],   # WP2: 5km East, 5km South
            [-2.0, -5.0],  # WP3: 2km West, 5km South
            [0.0, 0.0]     # WP4: SFO Center
        ]
        wp_e, wp_n = zip(*enu_waypoints)
        ax1.scatter(wp_e, wp_n, color='red', marker='X', s=100, label="Waypoints")
        for i, (we, wn) in enumerate(enu_waypoints):
            ax1.text(we + 0.1, wn + 0.1, f"WP{i}", color='darkred', fontsize=10)
            
        ax1.set_title("Scenario A: Flight Path around SFO Airport")
        ax1.set_xlabel("East offset (km)")
        ax1.set_ylabel("North offset (km)")
        ax1.grid(True)
        ax1.legend()
        ax1.axis("equal")
        
        # 2. Altitude Profile
        ax2.plot(df["time"], df["Up"], 'g-', label="Airliner Altitude")
        ax2.set_title("Scenario A: Altitude Profile")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Altitude above SFO (m)")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        output_plot = "output/scenario_a_flight_profile.png"
        plt.savefig(output_plot)
        print(f"Saved flight profile plot to {output_plot}")
        plt.close()

def plot_scenario_b():
    csv_path = "output/scenario_b_trajectory.csv"
    if not os.path.exists(csv_path):
        print(f"Scenario B output file {csv_path} not found. Run scenario_b.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Tactial reference origin (Equator)
    ref_lat, ref_lon, ref_alt = 0.0, 0.0, 0.0
    
    # Convert drone coordinates to ENU
    drone_e, drone_n = [], []
    for _, row in df.iterrows():
        e, n, u = ecef_to_enu(
            row["Drone_x"], row["Drone_y"], row["Drone_z"],
            ref_lat, ref_lon, ref_alt
        )
        drone_e.append(e)
        drone_n.append(n)
        
    # Convert F18 coordinates to ENU
    f18_e, f18_n = [], []
    for _, row in df.iterrows():
        e, n, u = ecef_to_enu(
            row["F18_x"], row["F18_y"], row["F18_z"],
            ref_lat, ref_lon, ref_alt
        )
        f18_e.append(e)
        f18_n.append(n)
        
    # Convert Missile coordinates to ENU (handling NaN before it is spawned)
    missile_e, missile_n = [], []
    for _, row in df.iterrows():
        # Check if Missile exists in columns and is not NaN
        if "Missile_x" in row and not np.isnan(row["Missile_x"]):
            e, n, u = ecef_to_enu(
                row["Missile_x"], row["Missile_y"], row["Missile_z"],
                ref_lat, ref_lon, ref_alt
            )
            missile_e.append(e)
            missile_n.append(n)
        else:
            missile_e.append(np.nan)
            missile_n.append(np.nan)
            
    df["Drone_E"] = drone_e
    df["Drone_N"] = drone_n
    df["F18_E"] = f18_e
    df["F18_N"] = f18_n
    df["Missile_E"] = missile_e
    df["Missile_N"] = missile_n
    
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 8))
        
        plt.plot(df["Drone_E"] / 1000.0, df["Drone_N"] / 1000.0, 'g-', label="Target Drone (30 m/s)")
        plt.plot(df["F18_E"] / 1000.0, df["F18_N"] / 1000.0, 'b--', label="F18 Interceptor")
        
        # Filter out NaN for missile to plot
        missile_valid = df.dropna(subset=["Missile_E"])
        if not missile_valid.empty:
            plt.plot(missile_valid["Missile_E"] / 1000.0, missile_valid["Missile_N"] / 1000.0, 'r-', linewidth=2, label="Guided Missile")
            # Mark launch point
            plt.scatter(missile_valid["Missile_E"].iloc[0] / 1000.0, missile_valid["Missile_N"].iloc[0] / 1000.0, color='orange', marker='o', s=120, label="Launch Event (t=1s)", zorder=5)
            # Mark intercept point
            plt.scatter(missile_valid["Missile_E"].iloc[-1] / 1000.0, missile_valid["Missile_N"].iloc[-1] / 1000.0, color='red', marker='*', s=250, label="Intercept Event", zorder=6)
            
        plt.title("Scenario B: Drone Intercept Trajectories")
        plt.xlabel("East offset (km)")
        plt.ylabel("North offset (km)")
        plt.grid(True)
        plt.legend()
        plt.axis("equal")
        
        output_plot = "output/scenario_b_intercept_paths.png"
        plt.savefig(output_plot)
        print(f"Saved intercept path plot to {output_plot}")
        plt.close()

if __name__ == "__main__":
    plot_scenario_a()
    plot_scenario_b()
