import csv
from mover_sim.math.coordinates import ecef_to_lla

class CSVLogger:
    """
    Observer that writes the trajectories of all platforms in the simulation 
    to a CSV file.
    """
    def __init__(self, engine, filepath, log_interval=1.0):
        """
        Parameters:
            engine: The SimulationEngine instance.
            filepath: Path to the output CSV file.
            log_interval: Minimum time interval (seconds) between logs.
        """
        self.engine = engine
        self.filepath = filepath
        self.log_interval = log_interval
        self.last_log_time = -float('inf')
        self.file = None
        self.writer = None
        
        # Subscribe to simulation events
        self.engine.broker.subscribe("sim_start", self.on_sim_start)
        self.engine.broker.subscribe("position_updated", self.on_position_updated)
        self.engine.broker.subscribe("sim_end", self.on_sim_end)

    def on_sim_start(self, t):
        """
        Initialize the CSV file and write the header.
        """
        self.file = open(self.filepath, mode='w', newline='')
        self.writer = csv.writer(self.file)
        
        # Build header row
        header = ["time"]
        for plat_id in sorted(self.engine.platforms.keys()):
            header.extend([
                f"{plat_id}_x", f"{plat_id}_y", f"{plat_id}_z",
                f"{plat_id}_lat", f"{plat_id}_lon", f"{plat_id}_alt",
                f"{plat_id}_vx", f"{plat_id}_vy", f"{plat_id}_vz"
            ])
        self.writer.writerow(header)
        
        # Log initial state
        self.log_state(t)

    def on_position_updated(self, t):
        """
        Log current states if the log interval has elapsed.
        """
        if t - self.last_log_time >= self.log_interval - 1e-9:
            self.log_state(t)

    def log_state(self, t):
        """
        Write the current coordinates and velocities of all platforms to the file.
        """
        if not self.writer:
            return
            
        row = [t]
        for plat_id in sorted(self.engine.platforms.keys()):
            plat = self.engine.platforms[plat_id]
            pos = plat.mover.position
            vel = plat.mover.velocity
            lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
            row.extend([
                pos[0], pos[1], pos[2],
                lat, lon, alt,
                vel[0], vel[1], vel[2]
            ])
        self.writer.writerow(row)
        self.last_log_time = t

    def on_sim_end(self, t):
        """
        Ensure final state is logged and close the file.
        """
        if t > self.last_log_time:
            self.log_state(t)
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
