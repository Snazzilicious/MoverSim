import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import HDF5Logger
from mover_sim.core.platform import Platform
from mover_sim.math.coordinates import ecef_to_lla, lla_to_ecef
from mover_sim.math.orientation import build_aircraft_body_axes, project_to_rotation_matrix
from mover_sim.models.aircraft_mover import FixedWingAutopilot, FixedWingMover

AirLaunchedCruiseMissileMover = FixedWingMover

class AirLaunchedCruiseMissileController(FixedWingAutopilot):

    DROP_PHASE = "drop"
    CRUISE_PHASE = "cruise"

    def __init__(
        self,
        cruise_speed,
        cruise_altitude,
        cruise_heading,
        drop_duration,
        update_interval=0.1,
    ):
        super().__init__(
            waypoints=[],
            target_speed=cruise_speed,
            update_interval=update_interval,
        )
        self.cruise_speed = float(cruise_speed)
        self.cruise_altitude = float(cruise_altitude)
        self.cruise_heading = float(cruise_heading)
        self.drop_duration = float(drop_duration)
        self.phase = self.DROP_PHASE
        self.t_launch = None
        self._drop_start_published = False
        self._drop_end_published = False

    def _command_drop_behavior(self, mover):
        # Keep the released missile unpowered and passive during the short drop window.
        mover.thrust_cmd = 0.0
        mover.roll_cmd = 0.0
        mover.pitch_cmd = 0.0
        mover.yaw_cmd = 0.0

    def initialize(self, engine):
        self.t_launch = engine.t
        self.phase = self.DROP_PHASE
        super().initialize(engine)
        if not self._drop_start_published:
            engine.broker.publish("missile_drop_start", self.platform)
            self._drop_start_published = True

    def update( self, t, engine ):
        if self.t_launch is None:
            self.t_launch = t

        if self.phase == self.DROP_PHASE and t - self.t_launch < self.drop_duration:
            self._command_drop_behavior(self.platform.mover)
            return

        if self.phase == self.DROP_PHASE:
            self.phase = self.CRUISE_PHASE
            if not self._drop_end_published:
                engine.broker.publish("missile_drop_end", self.platform)
                self._drop_end_published = True

        if self.phase != self.CRUISE_PHASE:
            return
        
        super().update( t, engine )


AirLaunchedCruiseMissileMothershipMover = FixedWingMover
AirLaunchedCruiseMissileMothershipController = FixedWingAutopilot


class MissileLaunchEvent:
    def __init__( self, missile_launch_time ):
        ...

    @property
    def time(self):
        return self.missile_launch_time
    
    @property
    def name(self):
        return "Missile launch"
    
    @property
    def interval(self):
        return None
    
    def __call__( self, engine ):
        engine.broker.publish("missile_drop_end", self.platform)

        # spawn missile with same state as mothership
        engine.register_platform(...)

        # schedule a begin-rtb event 1-2 seconds in the future
        rtb = MothershipRTBEvent( engine.t + 2, mothership_platform )
        engine.schedule( rtb.time, rtb, rtb.name, rtb.interval )

class MothershipRTBEvent:
    def __init__( self, time, mothership_platform ):
        ...

    @property
    def time(self):
        return self._time
    
    @property
    def name(self):
        return "Mothership RTB"
    
    @property
    def interval(self):
        return None
    
    def __call__( self, engine ):
        engine.broker.publish("mothership_rtb_start", self.platform)

        # Add 'home' waypoint to mothership's autopilot and ensure it is tracking it
        # can probably be like 100 km directly behind the mover


def _local_enu_basis(position_ecef):
    position = np.asarray(position_ecef, dtype=float)
    if position.shape != (3,):
        raise ValueError("position_ecef must have shape (3,)")

    lat_deg, lon_deg, _ = ecef_to_lla(position[0], position[1], position[2])
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ])
    up = position / max(np.linalg.norm(position), 1e-6)
    return east, north, up


def _velocity_from_heading_speed(position_ecef, heading, speed, flight_path_angle=0.0):
    heading = float(heading)
    speed = float(speed)
    flight_path_angle = float(flight_path_angle)
    if speed <= 0.0:
        raise ValueError("speed must be greater than 0")

    east, north, up = _local_enu_basis(position_ecef)
    horizontal_direction = np.cos(heading) * north + np.sin(heading) * east
    direction = np.cos(flight_path_angle) * horizontal_direction + np.sin(flight_path_angle) * up
    direction /= max(np.linalg.norm(direction), 1e-6)
    return speed * direction



SCENARIO_EVENT_TOPICS = [
    "platform_registered",
    "missile_release",
    "missile_drop_start",
    "missile_drop_end",
    "mothership_rtb_start",
]


def run_air_launched_cruise_missile_scenario(
    mothership_initial_position_ecef,
    mothership_cruise_speed,
    mothership_cruise_altitude,
    mothership_cruise_heading,
    mothership_rtb_position_ecef,
    missile_launch_time,
    missile_cruise_speed,
    missile_cruise_altitude,
    missile_cruise_heading,
    missile_drop_duration,
    t_end,
    sample_interval,
    output_group,
):
    """Run an air-launched cruise-missile mission and save HDF5 telemetry.

    This is the public scenario entry point. Its arguments are mission-level inputs,
    while mover-, controller-, and event-specific implementation details remain internal.

    Args:
        mothership_initial_position_ecef: Initial mothership ECEF position vector in meters.
        mothership_cruise_speed: Commanded mothership cruise speed in meters/second.
        mothership_cruise_altitude: Commanded mothership cruise altitude above the WGS-84 ellipsoid in meters.
        mothership_cruise_heading: Commanded mothership local-ENU azimuth heading in radians.
        mothership_rtb_position_ecef: Return-to-base destination in ECEF meters.
        missile_launch_time: Scenario time at which the missile is released in seconds.
        missile_cruise_speed: Commanded missile cruise speed in meters/second.
        missile_cruise_altitude: Commanded missile cruise altitude above the WGS-84 ellipsoid in meters.
        missile_cruise_heading: Commanded missile local-ENU azimuth heading in radians.
        missile_drop_duration: Duration of the unpowered drop phase in seconds.
        t_end: Maximum scenario run time in seconds.
        sample_interval: HDF5 logging sample interval in seconds.
        output_group: Caller-created `h5py.Group` used as the root for this scenario run.

    Returns:
        A dictionary containing the simulation engine, mothership platform, logger, and output group.
    """

    if missile_launch_time > t_end:
        raise ValueError("missile_launch_time must be less than or equal to t_end")

    mothership_initial_velocity = _velocity_from_heading_speed(
        mothership_initial_position_ecef,
        mothership_cruise_heading,
        mothership_cruise_speed,
    )
    mothership_initial_body_rates = np.zeros(3)

    engine = SimulationEngine()
    mothership_mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=mothership_initial_position_ecef,
        initial_velocity=mothership_initial_velocity,
        initial_body_rates=mothership_initial_body_rates,
    )
    mothership_controller = AirLaunchedCruiseMissileMothershipController(
        cruise_speed=mothership_cruise_speed,
        cruise_altitude=mothership_cruise_altitude,
        cruise_heading=mothership_cruise_heading,
        rtb_position_ecef=mothership_rtb_position_ecef
    )
    mothership_platform = Platform("mothership", mothership_mover, mothership_controller)
    engine.register_platform(mothership_platform)

    missile_launch = MissileLaunchEvent( missile_launch_time )
    engine.schedule( missile_launch.time, missile_launch, missile_launch.name, missile_launch.interval )

    logger = HDF5Logger(
        engine,
        output_group,
        sample_interval=sample_interval,
        include_state=True,
        include_lla=True,
        include_events=True,
        event_topics=SCENARIO_EVENT_TOPICS,
    )

    engine.run(t_end)

    return {
        "engine": engine,
        "mothership_platform": mothership_platform,
        "logger": logger,
        "output_group": output_group,
    }


def run_example():
    print("=== Running Scenario: Air-Launched Cruise Missile ===")

    os.makedirs("output", exist_ok=True)
    output_path = Path("output/scenario_air_launched_cruise_missile.h5")

    with h5py.File(output_path, "w") as h5:
        result = run_air_launched_cruise_missile_scenario(
            mothership_initial_position_ecef=lla_to_ecef(37.6193, -122.3750, 1500.0),
            mothership_cruise_speed=200.0,
            mothership_cruise_altitude=1500.0,
            mothership_cruise_heading=0.0,
            mothership_rtb_position_ecef=lla_to_ecef(37.6193, -122.3650, 1500.0),
            missile_launch_time=2.0,
            missile_cruise_speed=250.0,
            missile_cruise_altitude=1200.0,
            missile_cruise_heading=0.0,
            missile_drop_duration=0.5,
            t_end=20.0,
            sample_interval=0.1,
            output_group=h5.create_group("air_run_001"),
        )

    print(f"Simulation ended at t = {result['engine'].t:.2f}s. Telemetry written to {output_path}")


if __name__ == "__main__":
    run_example()
