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
        self.k_cruise_yaw = 100.0 / np.radians(30.0)
        self._drop_start_published = False
        self._drop_end_published = False

    def _command_drop_behavior(self, mover):
        # Keep the released missile unpowered and passive during the short drop window.
        mover.thrust_cmd = 0.0
        mover.roll_cmd = 0.0
        mover.pitch_cmd = 0.0
        mover.yaw_cmd = 0.0

    def _desired_cruise_horizontal_direction(self, mover):
        # `cruise_heading` is defined in the missile's local ENU frame, so resolve the
        # desired world-frame direction from the current position rather than storing one
        # fixed ECEF direction at cruise entry.
        east, north, _ = _local_enu_basis(mover.position)
        desired_horizontal = (
            np.cos(self.cruise_heading) * north
            + np.sin(self.cruise_heading) * east
        )
        desired_norm = np.linalg.norm(desired_horizontal)
        if desired_norm <= 1e-6:
            return north
        return desired_horizontal / desired_norm

    def _enter_hold_mode(self, mover):
        # Reuse the base class empty-route hold path for cruise, but seed it with the
        # missile's commanded mission targets instead of its post-drop state.
        self.hold_active = True
        self.hold_speed = self.cruise_speed
        self.hold_altitude = self.cruise_altitude
        self.hold_horizontal_direction = self._desired_cruise_horizontal_direction(mover)

    def _update_hold_mode(self, mover):
        # The inherited hold update freezes one world-frame heading vector. Recompute the
        # target from the current local ENU basis so the missile continues to track the
        # commanded local heading as it moves over the Earth.
        _, vel, speed, local_up, current_altitude = self._flight_condition(mover)
        desired_horizontal = self._desired_cruise_horizontal_direction(mover)
        self.hold_horizontal_direction = desired_horizontal
        heading_error = self._heading_error_to_direction(
            mover,
            desired_horizontal,
            pos=mover.position,
            vel=vel,
            local_up=local_up,
        )
        vertical_error = self.cruise_altitude - current_altitude
        self._apply_guidance(
            mover,
            heading_error,
            vertical_error,
            self.cruise_speed,
            vel=vel,
            speed=speed,
            local_up=local_up,
            alt=current_altitude,
        )
        # Match the lateral-control convention used by the released-missile dynamics and
        # add direct yaw assistance so cruise heading convergence is stronger than the
        # generic bank-only empty-route hold logic.
        mover.roll_cmd *= -1.0
        mover.yaw_cmd = np.clip(self.k_cruise_yaw * heading_error, -100.0, 100.0)

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
            self.hold_active = False
            if not self._drop_end_published:
                engine.broker.publish("missile_drop_end", self.platform)
                self._drop_end_published = True

        if self.phase != self.CRUISE_PHASE:
            return
        
        super().update( t, engine )


AirLaunchedCruiseMissileMothershipMover = FixedWingMover


class AirLaunchedCruiseMissileMothershipController(FixedWingAutopilot):

    CRUISE_MODE = "cruise"
    RTB_MODE = "rtb"

    def __init__(
        self,
        cruise_speed,
        cruise_altitude,
        cruise_heading,
        rtb_position_ecef=None,
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
        self.rtb_position_ecef = self._coerce_optional_vector3(rtb_position_ecef, "rtb_position_ecef")
        self.mode = self.CRUISE_MODE

    def _coerce_optional_vector3(self, value, name):
        if value is None:
            return None
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        return vector

    def _desired_cruise_horizontal_direction(self, mover):
        east, north, _ = _local_enu_basis(mover.position)
        desired_horizontal = (
            np.cos(self.cruise_heading) * north
            + np.sin(self.cruise_heading) * east
        )
        desired_norm = np.linalg.norm(desired_horizontal)
        if desired_norm <= 1e-6:
            return north
        return desired_horizontal / desired_norm

    def _desired_rtb_horizontal_direction(self, mover):
        if self.rtb_position_ecef is None:
            return self._desired_cruise_horizontal_direction(mover)

        # Steer toward the RTB destination using the current local horizontal plane so
        # guidance remains well-defined even as the mothership moves over the Earth.
        local_up = mover.position / max(np.linalg.norm(mover.position), 1e-6)
        rel = self.rtb_position_ecef - mover.position
        rel_horizontal = rel - np.dot(rel, local_up) * local_up
        rel_horizontal_norm = np.linalg.norm(rel_horizontal)
        if rel_horizontal_norm <= 1e-6:
            return self._desired_cruise_horizontal_direction(mover)
        return rel_horizontal / rel_horizontal_norm

    def _target_altitude(self):
        # During RTB, use the destination altitude so the mothership converges to the
        # return point rather than holding its original cruise altitude indefinitely.
        if self.mode == self.RTB_MODE and self.rtb_position_ecef is not None:
            return ecef_to_lla(
                self.rtb_position_ecef[0],
                self.rtb_position_ecef[1],
                self.rtb_position_ecef[2],
            )[2]
        return self.cruise_altitude

    def enter_rtb(self, rtb_position_ecef=None):
        if rtb_position_ecef is not None:
            self.rtb_position_ecef = self._coerce_optional_vector3(rtb_position_ecef, "rtb_position_ecef")
        self.mode = self.RTB_MODE
        # Clear any cached hold targets from cruise mode so the next update rebuilds the
        # hold state entirely from the RTB destination.
        self.hold_speed = None
        self.hold_altitude = None
        self.hold_horizontal_direction = None
        self.hold_active = False

    def _enter_hold_mode(self, mover):
        # Reuse the base class empty-route hold path, but seed it from the active mission
        # mode so cruise and RTB both flow through the same update mechanism.
        self.hold_active = True
        self.hold_speed = self.cruise_speed
        self.hold_altitude = self._target_altitude()
        if self.mode == self.RTB_MODE:
            self.hold_horizontal_direction = self._desired_rtb_horizontal_direction(mover)
        else:
            self.hold_horizontal_direction = self._desired_cruise_horizontal_direction(mover)

    def _update_hold_mode(self, mover):
        _, vel, speed, local_up, current_altitude = self._flight_condition(mover)
        if self.mode == self.RTB_MODE:
            desired_horizontal = self._desired_rtb_horizontal_direction(mover)
        else:
            desired_horizontal = self._desired_cruise_horizontal_direction(mover)

        self.hold_horizontal_direction = desired_horizontal
        heading_error = self._heading_error_to_direction(
            mover,
            desired_horizontal,
            pos=mover.position,
            vel=vel,
            local_up=local_up,
        )
        vertical_error = self._target_altitude() - current_altitude
        self._apply_guidance(
            mover,
            heading_error,
            vertical_error,
            self.cruise_speed,
            vel=vel,
            speed=speed,
            local_up=local_up,
            alt=current_altitude,
        )


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
