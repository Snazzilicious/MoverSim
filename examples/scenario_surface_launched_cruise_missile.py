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


class SurfaceLaunchedCruiseMissileMover(FixedWingMover):

    def __init__(
        self,
        initial_position,
        initial_velocity,
        initial_orientation=None,
        initial_body_rates=None,
        mass=10000.0,
        rotational_mass=None,
        area=30.0,
        cd0=0.02,
        cd_alpha=0.3,
        cd_beta=0.3,
        cl_alpha=4.5,
        cy_beta=-1.5,
        bank_restoring_coeff=2.0e4,
        alpha_restoring_coeff=3.0e4,
        beta_restoring_coeff=2.0e4,
        roll_damping_coeff=1.5e4,
        pitch_damping_coeff=2.0e4,
        yaw_damping_coeff=1.5e4,
        max_thrust=80000.0,
        max_roll_moment=5.0e4,
        max_pitch_moment=5.0e4,
        max_yaw_moment=2.0e4,
        use_coriolis=True,
        boost_thrust=0.0,
    ):
        super().__init__(
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            initial_orientation=initial_orientation,
            initial_body_rates=initial_body_rates,
            mass=mass,
            rotational_mass=rotational_mass,
            area=area,
            cd0=cd0,
            cd_alpha=cd_alpha,
            cd_beta=cd_beta,
            cl_alpha=cl_alpha,
            cy_beta=cy_beta,
            bank_restoring_coeff=bank_restoring_coeff,
            alpha_restoring_coeff=alpha_restoring_coeff,
            beta_restoring_coeff=beta_restoring_coeff,
            roll_damping_coeff=roll_damping_coeff,
            pitch_damping_coeff=pitch_damping_coeff,
            yaw_damping_coeff=yaw_damping_coeff,
            max_thrust=max_thrust,
            max_roll_moment=max_roll_moment,
            max_pitch_moment=max_pitch_moment,
            max_yaw_moment=max_yaw_moment,
            use_coriolis=use_coriolis,
        )
        self.boost_thrust = float(boost_thrust)
        self.boost_active = True
    
    def _thrust_vector( self, forward ):
        if self.boost_active:
            return self.boost_thrust * forward
        else:
            return super()._thrust_vector( forward )

class SurfaceLaunchedCruiseMissileController(FixedWingAutopilot):

    BOOST_PHASE = "boost"
    CRUISE_PHASE = "cruise"

    def __init__(
        self,
        cruise_speed,
        cruise_altitude,
        cruise_heading,
        boost_duration,
        launch_pitch_angle,
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
        self.boost_duration = float(boost_duration)
        self.launch_pitch_angle = float(launch_pitch_angle)
        self.phase = self.BOOST_PHASE
        self.t_launch = None
        self.k_boost_pitch = 100.0 / np.radians(20.0)
        self.k_cruise_yaw = 100.0 / np.radians(30.0)
        self._boost_start_published = False
        self._boost_end_published = False

    def _local_horizontal_basis(self, position):
        local_up = position / max(np.linalg.norm(position), 1e-6)
        earth_z = np.array([0.0, 0.0, 1.0])
        east = np.cross(earth_z, local_up)
        east_norm = np.linalg.norm(east)
        if east_norm <= 1e-6:
            east = np.array([0.0, 1.0, 0.0])
        else:
            east = east / east_norm
        north = np.cross(local_up, east)
        north = north / max(np.linalg.norm(north), 1e-6)
        return east, north, local_up

    def _command_boost_attitude(self, mover):
        east, north, local_up = self._local_horizontal_basis(mover.position)
        desired_horizontal = (
            np.cos(self.cruise_heading) * north
            + np.sin(self.cruise_heading) * east
        )
        heading_error = self._heading_error_to_direction(
            mover,
            desired_horizontal,
            pos=mover.position,
            vel=mover.velocity,
            local_up=local_up,
        )

        forward = mover.orientation[:, 0]
        forward_horizontal = forward - np.dot(forward, local_up) * local_up
        current_pitch = np.arctan2(
            np.dot(forward, local_up),
            max(np.linalg.norm(forward_horizontal), 1e-6),
        )
        pitch_error = self.launch_pitch_angle - current_pitch

        mover.roll_cmd = np.clip(-self.k_heading * heading_error, -100.0, 100.0)
        mover.pitch_cmd = np.clip(self.k_boost_pitch * pitch_error, -100.0, 100.0)
        mover.yaw_cmd = 0.0
        mover.thrust_cmd = 0.0

    def _desired_cruise_horizontal_direction(self, mover):
        # `cruise_heading` is defined in the mover's local ENU frame, so the desired
        # world-frame horizontal direction must be recomputed from the current position
        # rather than captured once at cruise entry.
        east, north, _ = self._local_horizontal_basis(mover.position)
        desired_horizontal = (
            np.cos(self.cruise_heading) * north
            + np.sin(self.cruise_heading) * east
        )
        desired_norm = np.linalg.norm(desired_horizontal)
        if desired_norm <= 1e-6:
            return north
        return desired_horizontal / desired_norm

    def _enter_hold_mode(self, mover):
        # Reuse the base class's empty-route hold path for cruise, but seed it with the
        # mission targets instead of freezing the post-boost speed/altitude/direction.
        self.hold_active = True
        self.hold_speed = self.cruise_speed
        self.hold_altitude = self.cruise_altitude
        self.hold_horizontal_direction = self._desired_cruise_horizontal_direction(mover)

    def _update_hold_mode(self, mover):
        # The inherited hold update would keep following the single ECEF direction stored
        # in `hold_horizontal_direction`. That is not enough for a local ENU heading,
        # because the corresponding world-frame direction changes as the missile moves.
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
        # This missile's lateral response uses the opposite roll-command sign from the
        # generic FixedWingAutopilot hold logic, so flip the inherited bank command here
        # before adding scenario-specific yaw assistance.
        mover.roll_cmd *= -1.0
        # FixedWingAutopilot's generic hold mode uses bank-only heading correction.
        # Add direct yaw assistance here because this missile scenario needs stronger
        # lateral convergence than the default empty-route hold behavior provides.
        mover.yaw_cmd = np.clip(self.k_cruise_yaw * heading_error, -100.0, 100.0)

    def initialize(self, engine):
        self.t_launch = engine.t
        self.phase = self.BOOST_PHASE
        self.platform.mover.boost_active = True
        super().initialize(engine)
        if not self._boost_start_published:
            engine.broker.publish("boost_start", self.platform)
            self._boost_start_published = True

    def update( self, t, engine ):
        if self.t_launch is None:
            self.t_launch = t

        if self.phase == self.BOOST_PHASE and t - self.t_launch < self.boost_duration:
            self.platform.mover.boost_active = True
            self._command_boost_attitude(self.platform.mover)
            return

        if self.phase == self.BOOST_PHASE:
            self.platform.mover.boost_active = False
            self.phase = self.CRUISE_PHASE
            self.hold_active = False
            if not self._boost_end_published:
                engine.broker.publish("boost_end", self.platform)
                self._boost_end_published = True

        if self.phase != self.CRUISE_PHASE:
            return
        
        super().update( t, engine )
    


def _initial_orientation_from_heading_pitch(initial_position_ecef, heading, pitch_angle):
    """Return an ECEF forward-right-up orientation matrix for the launch attitude."""
    position = np.asarray(initial_position_ecef, dtype=float)
    if position.shape != (3,):
        raise ValueError("initial_position_ecef must have shape (3,)")

    heading = float(heading)
    pitch_angle = float(pitch_angle)

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

    forward_horizontal = np.cos(heading) * north + np.sin(heading) * east
    forward = np.cos(pitch_angle) * forward_horizontal + np.sin(pitch_angle) * up
    forward_axis, right_axis, up_axis = build_aircraft_body_axes(forward, up)
    return project_to_rotation_matrix(np.column_stack([forward_axis, right_axis, up_axis]))


SCENARIO_EVENT_TOPICS = [
    "platform_registered",
    "boost_start",
    "boost_end",
]


def run_surface_launched_cruise_missile_scenario(
    initial_position_ecef,
    cruise_speed,
    cruise_altitude,
    cruise_heading,
    boost_duration,
    boost_acceleration,
    launch_pitch_angle,
    t_end,
    sample_interval,
    output_group,
):
    """Run a surface-launched cruise-missile mission and save HDF5 telemetry.

    This is the public scenario entry point. Its arguments are mission-level inputs,
    while mover- and controller-specific implementation details remain internal.

    Args:
        initial_position_ecef: Initial missile ECEF position vector in meters.
        cruise_speed: Commanded cruise speed in meters/second.
        cruise_altitude: Commanded cruise altitude above the WGS-84 ellipsoid in meters.
        cruise_heading: Commanded local-ENU azimuth heading in radians.
        boost_duration: Duration of the boost phase in seconds.
        boost_acceleration: Forward boost acceleration command in meters/second^2.
        launch_pitch_angle: Initial launch pitch angle in radians.
        t_end: Maximum scenario run time in seconds.
        sample_interval: HDF5 logging sample interval in seconds.
        output_group: Caller-created `h5py.Group` used as the root for this scenario run.

    Returns:
        A dictionary containing the simulation engine, platform, logger, and output group.
    """

    initial_orientation = _initial_orientation_from_heading_pitch(
        initial_position_ecef,
        cruise_heading,
        launch_pitch_angle,
    )
    initial_velocity = np.zeros(3)
    initial_body_rates = np.zeros(3)
    mover_mass = 10000.0
    boost_thrust = boost_acceleration * mover_mass

    engine = SimulationEngine()
    mover = SurfaceLaunchedCruiseMissileMover(
        initial_position=initial_position_ecef,
        initial_velocity=initial_velocity,
        initial_orientation=initial_orientation,
        initial_body_rates=initial_body_rates,
        mass=mover_mass,
        boost_thrust=boost_thrust,
    )
    controller = SurfaceLaunchedCruiseMissileController(
        cruise_speed=cruise_speed,
        cruise_altitude=cruise_altitude,
        cruise_heading=cruise_heading,
        boost_duration=boost_duration,
        launch_pitch_angle=launch_pitch_angle,
    )
    platform = Platform("surface_cruise_missile", mover, controller)
    engine.register_platform(platform)

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
        "platform": platform,
        "logger": logger,
        "output_group": output_group,
    }




def run_example():
    print("=== Running Scenario: Surface-Launched Cruise Missile ===")

    os.makedirs("output", exist_ok=True)
    output_path = Path("output/scenario_surface_launched_cruise_missile.h5")

    with h5py.File(output_path, "w") as h5:
        result = run_surface_launched_cruise_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.6193, -122.3750, 10.0),
            cruise_speed=250.0,
            cruise_altitude=1200.0,
            cruise_heading=0.0,
            boost_duration=2.0,
            boost_acceleration=30.0,
            launch_pitch_angle=0.35,
            t_end=20.0,
            sample_interval=0.1,
            output_group=h5.create_group("surface_run_001"),
        )

    print(f"Simulation ended at t = {result['engine'].t:.2f}s. Telemetry written to {output_path}")


if __name__ == "__main__":
    run_example()
