"""Scenario 1: surface-launched cruise missile."""

from pathlib import Path

import numpy as np

from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import HDF5Logger
from mover_sim.core.platform import Platform
from mover_sim.core.controller import Controller
from mover_sim.math.coordinates import ecef_to_lla
from mover_sim.math.orientation import (
    build_aircraft_body_axes,
    normalize_quaternion,
    quaternion_derivative_from_body_rates,
    quaternion_from_basis,
    rotate_vector_by_quaternion,
)
from mover_sim.math.physics import aerodynamic_drag_force, coriolis_acceleration, gravity
from mover_sim.models.aircraft_mover import Aircraft6DOFMover


SCENARIO_EVENT_TOPICS = [
    "platform_registered",
    "boost_start",
    "boost_end",
    "cruise_transition_start",
    "cruise_established",
    "ground_impact",
]


def _validate_vector3(name, value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a length-3 vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_positive_scalar(name, value):
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return scalar


def _validate_finite_scalar(name, value):
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_output_path(output_path):
    path = Path(output_path)
    if not str(path):
        raise ValueError("output_path must not be empty")
    return path


def _initial_orientation_from_heading_pitch(initial_position_ecef, heading, pitch_angle):
    position = _validate_vector3("initial_position_ecef", initial_position_ecef)
    heading = _validate_finite_scalar("heading", heading)
    pitch_angle = _validate_finite_scalar("pitch_angle", pitch_angle)

    lat_deg, lon_deg, _ = ecef_to_lla(position[0], position[1], position[2])
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ])
    up = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])

    forward_horizontal = np.cos(heading) * north + np.sin(heading) * east
    forward = np.cos(pitch_angle) * forward_horizontal + np.sin(pitch_angle) * up
    forward_axis, right_axis, up_axis = build_aircraft_body_axes(forward, up)
    return quaternion_from_basis(forward_axis, right_axis, up_axis)


class SurfaceLaunchedCruiseMissileMover(Aircraft6DOFMover):
    """Scenario 1 rigid-body missile mover with the standard 13-state layout."""

    DEFAULT_MASS = 1000.0
    DEFAULT_AREA = 2.0
    DEFAULT_CD0 = 0.08
    DEFAULT_T_MAX = 120000.0
    DEFAULT_INERTIA = np.diag([1800.0, 9000.0, 9000.0])
    DEFAULT_ANGULAR_DAMPING = np.array([8.0e3, 2.0e4, 2.0e4])
    DEFAULT_BOOST_ACCELERATION = 0.0

    def __init__(
        self,
        initial_position,
        initial_velocity=None,
        initial_orientation=None,
        initial_body_rates=None,
        mass=DEFAULT_MASS,
        inertia=None,
        area=DEFAULT_AREA,
        cd0=DEFAULT_CD0,
        t_max=DEFAULT_T_MAX,
        angular_damping=None,
        use_coriolis=True,
        quaternion_normalization_gain=2.0,
    ):
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        if inertia is None:
            inertia = self.DEFAULT_INERTIA.copy()
        if angular_damping is None:
            angular_damping = self.DEFAULT_ANGULAR_DAMPING.copy()

        super().__init__(
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            initial_orientation=initial_orientation,
            initial_body_rates=initial_body_rates,
            mass=mass,
            inertia=inertia,
            area=area,
            cd0=cd0,
            t_max=t_max,
            angular_damping=angular_damping,
            use_coriolis=use_coriolis,
            quaternion_normalization_gain=quaternion_normalization_gain,
        )
        self.boost_acceleration_cmd = self.DEFAULT_BOOST_ACCELERATION

    def set_boost_acceleration(self, boost_acceleration):
        self.boost_acceleration_cmd = max(0.0, float(boost_acceleration))

    def compute_state_derivative(self, t, state):
        pos = state[self.get_position_slice()]
        vel = state[self.get_velocity_slice()]
        quat_state = state[self.get_orientation_slice()]
        quat = normalize_quaternion(quat_state)
        body_rates = state[self.get_body_rate_slice()]

        dpos = vel
        dvel = self._compute_world_acceleration(pos, vel, quat)
        quat_dot = quaternion_derivative_from_body_rates(quat, body_rates)
        quat_dot += self.quaternion_normalization_gain * (1.0 - np.dot(quat_state, quat_state)) * quat_state
        body_rates_dot = self._compute_body_rate_derivative(body_rates)
        return np.concatenate([dpos, dvel, quat_dot, body_rates_dot])

    def _compute_world_acceleration(self, pos, vel, quaternion):
        _, _, alt = ecef_to_lla(pos[0], pos[1], pos[2])

        boost_force = self.mass * max(0.0, self.boost_acceleration_cmd)
        forward_force = np.clip(self.thrust_cmd, 0.0, self.t_max) + boost_force

        thrust_force_body = np.array([
            forward_force,
            0.0,
            0.0,
        ])
        thrust_force_world = rotate_vector_by_quaternion(thrust_force_body, quaternion)
        drag_force_world = aerodynamic_drag_force(vel, alt, self.cd0, self.area)

        acceleration = gravity(pos) + (thrust_force_world + drag_force_world) / self.mass
        if self.use_coriolis:
            acceleration += coriolis_acceleration(vel)
        return acceleration


class SurfaceLaunchedCruiseMissileController(Controller):
    """Phase-based guidance controller for Scenario 1."""

    BOOST_PHASE = "boost"
    TRANSITION_PHASE = "transition_to_cruise"
    CRUISE_PHASE = "cruise"
    IMPACT_PHASE = "impact_terminal"

    def __init__(
        self,
        cruise_speed,
        cruise_altitude,
        cruise_heading,
        boost_duration,
        boost_acceleration,
        launch_pitch_angle,
        update_interval=0.05,
    ):
        super().__init__(update_interval=update_interval)
        self.cruise_speed = float(cruise_speed)
        self.cruise_altitude = float(cruise_altitude)
        self.cruise_heading = float(cruise_heading)
        self.boost_duration = float(boost_duration)
        self.boost_acceleration = float(boost_acceleration)
        self.launch_pitch_angle = float(launch_pitch_angle)
        self.phase = self.BOOST_PHASE
        self.cruise_established = False
        self._boost_start_published = False
        self._boost_end_published = False
        self._transition_start_published = False
        self._cruise_established_published = False
        self._ground_impact_published = False
        self.k_roll_rate = 6.0e4
        self.k_speed = 1200.0
        self.k_altitude_to_pitch = 2.0e-4
        self.k_pitch = 2.0e5
        self.k_pitch_rate = 8.0e4
        self.k_yaw = 2.0e5
        self.k_yaw_rate = 8.0e4
        self.max_pitch_command = np.radians(25.0)
        self.cruise_heading_tolerance = np.radians(5.0)
        self.cruise_altitude_tolerance = 100.0
        self.cruise_speed_tolerance = 20.0

    def initialize(self, engine):
        super().initialize(engine)
        if not self._boost_start_published:
            engine.broker.publish("boost_start", self.platform)
            self._boost_start_published = True

    def _command_attitude_hold(self, mover, desired_heading, desired_pitch):
        desired_quaternion = _initial_orientation_from_heading_pitch(
            mover.position,
            desired_heading,
            desired_pitch,
        )
        desired_forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], desired_quaternion)
        current_forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], mover.orientation)

        local_up = mover.position / np.linalg.norm(mover.position)
        desired_horizontal = desired_forward - np.dot(desired_forward, local_up) * local_up
        current_horizontal = current_forward - np.dot(current_forward, local_up) * local_up

        desired_horizontal_norm = max(np.linalg.norm(desired_horizontal), 1e-8)
        current_horizontal_norm = max(np.linalg.norm(current_horizontal), 1e-8)
        desired_horizontal = desired_horizontal / desired_horizontal_norm
        current_horizontal = current_horizontal / current_horizontal_norm

        heading_error = np.arctan2(
            np.dot(np.cross(current_horizontal, desired_horizontal), local_up),
            np.clip(np.dot(current_horizontal, desired_horizontal), -1.0, 1.0),
        )
        desired_pitch_angle = np.arctan2(np.dot(desired_forward, local_up), desired_horizontal_norm)
        current_pitch_angle = np.arctan2(np.dot(current_forward, local_up), current_horizontal_norm)
        pitch_error = desired_pitch_angle - current_pitch_angle

        mover.roll_moment_cmd = -self.k_roll_rate * mover.body_rates[0]
        mover.pitch_moment_cmd = self.k_pitch * pitch_error + self.k_pitch_rate * mover.body_rates[1]
        mover.yaw_moment_cmd = self.k_yaw * heading_error - self.k_yaw_rate * mover.body_rates[2]

    def _compute_heading_error(self, mover, desired_heading):
        desired_quaternion = _initial_orientation_from_heading_pitch(mover.position, desired_heading, 0.0)
        desired_forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], desired_quaternion)
        current_forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], mover.orientation)

        local_up = mover.position / np.linalg.norm(mover.position)
        desired_horizontal = desired_forward - np.dot(desired_forward, local_up) * local_up
        current_horizontal = current_forward - np.dot(current_forward, local_up) * local_up

        desired_horizontal /= max(np.linalg.norm(desired_horizontal), 1e-8)
        current_horizontal /= max(np.linalg.norm(current_horizontal), 1e-8)
        return np.arctan2(
            np.dot(np.cross(current_horizontal, desired_horizontal), local_up),
            np.clip(np.dot(current_horizontal, desired_horizontal), -1.0, 1.0),
        )

    def _is_cruise_established(self, mover):
        speed = np.linalg.norm(mover.velocity)
        _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
        heading_error = self._compute_heading_error(mover, self.cruise_heading)
        return (
            abs(speed - self.cruise_speed) <= self.cruise_speed_tolerance
            and abs(altitude - self.cruise_altitude) <= self.cruise_altitude_tolerance
            and abs(heading_error) <= self.cruise_heading_tolerance
        )

    def _has_ground_impact(self, mover):
        _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
        return altitude <= 0.0

    def update(self, t, engine):
        mover = self.platform.mover
        if not isinstance(mover, SurfaceLaunchedCruiseMissileMover):
            return

        if self.phase != self.IMPACT_PHASE and self._has_ground_impact(mover):
            self.phase = self.IMPACT_PHASE

        if self.phase == self.IMPACT_PHASE:
            mover.set_boost_acceleration(0.0)
            mover.thrust_cmd = 0.0
            mover.roll_moment_cmd = 0.0
            mover.pitch_moment_cmd = 0.0
            mover.yaw_moment_cmd = 0.0
            if not self._ground_impact_published:
                engine.broker.publish("ground_impact", self.platform)
                self._ground_impact_published = True
            engine.stop()
            return

        if self.phase == self.BOOST_PHASE and t < self.boost_duration:
            mover.set_boost_acceleration(self.boost_acceleration)
            mover.thrust_cmd = 0.0
            self._command_attitude_hold(mover, self.cruise_heading, self.launch_pitch_angle)

        if self.phase == self.BOOST_PHASE and t >= self.boost_duration and not self._boost_end_published:
            engine.broker.publish("boost_end", self.platform)
            self._boost_end_published = True

        if self.phase == self.BOOST_PHASE and t >= self.boost_duration:
            mover.set_boost_acceleration(0.0)
            self.phase = self.TRANSITION_PHASE
            if not self._transition_start_published:
                engine.broker.publish("cruise_transition_start", self.platform)
                self._transition_start_published = True

        if self.phase == self.TRANSITION_PHASE:
            speed = np.linalg.norm(mover.velocity)
            _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
            altitude_error = self.cruise_altitude - altitude
            desired_pitch = np.clip(
                self.k_altitude_to_pitch * altitude_error,
                -self.max_pitch_command,
                self.max_pitch_command,
            )

            mover.set_boost_acceleration(0.0)
            mover.thrust_cmd = np.clip(
                mover.mass * self.k_speed * max(self.cruise_speed - speed, 0.0),
                0.0,
                mover.t_max,
            )
            self._command_attitude_hold(mover, self.cruise_heading, desired_pitch)

            if not self.cruise_established and self._is_cruise_established(mover):
                self.cruise_established = True
                if not self._cruise_established_published:
                    engine.broker.publish("cruise_established", self.platform)
                    self._cruise_established_published = True
                self.phase = self.CRUISE_PHASE

        if self.phase == self.CRUISE_PHASE:
            speed = np.linalg.norm(mover.velocity)
            _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
            altitude_error = self.cruise_altitude - altitude
            desired_pitch = np.clip(
                self.k_altitude_to_pitch * altitude_error,
                -0.5 * self.max_pitch_command,
                0.5 * self.max_pitch_command,
            )

            mover.set_boost_acceleration(0.0)
            mover.thrust_cmd = np.clip(
                mover.mass * self.k_speed * max(self.cruise_speed - speed, 0.0),
                0.0,
                mover.t_max,
            )
            self._command_attitude_hold(mover, self.cruise_heading, desired_pitch)


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
    output_path,
):
    """Run the surface-launched cruise missile scenario."""
    initial_position_ecef = _validate_vector3("initial_position_ecef", initial_position_ecef)
    cruise_speed = _validate_positive_scalar("cruise_speed", cruise_speed)
    cruise_altitude = _validate_finite_scalar("cruise_altitude", cruise_altitude)
    cruise_heading = _validate_finite_scalar("cruise_heading", cruise_heading)
    boost_duration = _validate_positive_scalar("boost_duration", boost_duration)
    boost_acceleration = _validate_positive_scalar("boost_acceleration", boost_acceleration)
    launch_pitch_angle = _validate_finite_scalar("launch_pitch_angle", launch_pitch_angle)
    t_end = _validate_positive_scalar("t_end", t_end)
    sample_interval = _validate_positive_scalar("sample_interval", sample_interval)
    output_path = _validate_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_orientation = _initial_orientation_from_heading_pitch(
        initial_position_ecef,
        cruise_heading,
        launch_pitch_angle,
    )
    initial_velocity = np.zeros(3)
    initial_body_rates = np.zeros(3)

    engine = SimulationEngine()
    mover = SurfaceLaunchedCruiseMissileMover(
        initial_position=initial_position_ecef,
        initial_velocity=initial_velocity,
        initial_orientation=initial_orientation,
        initial_body_rates=initial_body_rates,
    )
    controller = SurfaceLaunchedCruiseMissileController(
        cruise_speed=cruise_speed,
        cruise_altitude=cruise_altitude,
        cruise_heading=cruise_heading,
        boost_duration=boost_duration,
        boost_acceleration=boost_acceleration,
        launch_pitch_angle=launch_pitch_angle,
    )
    platform = Platform("surface_cruise_missile", mover, controller)
    engine.register_platform(platform)

    logger = HDF5Logger(
        engine,
        str(output_path),
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
        "output_path": output_path,
    }
