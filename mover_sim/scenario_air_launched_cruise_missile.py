"""Scenario 2: air-launched cruise missile."""

from pathlib import Path

import numpy as np

from mover_sim.core.controller import Controller
from mover_sim.core.observer import HDF5Logger
from mover_sim.core.platform import Platform
from mover_sim.core.engine import SimulationEngine
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
    "missile_release",
    "missile_drop_start",
    "missile_drop_end",
    "missile_ignite",
    "missile_cruise_established",
    "mothership_rtb_start",
    "mothership_rtb_arrival",
    "missile_ground_impact",
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


def _validate_non_negative_scalar(name, value):
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if scalar < 0.0:
        raise ValueError(f"{name} must be greater than or equal to 0")
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


def _orientation_from_heading_pitch(position_ecef, heading, pitch_angle):
    position = _validate_vector3("position_ecef", position_ecef)
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


def _velocity_from_heading_speed(position_ecef, heading, speed, flight_path_angle=0.0):
    position = _validate_vector3("position_ecef", position_ecef)
    heading = _validate_finite_scalar("heading", heading)
    speed = _validate_positive_scalar("speed", speed)
    flight_path_angle = _validate_finite_scalar("flight_path_angle", flight_path_angle)

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

    horizontal_direction = np.cos(heading) * north + np.sin(heading) * east
    direction = np.cos(flight_path_angle) * horizontal_direction + np.sin(flight_path_angle) * up
    direction /= np.linalg.norm(direction)
    return speed * direction


def _make_missile_release_callback(mothership_platform, spawn_missile):
    def release_callback(engine):
        spawn_missile(engine, mothership_platform)

    return release_callback


def _derive_missile_release_state(mothership_platform):
    mover = mothership_platform.mover
    if not isinstance(mover, AirLaunchedCruiseMissileMothershipMover):
        raise ValueError("mothership_platform must contain an AirLaunchedCruiseMissileMothershipMover")

    return {
        "position": np.asarray(mover.position, dtype=float).copy(),
        "velocity": np.asarray(mover.velocity, dtype=float).copy(),
        "orientation": np.asarray(mover.orientation, dtype=float).copy(),
        "body_rates": np.asarray(mover.body_rates, dtype=float).copy(),
    }


def _build_released_missile(
    release_state,
    release_time,
    missile_cruise_speed,
    missile_cruise_altitude,
    missile_cruise_heading,
    missile_drop_duration,
):
    mover = AirLaunchedCruiseMissileMover(
        initial_position=release_state["position"],
        initial_velocity=release_state["velocity"],
        initial_orientation=release_state["orientation"],
        initial_body_rates=release_state["body_rates"],
    )
    controller = AirLaunchedCruiseMissileController(
        release_time=release_time,
        cruise_speed=missile_cruise_speed,
        cruise_altitude=missile_cruise_altitude,
        cruise_heading=missile_cruise_heading,
        drop_duration=missile_drop_duration,
    )
    return mover, controller


def _spawn_released_missile(
    engine,
    mothership_platform,
    release_time,
    missile_cruise_speed,
    missile_cruise_altitude,
    missile_cruise_heading,
    missile_drop_duration,
):
    release_state = _derive_missile_release_state(mothership_platform)
    mover, controller = _build_released_missile(
        release_state,
        release_time,
        missile_cruise_speed,
        missile_cruise_altitude,
        missile_cruise_heading,
        missile_drop_duration,
    )
    platform = Platform("released_missile", mover, controller)
    engine.register_platform(platform)
    engine.broker.publish("missile_release", platform)
    return platform


def _begin_mothership_rtb(engine, mothership_platform):
    controller = mothership_platform.controller
    if not isinstance(controller, AirLaunchedCruiseMissileMothershipController):
        raise ValueError("mothership_platform must contain an AirLaunchedCruiseMissileMothershipController")
    controller.phase = controller.POST_RELEASE_RTB_PHASE
    engine.broker.publish("mothership_rtb_start", mothership_platform)


class AirLaunchedCruiseMissileMothershipMover(Aircraft6DOFMover):
    """Scenario 2 mothership mover with the standard 13-state layout."""

    DEFAULT_MASS = 12000.0
    DEFAULT_AREA = 35.0
    DEFAULT_CD0 = 0.02
    DEFAULT_T_MAX = 90000.0
    DEFAULT_INERTIA = np.diag([8.0e4, 1.2e5, 1.0e5])
    DEFAULT_ANGULAR_DAMPING = np.array([5.0e4, 6.0e4, 5.0e4])

    def __init__(
        self,
        initial_position,
        initial_velocity,
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

        thrust_force_body = np.array([
            np.clip(self.thrust_cmd, 0.0, self.t_max),
            0.0,
            0.0,
        ])
        thrust_force_world = rotate_vector_by_quaternion(thrust_force_body, quaternion)
        drag_force_world = aerodynamic_drag_force(vel, alt, self.cd0, self.area)

        acceleration = gravity(pos) + (thrust_force_world + drag_force_world) / self.mass
        if self.use_coriolis:
            acceleration += coriolis_acceleration(vel)
        return acceleration


class AirLaunchedCruiseMissileMover(Aircraft6DOFMover):
    """Scenario 2 released missile mover with the standard 13-state layout."""

    DEFAULT_MASS = 1000.0
    DEFAULT_AREA = 2.0
    DEFAULT_CD0 = 0.08
    DEFAULT_T_MAX = 120000.0
    DEFAULT_INERTIA = np.diag([1800.0, 9000.0, 9000.0])
    DEFAULT_ANGULAR_DAMPING = np.array([8.0e3, 2.0e4, 2.0e4])

    def __init__(
        self,
        initial_position,
        initial_velocity,
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

        thrust_force_body = np.array([
            np.clip(self.thrust_cmd, 0.0, self.t_max),
            0.0,
            0.0,
        ])
        thrust_force_world = rotate_vector_by_quaternion(thrust_force_body, quaternion)
        drag_force_world = aerodynamic_drag_force(vel, alt, self.cd0, self.area)

        acceleration = gravity(pos) + (thrust_force_world + drag_force_world) / self.mass
        if self.use_coriolis:
            acceleration += coriolis_acceleration(vel)
        return acceleration


class AirLaunchedCruiseMissileMothershipController(Controller):
    """Phase-based mothership controller for Scenario 2."""

    PRE_RELEASE_CRUISE_PHASE = "pre_release_cruise"
    POST_RELEASE_RTB_PHASE = "post_release_rtb"
    RTB_HOLD_PHASE = "rtb_hold"

    def __init__(
        self,
        cruise_speed,
        cruise_altitude,
        cruise_heading,
        rtb_position_ecef,
        update_interval=0.05,
    ):
        super().__init__(update_interval=update_interval)
        self.cruise_speed = float(cruise_speed)
        self.cruise_altitude = float(cruise_altitude)
        self.cruise_heading = float(cruise_heading)
        self.rtb_position_ecef = _validate_vector3("rtb_position_ecef", rtb_position_ecef)
        self.phase = self.PRE_RELEASE_CRUISE_PHASE
        self.k_roll_rate = 6.0e4
        self.k_speed = 1200.0
        self.k_altitude_to_pitch = 2.0e-4
        self.k_pitch = 2.0e5
        self.k_pitch_rate = 8.0e4
        self.k_yaw = 2.0e5
        self.k_yaw_rate = 8.0e4
        self.max_pitch_command = np.radians(20.0)
        self.rtb_arrival_radius = 200.0
        self._rtb_arrival_published = False

    def _command_attitude_hold(self, mover, desired_heading, desired_pitch):
        desired_quaternion = _orientation_from_heading_pitch(
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

    def _heading_to_point(self, position, target_position):
        local_up = position / np.linalg.norm(position)
        lat_deg, lon_deg, _ = ecef_to_lla(position[0], position[1], position[2])
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.array([
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ])

        rel = target_position - position
        rel_horizontal = rel - np.dot(rel, local_up) * local_up
        east_component = np.dot(rel_horizontal, east)
        north_component = np.dot(rel_horizontal, north)
        return np.arctan2(east_component, north_component)

    def update(self, t, engine):
        mover = self.platform.mover
        if not isinstance(mover, AirLaunchedCruiseMissileMothershipMover):
            return

        if self.phase == self.PRE_RELEASE_CRUISE_PHASE:
            speed = np.linalg.norm(mover.velocity)
            _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
            altitude_error = self.cruise_altitude - altitude
            desired_pitch = np.clip(
                self.k_altitude_to_pitch * altitude_error,
                -self.max_pitch_command,
                self.max_pitch_command,
            )

            mover.thrust_cmd = np.clip(
                mover.mass * self.k_speed * max(self.cruise_speed - speed, 0.0),
                0.0,
                mover.t_max,
            )
            self._command_attitude_hold(mover, self.cruise_heading, desired_pitch)

        if self.phase == self.POST_RELEASE_RTB_PHASE:
            speed = np.linalg.norm(mover.velocity)
            _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
            _, _, target_altitude = ecef_to_lla(
                self.rtb_position_ecef[0],
                self.rtb_position_ecef[1],
                self.rtb_position_ecef[2],
            )
            altitude_error = target_altitude - altitude
            desired_pitch = np.clip(
                self.k_altitude_to_pitch * altitude_error,
                -self.max_pitch_command,
                self.max_pitch_command,
            )
            desired_heading = self._heading_to_point(mover.position, self.rtb_position_ecef)

            mover.thrust_cmd = np.clip(
                mover.mass * self.k_speed * max(self.cruise_speed - speed, 0.0),
                0.0,
                mover.t_max,
            )
            self._command_attitude_hold(mover, desired_heading, desired_pitch)

            if np.linalg.norm(self.rtb_position_ecef - mover.position) <= self.rtb_arrival_radius:
                if not self._rtb_arrival_published:
                    engine.broker.publish("mothership_rtb_arrival", self.platform)
                    self._rtb_arrival_published = True
                self.phase = self.RTB_HOLD_PHASE

        if self.phase == self.RTB_HOLD_PHASE:
            mover.thrust_cmd = 0.0
            mover.roll_moment_cmd = -self.k_roll_rate * mover.body_rates[0]
            mover.pitch_moment_cmd = self.k_pitch_rate * mover.body_rates[1]
            mover.yaw_moment_cmd = -self.k_yaw_rate * mover.body_rates[2]


class AirLaunchedCruiseMissileController(Controller):
    """Phase-based released-missile controller for Scenario 2."""

    DROP_PHASE = "drop"
    IGNITE_TRANSITION_PHASE = "ignite_transition"
    CRUISE_PHASE = "cruise"
    IMPACT_PHASE = "impact_terminal"

    def __init__(
        self,
        release_time,
        cruise_speed,
        cruise_altitude,
        cruise_heading,
        drop_duration,
        update_interval=0.05,
    ):
        super().__init__(update_interval=update_interval)
        self.release_time = float(release_time)
        self.cruise_speed = float(cruise_speed)
        self.cruise_altitude = float(cruise_altitude)
        self.cruise_heading = float(cruise_heading)
        self.drop_duration = float(drop_duration)
        self.phase = self.DROP_PHASE
        self._drop_start_published = False
        self._drop_end_published = False
        self._ignite_published = False
        self._cruise_established_published = False
        self._ground_impact_published = False
        self._release_orientation = None
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
        if not self._drop_start_published:
            engine.broker.publish("missile_drop_start", self.platform)
            self._drop_start_published = True
        mover = self.platform.mover
        if isinstance(mover, AirLaunchedCruiseMissileMover):
            self._release_orientation = np.asarray(mover.orientation, dtype=float).copy()

    def _command_release_attitude_hold(self, mover):
        if self._release_orientation is None:
            return

        desired_forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], self._release_orientation)
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

    def _command_attitude_hold(self, mover, desired_heading, desired_pitch):
        desired_quaternion = _orientation_from_heading_pitch(
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
        desired_quaternion = _orientation_from_heading_pitch(mover.position, desired_heading, 0.0)
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
        if not isinstance(mover, AirLaunchedCruiseMissileMover):
            return

        if self.phase != self.IMPACT_PHASE and self._has_ground_impact(mover):
            self.phase = self.IMPACT_PHASE

        if self.phase == self.IMPACT_PHASE:
            mover.thrust_cmd = 0.0
            mover.roll_moment_cmd = 0.0
            mover.pitch_moment_cmd = 0.0
            mover.yaw_moment_cmd = 0.0
            if not self._ground_impact_published:
                engine.broker.publish("missile_ground_impact", self.platform)
                self._ground_impact_published = True
            engine.stop()
            return

        if self.phase == self.DROP_PHASE and t < self.release_time + self.drop_duration:
            mover.thrust_cmd = 0.0
            self._command_release_attitude_hold(mover)

        if self.phase == self.DROP_PHASE and t >= self.release_time + self.drop_duration:
            if not self._drop_end_published:
                engine.broker.publish("missile_drop_end", self.platform)
                self._drop_end_published = True
            if not self._ignite_published:
                engine.broker.publish("missile_ignite", self.platform)
                self._ignite_published = True
            self.phase = self.IGNITE_TRANSITION_PHASE

        if self.phase == self.IGNITE_TRANSITION_PHASE:
            speed = np.linalg.norm(mover.velocity)
            _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
            altitude_error = self.cruise_altitude - altitude
            desired_pitch = np.clip(
                self.k_altitude_to_pitch * altitude_error,
                -self.max_pitch_command,
                self.max_pitch_command,
            )

            mover.thrust_cmd = np.clip(
                mover.mass * self.k_speed * max(self.cruise_speed - speed, 0.0),
                0.0,
                mover.t_max,
            )
            self._command_attitude_hold(mover, self.cruise_heading, desired_pitch)

            if self._is_cruise_established(mover):
                if not self._cruise_established_published:
                    engine.broker.publish("missile_cruise_established", self.platform)
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

            mover.thrust_cmd = np.clip(
                mover.mass * self.k_speed * max(self.cruise_speed - speed, 0.0),
                0.0,
                mover.t_max,
            )
            self._command_attitude_hold(mover, self.cruise_heading, desired_pitch)


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
    output_path,
):
    """Run the air-launched cruise missile scenario."""
    mothership_initial_position_ecef = _validate_vector3(
        "mothership_initial_position_ecef",
        mothership_initial_position_ecef,
    )
    mothership_cruise_speed = _validate_positive_scalar(
        "mothership_cruise_speed",
        mothership_cruise_speed,
    )
    mothership_cruise_altitude = _validate_finite_scalar(
        "mothership_cruise_altitude",
        mothership_cruise_altitude,
    )
    mothership_cruise_heading = _validate_finite_scalar(
        "mothership_cruise_heading",
        mothership_cruise_heading,
    )
    mothership_rtb_position_ecef = _validate_vector3(
        "mothership_rtb_position_ecef",
        mothership_rtb_position_ecef,
    )
    missile_launch_time = _validate_non_negative_scalar(
        "missile_launch_time",
        missile_launch_time,
    )
    missile_cruise_speed = _validate_positive_scalar(
        "missile_cruise_speed",
        missile_cruise_speed,
    )
    missile_cruise_altitude = _validate_finite_scalar(
        "missile_cruise_altitude",
        missile_cruise_altitude,
    )
    missile_cruise_heading = _validate_finite_scalar(
        "missile_cruise_heading",
        missile_cruise_heading,
    )
    missile_drop_duration = _validate_non_negative_scalar(
        "missile_drop_duration",
        missile_drop_duration,
    )
    t_end = _validate_positive_scalar("t_end", t_end)
    sample_interval = _validate_positive_scalar("sample_interval", sample_interval)
    output_path = _validate_output_path(output_path)

    if missile_launch_time > t_end:
        raise ValueError("missile_launch_time must be less than or equal to t_end")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mothership_initial_orientation = _orientation_from_heading_pitch(
        mothership_initial_position_ecef,
        mothership_cruise_heading,
        0.0,
    )
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
        initial_orientation=mothership_initial_orientation,
        initial_body_rates=mothership_initial_body_rates,
    )
    mothership_controller = AirLaunchedCruiseMissileMothershipController(
        cruise_speed=mothership_cruise_speed,
        cruise_altitude=mothership_cruise_altitude,
        cruise_heading=mothership_cruise_heading,
        rtb_position_ecef=mothership_rtb_position_ecef,
    )
    mothership_platform = Platform("mothership", mothership_mover, mothership_controller)
    engine.register_platform(mothership_platform)

    def spawn_missile(current_engine, current_mothership_platform):
        _spawn_released_missile(
            current_engine,
            current_mothership_platform,
            missile_launch_time,
            missile_cruise_speed,
            missile_cruise_altitude,
            missile_cruise_heading,
            missile_drop_duration,
        )
        _begin_mothership_rtb(current_engine, current_mothership_platform)

    release_callback = _make_missile_release_callback(mothership_platform, spawn_missile)
    engine.schedule(missile_launch_time, release_callback, "MissileRelease")

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
        "mothership_platform": mothership_platform,
        "logger": logger,
        "output_path": output_path,
    }
