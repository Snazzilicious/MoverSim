"""Scenario 3: ballistic missile."""

import numpy as np

from mover_sim.core.controller import Controller
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import HDF5Logger
from mover_sim.core.platform import Platform
from mover_sim.hdf5_utils import validate_output_group
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


REQUIRED_STAGE_FIELDS = (
    "dry_mass",
    "propellant_mass",
    "burn_duration",
    "drag_coefficient",
    "reference_area",
    "separation_delay",
)


SCENARIO_EVENT_TOPICS = [
    "platform_registered",
    "stage_1_burnout",
    "stage_1_separation",
    "stage_2_burnout",
    "stage_2_separation",
    "ballistic_coast_start",
    "spent_stage_ground_impact",
    "active_body_ground_impact",
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


def _validate_stages(name, value):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of stage definitions")
    if len(value) not in (1, 2):
        raise ValueError(f"{name} must contain exactly one or two stage definitions")
    return value


def _validate_stage_definitions(stages):
    validated = []
    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            raise ValueError(f"stages[{index}] must be a dictionary")

        for field in REQUIRED_STAGE_FIELDS:
            if field not in stage:
                raise ValueError(f"stages[{index}] is missing required field '{field}'")

        if "thrust" not in stage and "thrust_profile" not in stage:
            raise ValueError(
                f"stages[{index}] must define either 'thrust' or 'thrust_profile'"
            )

        validated_stage = dict(stage)
        for field in REQUIRED_STAGE_FIELDS:
            validated_stage[field] = _validate_positive_scalar(
                f"stages[{index}].{field}",
                validated_stage[field],
            )

        if "thrust" in validated_stage:
            validated_stage["thrust"] = _validate_positive_scalar(
                f"stages[{index}].thrust",
                validated_stage["thrust"],
            )
        if "thrust_profile" in validated_stage and not validated_stage["thrust_profile"]:
            raise ValueError(f"stages[{index}].thrust_profile must not be empty")

        validated.append(validated_stage)

    return validated


def _local_enu_basis(position_ecef):
    position = _validate_vector3("position_ecef", position_ecef)
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
    return east, north, up


def _derive_ascent_azimuth(initial_position_ecef, target_position_ecef):
    initial_position = _validate_vector3("initial_position_ecef", initial_position_ecef)
    target_position = _validate_vector3("target_position_ecef", target_position_ecef)
    east, north, up = _local_enu_basis(initial_position)

    rel = target_position - initial_position
    rel_horizontal = rel - np.dot(rel, up) * up
    east_component = np.dot(rel_horizontal, east)
    north_component = np.dot(rel_horizontal, north)
    return np.arctan2(east_component, north_component)


def _orientation_from_ascent_azimuth(position_ecef, ascent_azimuth, ascent_pitch):
    position = _validate_vector3("position_ecef", position_ecef)
    ascent_azimuth = _validate_finite_scalar("ascent_azimuth", ascent_azimuth)
    ascent_pitch = _validate_finite_scalar("ascent_pitch", ascent_pitch)
    east, north, up = _local_enu_basis(position)

    forward_horizontal = np.cos(ascent_azimuth) * north + np.sin(ascent_azimuth) * east
    forward = np.cos(ascent_pitch) * forward_horizontal + np.sin(ascent_pitch) * up
    forward_axis, right_axis, up_axis = build_aircraft_body_axes(forward, up)
    return quaternion_from_basis(forward_axis, right_axis, up_axis)


def _derive_ascent_program(initial_position_ecef, target_position_ecef, peak_altitude):
    initial_position = _validate_vector3("initial_position_ecef", initial_position_ecef)
    target_position = _validate_vector3("target_position_ecef", target_position_ecef)
    peak_altitude = _validate_positive_scalar("peak_altitude", peak_altitude)

    _, _, initial_altitude = ecef_to_lla(
        initial_position[0],
        initial_position[1],
        initial_position[2],
    )
    altitude_gain = max(peak_altitude - initial_altitude, 1.0)
    ascent_azimuth = _derive_ascent_azimuth(initial_position, target_position)

    if altitude_gain < 10_000.0:
        initial_ascent_pitch = np.radians(55.0)
    elif altitude_gain < 50_000.0:
        initial_ascent_pitch = np.radians(70.0)
    else:
        initial_ascent_pitch = np.radians(82.0)

    return {
        "ascent_azimuth": ascent_azimuth,
        "initial_ascent_pitch": initial_ascent_pitch,
    }


def _derive_spent_stage_state(active_platform):
    mover = active_platform.mover
    if not isinstance(mover, BallisticMissileMover):
        raise ValueError("active_platform must contain a BallisticMissileMover")

    return {
        "position": np.asarray(mover.position, dtype=float).copy(),
        "velocity": np.asarray(mover.velocity, dtype=float).copy(),
        "orientation": np.asarray(mover.orientation, dtype=float).copy(),
        "body_rates": np.asarray(mover.body_rates, dtype=float).copy(),
    }


def _build_spent_stage(spent_stage_state, stage_definition):
    return SpentStageMover(
        initial_position=spent_stage_state["position"],
        initial_velocity=spent_stage_state["velocity"],
        initial_orientation=spent_stage_state["orientation"],
        initial_body_rates=spent_stage_state["body_rates"],
        mass=stage_definition["dry_mass"],
        area=stage_definition["reference_area"],
        cd0=stage_definition["drag_coefficient"],
    )


def _spawn_spent_stage(engine, active_platform, stage_definition, platform_id):
    spent_stage_state = _derive_spent_stage_state(active_platform)
    spent_stage_mover = _build_spent_stage(spent_stage_state, stage_definition)
    spent_stage_platform = Platform(
        platform_id,
        spent_stage_mover,
        properties={"phase": "separated_ballistic_fall", "frozen": False},
    )
    engine.register_platform(spent_stage_platform)
    return spent_stage_platform


def _spent_stage_has_ground_impact(platform):
    mover = platform.mover
    if not isinstance(mover, SpentStageMover):
        return False
    _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    return altitude < 0.0


def _publish_spent_stage_ground_impact(engine, platform):
    engine.broker.publish("spent_stage_ground_impact", platform)


def _freeze_spent_stage(platform):
    mover = platform.mover
    if not isinstance(mover, SpentStageMover):
        return

    if mover._context is not None:
        state_slice = mover._context.get_state_slice(mover)
        state = mover._context.committed_y[state_slice].copy()
        state[mover.get_velocity_slice()] = 0.0
        state[mover.get_body_rate_slice()] = 0.0
        mover._context.committed_y[state_slice] = state
    else:
        mover._initial_state[mover.get_velocity_slice()] = 0.0
        mover._initial_state[mover.get_body_rate_slice()] = 0.0

    mover.thrust_cmd = 0.0
    mover.roll_moment_cmd = 0.0
    mover.pitch_moment_cmd = 0.0
    mover.yaw_moment_cmd = 0.0
    platform.properties["phase"] = "ground_impact_frozen"
    platform.properties["frozen"] = True


class BallisticMissileMover(Aircraft6DOFMover):
    """Scenario 3 active ballistic missile mover with the standard 13-state layout."""

    DEFAULT_MASS = 5000.0
    DEFAULT_AREA = 3.0
    DEFAULT_CD0 = 0.08
    DEFAULT_T_MAX = 250000.0
    DEFAULT_INERTIA = np.diag([5000.0, 25000.0, 25000.0])
    DEFAULT_ANGULAR_DAMPING = np.array([2.0e4, 6.0e4, 6.0e4])

    def __init__(
        self,
        initial_position,
        initial_velocity=None,
        initial_orientation=None,
        initial_body_rates=None,
        stages=None,
        mass=DEFAULT_MASS,
        inertia=None,
        area=DEFAULT_AREA,
        cd0=DEFAULT_CD0,
        t_max=DEFAULT_T_MAX,
        angular_damping=None,
        use_coriolis=True,
        quaternion_normalization_gain=2.0,
    ):
        """Initialize the active ballistic missile mover for Scenario 3.

        Args:
            initial_position: Initial ECEF position vector in meters.
            initial_velocity: Optional initial ECEF velocity vector in meters/second.
            initial_orientation: Optional initial orientation quaternion `[qw, qx, qy, qz]`.
            initial_body_rates: Optional initial body-rate vector `[p, q, r]` in radians/second.
            stages: Sequence of validated stage definitions for the active vehicle stack.
            mass: Default fallback mass in kilograms when stage data is not provided.
            inertia: Optional 3x3 body inertia matrix.
            area: Default fallback reference aerodynamic area in square meters.
            cd0: Default fallback zero-lift drag coefficient.
            t_max: Maximum commanded forward thrust in newtons.
            angular_damping: Optional body-rate damping coefficients.
            use_coriolis: Whether to include Coriolis acceleration in world-frame dynamics.
            quaternion_normalization_gain: Stabilization gain used to keep the quaternion normalized.
        """
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
        self.stages = list(stages or [])
        self.active_stage_index = 0
        self.current_dry_mass = self.stages[0]["dry_mass"] if self.stages else mass
        self.current_propellant_mass = self.stages[0]["propellant_mass"] if self.stages else 0.0
        self.current_drag_coefficient = self.stages[0]["drag_coefficient"] if self.stages else cd0
        self.current_reference_area = self.stages[0]["reference_area"] if self.stages else area
        self.current_separation_delay = self.stages[0]["separation_delay"] if self.stages else 0.0

    def _active_stage(self):
        if not self.stages or self.active_stage_index >= len(self.stages):
            return None
        return self.stages[self.active_stage_index]

    def update_stage_mass(self, dt):
        stage = self._active_stage()
        if stage is None or self.current_propellant_mass <= 0.0:
            self.mass = self.current_dry_mass + max(self.current_propellant_mass, 0.0)
            return

        propellant_burn_rate = stage["propellant_mass"] / stage["burn_duration"]
        self.current_propellant_mass = max(
            0.0,
            self.current_propellant_mass - propellant_burn_rate * max(dt, 0.0),
        )
        self.mass = self.current_dry_mass + self.current_propellant_mass

    def current_stage_thrust(self, t):
        stage = self._active_stage()
        if stage is None or self.current_propellant_mass <= 0.0:
            return 0.0
        if "thrust" in stage:
            return float(stage["thrust"])

        thrust_profile = stage.get("thrust_profile", [])
        if not thrust_profile:
            return 0.0

        for entry in thrust_profile:
            if isinstance(entry, dict) and entry.get("time") is not None and entry.get("thrust") is not None:
                if t <= float(entry["time"]):
                    return float(entry["thrust"])

        last_entry = thrust_profile[-1]
        if isinstance(last_entry, dict) and last_entry.get("thrust") is not None:
            return float(last_entry["thrust"])
        return 0.0

    def compute_state_derivative(self, t, state):
        pos = state[self.get_position_slice()]
        vel = state[self.get_velocity_slice()]
        quat_state = state[self.get_orientation_slice()]
        quat = normalize_quaternion(quat_state)
        body_rates = state[self.get_body_rate_slice()]

        dpos = vel
        dvel = self._compute_world_acceleration(pos, vel, quat, t)
        quat_dot = quaternion_derivative_from_body_rates(quat, body_rates)
        quat_dot += self.quaternion_normalization_gain * (1.0 - np.dot(quat_state, quat_state)) * quat_state
        body_rates_dot = self._compute_body_rate_derivative(body_rates)
        return np.concatenate([dpos, dvel, quat_dot, body_rates_dot])

    def _compute_world_acceleration(self, pos, vel, quaternion, t=0.0):
        _, _, alt = ecef_to_lla(pos[0], pos[1], pos[2])

        thrust_force_body = np.array([
            np.clip(self.current_stage_thrust(t), 0.0, self.t_max),
            0.0,
            0.0,
        ])
        thrust_force_world = rotate_vector_by_quaternion(thrust_force_body, quaternion)
        drag_force_world = aerodynamic_drag_force(
            vel,
            alt,
            self.current_drag_coefficient,
            self.current_reference_area,
        )

        acceleration = gravity(pos) + (thrust_force_world + drag_force_world) / self.mass
        if self.use_coriolis:
            acceleration += coriolis_acceleration(vel)
        return acceleration


class SpentStageMover(Aircraft6DOFMover):
    """Scenario 3 spent-stage mover with the standard 13-state layout."""

    DEFAULT_MASS = 2000.0
    DEFAULT_AREA = 2.5
    DEFAULT_CD0 = 0.15
    DEFAULT_T_MAX = 0.0
    DEFAULT_INERTIA = np.diag([3000.0, 18000.0, 18000.0])
    DEFAULT_ANGULAR_DAMPING = np.array([1.0e4, 3.0e4, 3.0e4])

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
        """Initialize the passive spent-stage mover for Scenario 3.

        Args:
            initial_position: Initial ECEF position vector in meters at separation.
            initial_velocity: Initial ECEF velocity vector in meters/second at separation.
            initial_orientation: Optional initial orientation quaternion `[qw, qx, qy, qz]`.
            initial_body_rates: Optional initial body-rate vector `[p, q, r]` in radians/second.
            mass: Spent-stage mass in kilograms.
            inertia: Optional 3x3 body inertia matrix.
            area: Reference aerodynamic area in square meters.
            cd0: Zero-lift drag coefficient used by the passive drag model.
            t_max: Maximum thrust command in newtons. This remains zero for spent stages.
            angular_damping: Optional body-rate damping coefficients.
            use_coriolis: Whether to include Coriolis acceleration in world-frame dynamics.
            quaternion_normalization_gain: Stabilization gain used to keep the quaternion normalized.
        """
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

        thrust_force_body = np.array([0.0, 0.0, 0.0])
        thrust_force_world = rotate_vector_by_quaternion(thrust_force_body, quaternion)
        drag_force_world = aerodynamic_drag_force(vel, alt, self.cd0, self.area)

        acceleration = gravity(pos) + (thrust_force_world + drag_force_world) / self.mass
        if self.use_coriolis:
            acceleration += coriolis_acceleration(vel)
        return acceleration


class BallisticMissileController(Controller):
    """Phase-based controller for the active ballistic missile body in Scenario 3."""

    POWERED_ASCENT_STAGE_1 = "powered_ascent_stage_1"
    STAGE_1_SEPARATION = "stage_1_separation"
    POWERED_ASCENT_STAGE_2 = "powered_ascent_stage_2"
    STAGE_2_SEPARATION = "stage_2_separation"
    COAST_BALLISTIC = "coast_ballistic"
    TERMINAL_DESCENT = "terminal_descent"
    IMPACT_TERMINAL = "impact_terminal"

    def __init__(
        self,
        ascent_program,
        stages,
        peak_altitude,
        update_interval=0.05,
    ):
        """Initialize the active ballistic missile controller for Scenario 3.

        Args:
            ascent_program: Derived ascent steering program, including launch azimuth and initial ascent pitch.
            stages: Sequence of validated stage definitions used for burnout and separation timing.
            peak_altitude: Desired ballistic peak altitude in meters, used by descent-phase transition logic.
            update_interval: Controller update period in seconds.
        """
        super().__init__(update_interval=update_interval)
        self.ascent_program = dict(ascent_program)
        self.stages = list(stages)
        self.peak_altitude = float(peak_altitude)
        self.phase = self.POWERED_ASCENT_STAGE_1
        self._stage_1_burnout_published = False
        self._stage_1_separation_start_time = None
        self._stage_1_separation_published = False
        self._stage_2_burnout_published = False
        self._stage_2_separation_start_time = None
        self._stage_2_separation_published = False
        self._ballistic_coast_published = False
        self._active_body_ground_impact_published = False
        self.k_roll_rate = 8.0e4
        self.k_pitch = 3.0e5
        self.k_pitch_rate = 1.0e5
        self.k_yaw = 3.0e5
        self.k_yaw_rate = 1.0e5

    def _command_attitude_hold(self, mover, desired_azimuth, desired_pitch):
        desired_quaternion = _orientation_from_ascent_azimuth(
            mover.position,
            desired_azimuth,
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

    def _has_ground_impact(self, mover):
        _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
        return altitude < 0.0

    def update(self, t, engine):
        mover = self.platform.mover
        if not isinstance(mover, BallisticMissileMover):
            return

        if self.phase != self.IMPACT_TERMINAL and self._has_ground_impact(mover):
            self.phase = self.IMPACT_TERMINAL

        if self.phase == self.IMPACT_TERMINAL:
            mover.thrust_cmd = 0.0
            mover.roll_moment_cmd = 0.0
            mover.pitch_moment_cmd = 0.0
            mover.yaw_moment_cmd = 0.0
            if not self._active_body_ground_impact_published:
                engine.broker.publish("active_body_ground_impact", self.platform)
                self._active_body_ground_impact_published = True
            engine.stop()
            return

        if self.phase in (self.POWERED_ASCENT_STAGE_1, self.POWERED_ASCENT_STAGE_2):
            mover.update_stage_mass(self.update_interval)
            mover.thrust_cmd = mover.current_stage_thrust(t)
            self._command_attitude_hold(
                mover,
                self.ascent_program["ascent_azimuth"],
                self.ascent_program["initial_ascent_pitch"],
            )

        if self.phase == self.POWERED_ASCENT_STAGE_1 and t >= self.stages[0]["burn_duration"]:
            if not self._stage_1_burnout_published:
                engine.broker.publish("stage_1_burnout", self.platform)
                self._stage_1_burnout_published = True
            if self._stage_1_separation_start_time is None:
                self._stage_1_separation_start_time = t
            self.phase = self.STAGE_1_SEPARATION

        if self.phase == self.STAGE_1_SEPARATION:
            separation_delay = self.stages[0]["separation_delay"]
            if self._stage_1_separation_start_time is None:
                self._stage_1_separation_start_time = t
            if t < self._stage_1_separation_start_time + separation_delay:
                mover.thrust_cmd = 0.0
            elif not self._stage_1_separation_published:
                engine.broker.publish("stage_1_separation", self.platform)
                self._stage_1_separation_published = True
                if len(self.stages) > 1:
                    next_stage = self.stages[1]
                    mover.active_stage_index = 1
                    mover.current_dry_mass = next_stage["dry_mass"]
                    mover.current_propellant_mass = next_stage["propellant_mass"]
                    mover.current_drag_coefficient = next_stage["drag_coefficient"]
                    mover.current_reference_area = next_stage["reference_area"]
                    mover.current_separation_delay = next_stage["separation_delay"]
                    mover.mass = mover.current_dry_mass + mover.current_propellant_mass
                    self.phase = self.POWERED_ASCENT_STAGE_2
                else:
                    self.phase = self.COAST_BALLISTIC

        if len(self.stages) > 1 and self.phase == self.POWERED_ASCENT_STAGE_2:
            stage_2_burnout_time = (
                self.stages[0]["burn_duration"]
                + self.stages[0]["separation_delay"]
                + self.stages[1]["burn_duration"]
            )
            if t >= stage_2_burnout_time:
                if not self._stage_2_burnout_published:
                    engine.broker.publish("stage_2_burnout", self.platform)
                    self._stage_2_burnout_published = True
                if self._stage_2_separation_start_time is None:
                    self._stage_2_separation_start_time = t
                self.phase = self.STAGE_2_SEPARATION

        if self.phase == self.STAGE_2_SEPARATION:
            separation_delay = self.stages[1]["separation_delay"]
            if self._stage_2_separation_start_time is None:
                self._stage_2_separation_start_time = t
            if t < self._stage_2_separation_start_time + separation_delay:
                mover.thrust_cmd = 0.0
            elif not self._stage_2_separation_published:
                engine.broker.publish("stage_2_separation", self.platform)
                self._stage_2_separation_published = True
                self.phase = self.COAST_BALLISTIC

        if self.phase == self.COAST_BALLISTIC and not self._ballistic_coast_published:
            engine.broker.publish("ballistic_coast_start", self.platform)
            self._ballistic_coast_published = True

        if self.phase == self.COAST_BALLISTIC:
            mover.thrust_cmd = 0.0
            mover.roll_moment_cmd = -self.k_roll_rate * mover.body_rates[0]
            mover.pitch_moment_cmd = self.k_pitch_rate * mover.body_rates[1]
            mover.yaw_moment_cmd = -self.k_yaw_rate * mover.body_rates[2]

            _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
            local_up = mover.position / np.linalg.norm(mover.position)
            vertical_speed = np.dot(mover.velocity, local_up)
            if vertical_speed < 0.0 and altitude <= 0.5 * self.peak_altitude:
                self.phase = self.TERMINAL_DESCENT


def run_ballistic_missile_scenario(
    initial_position_ecef,
    target_position_ecef,
    peak_altitude,
    stages,
    t_end,
    sample_interval,
    output_group,
):
    """Run Scenario 3 and write HDF5 telemetry for the active missile and spent stages.

    Args:
        initial_position_ecef: Initial ballistic-missile ECEF position vector in meters.
        target_position_ecef: Target ECEF position vector in meters.
        peak_altitude: Desired ballistic peak altitude above the WGS-84 ellipsoid in meters.
        stages: One-stage or two-stage stack definition containing the physical parameters for each stage.
        t_end: Maximum scenario run time in seconds.
        sample_interval: HDF5 logging sample interval in seconds.
        output_group: Caller-created `h5py.Group` used as the root for this scenario run.

    Returns:
        A dictionary containing the simulation engine, active platform, logger, and output group.
    """
    initial_position_ecef = _validate_vector3("initial_position_ecef", initial_position_ecef)
    target_position_ecef = _validate_vector3("target_position_ecef", target_position_ecef)
    peak_altitude = _validate_positive_scalar("peak_altitude", peak_altitude)
    stages = _validate_stages("stages", stages)
    stages = _validate_stage_definitions(stages)
    t_end = _validate_positive_scalar("t_end", t_end)
    sample_interval = _validate_positive_scalar("sample_interval", sample_interval)
    output_group = validate_output_group(output_group)

    ascent_program = _derive_ascent_program(
        initial_position_ecef,
        target_position_ecef,
        peak_altitude,
    )
    initial_orientation = _orientation_from_ascent_azimuth(
        initial_position_ecef,
        ascent_program["ascent_azimuth"],
        ascent_program["initial_ascent_pitch"],
    )
    initial_velocity = np.zeros(3)
    initial_body_rates = np.zeros(3)

    engine = SimulationEngine()
    mover = BallisticMissileMover(
        initial_position=initial_position_ecef,
        initial_velocity=initial_velocity,
        initial_orientation=initial_orientation,
        initial_body_rates=initial_body_rates,
        stages=stages,
    )
    controller = BallisticMissileController(
        ascent_program=ascent_program,
        stages=stages,
        peak_altitude=peak_altitude,
    )
    platform = Platform("ballistic_missile", mover, controller)
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
