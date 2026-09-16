import numpy as np
from mover_sim.core.mover import IntegratedMover, TranslationalMover, TranslationalIntegratedMover
from mover_sim.core.controller import Controller
from mover_sim.core.engine import Event
from mover_sim.math.physics import aerodynamic_drag_force, air_density, centrifugal_acceleration, coriolis_acceleration, coriolis_vector, gravity, GM
from mover_sim.math.coordinates import ecef_to_lla, ecef_to_enu, lla_to_ecef
from mover_sim.math.orientation import (
    build_aircraft_body_axes,
    normalize_quaternion,
    project_to_rotation_matrix,
    quaternion_derivative_from_body_rates,
    quaternion_from_basis,
    rotate_vector_by_quaternion,
    skew_symmetric_to_vector,
    vector_to_skew_symmetric,
)

class AircraftMover(TranslationalIntegratedMover):
    """
    An integrated mover representing a simplified 6-DOF style aircraft.
    Uses thrust, lift, drag, and gravity to compute forces.
    """
    def __init__(self, initial_position, initial_velocity, mass=10000.0, area=30.0, cd0=0.02, t_max=80000.0):
        """
        Parameters:
            initial_position: ECEF coordinates [X, Y, Z] in meters.
            initial_velocity: ECEF velocity [Vx, Vy, Vz] in m/s.
            mass: Mass of the aircraft in kg.
            area: Wing reference area in m^2.
            cd0: Zero-lift drag coefficient (dimensionless).
            t_max: Maximum engine thrust in Newtons.
        """
        super().__init__(initial_position, initial_velocity=initial_velocity)
        self.mass = mass
        self.area = area
        self.cd0 = cd0
        self.t_max = t_max
        
        # Control inputs (can be set by a Controller/Autopilot)
        self.thrust_cmd = 0.0       # Thrust force in Newtons
        self.bank_angle_cmd = 0.0   # Bank/roll angle in radians
        self.lift_cmd = 0.0         # Commanded lift force in Newtons

    def compute_derivatives(self, t, pos, vel):
        """
        Compute derivatives (dPos/dt, dVel/dt) including aerodynamic forces, thrust, gravity, and Coriolis.
        """
        dpos, dvel = super().compute_derivatives(t, pos, vel)
        dvel = dvel + gravity(pos) + coriolis_acceleration(vel)
        
        v_mag = np.linalg.norm(vel)
        if v_mag < 1.0:
            # Avoid divide-by-zero for orientation at rest
            return dpos, dvel
            
        # Get altitude for air density
        lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
        
        # 2. Build local orientation vectors in ECEF
        u_v = vel / v_mag                      # Flight path / velocity direction
        u_pos = pos / np.linalg.norm(pos)      # Radial outward (approx local vertical)
        
        # Right wing vector (horizontal perpendicular to velocity)
        u_right = np.cross(u_v, u_pos)
        u_right_mag = np.linalg.norm(u_right)
        if u_right_mag < 1e-6:
            # Handle strict vertical flight case
            u_right = np.array([0.0, 1.0, 0.0])
        else:
            u_right = u_right / u_right_mag
            
        # Local vertical lift direction (upward perpendicular to velocity)
        u_up = np.cross(u_right, u_v)
        
        # Lift vector direction after banking by bank_angle_cmd
        # Positive bank rotates lift vector towards u_right (right roll)
        u_lift = np.cos(self.bank_angle_cmd) * u_up + np.sin(self.bank_angle_cmd) * u_right
        
        # 3. Compute Aerodynamic Drag
        drag_force = aerodynamic_drag_force(vel, alt, self.cd0, self.area)
        
        # 4. Compute Lift Force
        lift_force = self.lift_cmd * u_lift
        
        # 5. Compute Thrust Force
        thrust_force = self.thrust_cmd * u_v
        
        # Total aerodynamic and thrust acceleration
        accel_aero_thrust = (drag_force + lift_force + thrust_force) / self.mass
        
        # Combined acceleration
        dvel = dvel + accel_aero_thrust
        
        return dpos, dvel


class Aircraft6DOFMover(TranslationalMover, IntegratedMover):
    """Rigid-body aircraft mover with translational, attitude, and body-rate state."""

    def __init__(
        self,
        initial_position,
        initial_velocity,
        initial_orientation=None,
        initial_body_rates=None,
        mass=10000.0,
        inertia=None,
        area=30.0,
        cd0=0.02,
        t_max=80000.0,
        angular_damping=None,
        use_coriolis=True,
        quaternion_normalization_gain=2.0,
    ):
        """
        Parameters:
            initial_position: ECEF coordinates [X, Y, Z] in meters.
            initial_velocity: ECEF velocity [Vx, Vy, Vz] in m/s.
            initial_orientation: Optional scalar-first quaternion [w, x, y, z]. If omitted,
                an orientation is derived from the initial velocity and local vertical.
            initial_body_rates: Optional body angular rates [p, q, r] in rad/s.
            mass: Vehicle mass in kg.
            inertia: Body inertia as a `(3, 3)` tensor or `(3,)` principal moments.
            area: Reference area in m^2 used for drag.
            cd0: Zero-lift drag coefficient.
            t_max: Maximum thrust command in Newtons.
            angular_damping: Per-axis angular damping coefficients.
            use_coriolis: If True, include Coriolis acceleration in world-frame translation.
            quaternion_normalization_gain: Stabilization gain used to keep the integrated
                quaternion near unit length.

        State layout:
            [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
        """
        initial_position = np.asarray(initial_position, dtype=float)
        initial_velocity = np.asarray(initial_velocity, dtype=float)
        initial_body_rates = (
            np.asarray(initial_body_rates, dtype=float)
            if initial_body_rates is not None
            else np.zeros(3)
        )

        if initial_orientation is None:
            initial_orientation = self._derive_orientation_from_velocity(initial_position, initial_velocity)
        else:
            initial_orientation = normalize_quaternion(initial_orientation)

        state = np.concatenate([
            initial_position,
            initial_velocity,
            initial_orientation,
            initial_body_rates,
        ])
        super().__init__(state)

        self.mass = float(mass)
        self.inertia = self._coerce_inertia(inertia)
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.area = float(area)
        self.cd0 = float(cd0)
        self.t_max = float(t_max)
        self.angular_damping = self._coerce_angular_damping(angular_damping)
        self.use_coriolis = use_coriolis
        self.quaternion_normalization_gain = float(quaternion_normalization_gain)

        self.thrust_cmd = 0.0
        self.roll_moment_cmd = 0.0
        self.pitch_moment_cmd = 0.0
        self.yaw_moment_cmd = 0.0

    def get_orientation_slice(self):
        return slice(6, 10)

    def get_body_rate_slice(self):
        return slice(10, 13)

    @property
    def orientation(self):
        return self.get_state()[self.get_orientation_slice()]

    @property
    def body_rates(self):
        return self.get_state()[self.get_body_rate_slice()]

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

    def _derive_orientation_from_velocity(self, position, velocity):
        speed = np.linalg.norm(velocity)
        if speed < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])

        pos_norm = np.linalg.norm(position)
        local_vertical = position / pos_norm if pos_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        forward_axis, right_axis, up_axis = build_aircraft_body_axes(velocity, local_vertical)
        return quaternion_from_basis(forward_axis, right_axis, up_axis)

    def _coerce_inertia(self, inertia):
        if inertia is None:
            inertia = np.diag([8.0e4, 1.2e5, 1.0e5])
        inertia = np.asarray(inertia, dtype=float)
        if inertia.shape == (3,):
            inertia = np.diag(inertia)
        if inertia.shape != (3, 3):
            raise ValueError("inertia must have shape (3,) or (3, 3)")
        return inertia

    def _coerce_angular_damping(self, angular_damping):
        if angular_damping is None:
            return np.array([5.0e4, 6.0e4, 5.0e4])
        angular_damping = np.asarray(angular_damping, dtype=float)
        if angular_damping.shape != (3,):
            raise ValueError("angular_damping must have shape (3,)")
        return angular_damping

    def _compute_world_acceleration(self, pos, vel, quaternion):
        lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])

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

    def _compute_body_rate_derivative(self, body_rates):
        moments = np.array([
            self.roll_moment_cmd,
            -self.pitch_moment_cmd,
            self.yaw_moment_cmd,
        ])
        moments -= self.angular_damping * body_rates
        angular_momentum = self.inertia @ body_rates
        return self.inv_inertia @ (moments - np.cross(body_rates, angular_momentum))


class AircraftAutopilot(Controller):
    """
    Autopilot for AircraftMover to follow a series of waypoints at a commanded speed.
    """
    def __init__(self, waypoints, target_speed=150.0, waypoint_radius=500.0, update_interval=0.1):
        """
        Parameters:
            waypoints: List of ECEF coordinates [X, Y, Z] in meters.
            target_speed: Target speed in m/s.
            waypoint_radius: Distance in meters to trigger waypoint completion.
            update_interval: Autopilot execution period (seconds).
        """
        super().__init__(update_interval=update_interval)
        self.waypoints = [np.asarray(wp, dtype=float) for wp in waypoints]
        self.target_speed = target_speed
        self.waypoint_radius = waypoint_radius
        self.current_wp_idx = 0
        self.completed = False

    def update(self, t, engine):
        mover = self.platform.mover
        if not isinstance(mover, AircraftMover):
            return
            
        pos = mover.position
        vel = mover.velocity
        speed = np.linalg.norm(vel)
        
        if self.current_wp_idx >= len(self.waypoints):
            self.completed = True
            # Reached destination: just maintain speed and fly straight (bank=0, lift balances gravity)
            mover.thrust_cmd = 0.5 * air_density(ecef_to_lla(pos[0], pos[1], pos[2])[2]) * mover.cd0 * mover.area * (speed ** 2)
            mover.bank_angle_cmd = 0.0
            mover.lift_cmd = mover.mass * 9.81
            return
            
        # Get target waypoint
        wp_target = self.waypoints[self.current_wp_idx]
        
        # Convert target to ENU relative to current position
        lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
        e, n, u = ecef_to_enu(wp_target[0], wp_target[1], wp_target[2], lat, lon, alt)
        
        # Check waypoint transition
        dist_3d = np.sqrt(e**2 + n**2 + u**2)
        if dist_3d < self.waypoint_radius:
            self.current_wp_idx += 1
            engine.broker.publish("waypoint_reached", self.platform, self.current_wp_idx - 1)
            # Re-run update recursively to update commands for new waypoint
            self.update(t, engine)
            return

        # 1. Heading guidance
        # Convert velocity to ENU to determine current heading
        x_ref, y_ref, z_ref = lla_to_ecef(lat, lon, alt)
        v_e, v_n, v_u = ecef_to_enu(vel[0] + x_ref, vel[1] + y_ref, vel[2] + z_ref, lat, lon, alt)
        
        heading_tgt = np.arctan2(e, n)
        heading_curr = np.arctan2(v_e, v_n)
        
        heading_err = heading_tgt - heading_curr
        # Wrap to [-pi, pi]
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        
        # Proportional roll command to steer
        K_roll = 1.2
        max_bank = np.radians(45.0)
        mover.bank_angle_cmd = np.clip(K_roll * heading_err, -max_bank, max_bank)
        
        # 2. Altitude / Pitch guidance
        # Desired vertical velocity based on vertical distance to target
        K_climb = 0.15
        max_climb = 15.0
        v_u_tgt = np.clip(K_climb * u, -max_climb, max_climb)
        
        # Command vertical acceleration to track climb rate
        K_v_accel = 0.5
        a_u = K_v_accel * (v_u_tgt - v_u)
        
        # Local gravity magnitude at altitude
        r_mag = np.linalg.norm(pos)
        g_local = GM / (r_mag ** 2)
        
        # Commanded lift force: balance gravity, add climb acceleration, compensative for bank angle
        cos_bank = np.cos(mover.bank_angle_cmd)
        cos_bank_safe = max(0.5, cos_bank) # prevent divide-by-zero or excessive lift command
        
        lift_req = (mover.mass * (g_local + a_u)) / cos_bank_safe
        
        # Clamp lift load factor (0.0g to 2.5g)
        max_lift = 2.5 * mover.mass * g_local
        min_lift = 0.0
        mover.lift_cmd = np.clip(lift_req, min_lift, max_lift)
        
        # 3. Speed / Thrust guidance
        # Proportional control for thrust
        K_speed = 0.25
        drag_force_mag = np.linalg.norm(aerodynamic_drag_force(vel, alt, mover.cd0, mover.area))
        
        thrust_req = drag_force_mag + mover.mass * K_speed * (self.target_speed - speed)
        mover.thrust_cmd = np.clip(thrust_req, 0.0, mover.t_max)
        # Note: no solver reset needed — the engine recreates the RK45 solver at the
        # start of each step_continuous call, so updated forces are always picked up.


class Aircraft6DOFAutopilot(Controller):
    """Waypoint-following autopilot for Aircraft6DOFMover."""

    def __init__(self, waypoints, target_speed=150.0, waypoint_radius=500.0, update_interval=0.05):
        """
        Parameters:
            waypoints: List of ECEF coordinates [X, Y, Z] in meters.
            target_speed: Target speed in m/s.
            waypoint_radius: Distance in meters to trigger waypoint completion.
            update_interval: Autopilot execution period (seconds).
        """
        super().__init__(update_interval=update_interval)
        self.waypoints = [np.asarray(wp, dtype=float) for wp in waypoints]
        self.target_speed = float(target_speed)
        self.waypoint_radius = float(waypoint_radius)
        self.current_wp_idx = 0
        self.completed = False

        self.k_speed = 2000.0
        self.k_roll = 2.5e5
        self.k_roll_rate = 8.0e4
        self.k_pitch = 1.2e6
        self.k_pitch_rate = 2.0e5
        self.k_yaw = 1.0e5
        self.k_yaw_rate = 8.0e4

    def update(self, t, engine):
        mover = self.platform.mover
        if not isinstance(mover, Aircraft6DOFMover):
            return

        pos = mover.position
        vel = mover.velocity
        speed = np.linalg.norm(vel)

        if self.current_wp_idx >= len(self.waypoints):
            self.completed = True
            mover.thrust_cmd = 0.0
            mover.roll_moment_cmd = -self.k_roll_rate * mover.body_rates[0]
            mover.pitch_moment_cmd = self.k_pitch_rate * mover.body_rates[1]
            mover.yaw_moment_cmd = -self.k_yaw_rate * mover.body_rates[2]
            return

        wp_target = self.waypoints[self.current_wp_idx]
        rel_pos = wp_target - pos
        distance = np.linalg.norm(rel_pos)

        if distance < self.waypoint_radius:
            self.current_wp_idx += 1
            engine.broker.publish("waypoint_reached", self.platform, self.current_wp_idx - 1)
            self.update(t, engine)
            return

        local_up = pos / np.linalg.norm(pos)
        forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], mover.orientation)
        right = rotate_vector_by_quaternion([0.0, 1.0, 0.0], mover.orientation)

        rel_horizontal = rel_pos - np.dot(rel_pos, local_up) * local_up
        horizontal_norm = np.linalg.norm(rel_horizontal)
        if horizontal_norm > 1e-8:
            desired_horizontal = rel_horizontal / horizontal_norm
        else:
            desired_horizontal = forward - np.dot(forward, local_up) * local_up
            desired_horizontal /= max(np.linalg.norm(desired_horizontal), 1e-8)

        forward_horizontal = forward - np.dot(forward, local_up) * local_up
        forward_horizontal_norm = np.linalg.norm(forward_horizontal)
        if forward_horizontal_norm > 1e-8:
            forward_horizontal = forward_horizontal / forward_horizontal_norm
        else:
            forward_horizontal = desired_horizontal

        heading_error = np.arctan2(
            np.dot(np.cross(forward_horizontal, desired_horizontal), local_up),
            np.clip(np.dot(forward_horizontal, desired_horizontal), -1.0, 1.0),
        )

        desired_flight_path = np.arctan2(np.dot(rel_pos, local_up), max(horizontal_norm, 1e-8))
        current_flight_path = np.arctan2(np.dot(vel, local_up), max(np.linalg.norm(vel - np.dot(vel, local_up) * local_up), 1e-8))
        pitch_error = desired_flight_path - current_flight_path

        lateral_velocity = np.dot(vel, right)
        yaw_correction = lateral_velocity / max(speed, 1.0)

        drag_force_mag = np.linalg.norm(aerodynamic_drag_force(vel, ecef_to_lla(pos[0], pos[1], pos[2])[2], mover.cd0, mover.area))
        thrust_req = drag_force_mag + mover.mass * self.k_speed * (self.target_speed - speed)
        mover.thrust_cmd = np.clip(thrust_req, 0.0, mover.t_max)

        mover.roll_moment_cmd = self.k_roll * heading_error - self.k_roll_rate * mover.body_rates[0]
        mover.pitch_moment_cmd = self.k_pitch * pitch_error + self.k_pitch_rate * mover.body_rates[1]
        mover.yaw_moment_cmd = -self.k_yaw * yaw_correction - self.k_yaw_rate * mover.body_rates[2]



class FixedWingMover(TranslationalMover, IntegratedMover):
    """Rigid-body aircraft mover with translational, attitude, and body-rate state."""

    class OrientationCorrectionEvent(Event) :
        """Periodically projects a mover's committed orientation back onto `SO(3)`.
        """
        def __init__(self, mover, interval=1.0):
            self.mover = mover
            mover_name = mover.platform.id if mover.platform is not None else str(id(mover))
            super().__init__(
                0.0,
                self.fix_orientation,
                name=f"FixedWingOrientationCorrection_{mover_name}",
                interval=interval,
            )
        
        def fix_orientation(self, engine):
            if self.mover._context is not engine.context:
                return
            if self.mover not in engine.context._index_map:
                return

            mover_slice = engine.context.get_state_slice(self.mover)
            orientation_slice = self.mover.get_orientation_slice()
            start = mover_slice.start + orientation_slice.start
            stop = mover_slice.start + orientation_slice.stop

            orientation = engine.context.committed_y[start:stop].reshape((3, 3))
            corrected = project_to_rotation_matrix(orientation)
            engine.context.committed_y[start:stop] = corrected.reshape(-1)


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
    ):
        """
        Parameters:
            initial_position: ECEF coordinates [X, Y, Z] in meters.
            initial_velocity: ECEF velocity [Vx, Vy, Vz] in m/s.
            initial_orientation: Optional mover-frame forward, right, up basis matrix in ECEF. If omitted,
                an orientation is derived from the initial velocity and local vertical.
            initial_body_rates: Optional time derivatives of initial_orientation (axis of rotation with rate as magnitude).
            mass: Vehicle mass in kg.
            rotational_mass: Body inertia as a `(3, 3)` tensor or `(3,)` principal moments.
            area: Reference aerodynamic area in m^2.
            cd0: Zero-angle drag coefficient.
            cd_alpha: Additional drag coefficient applied with `alpha^2`.
            cd_beta: Additional drag coefficient applied with `beta^2`.
            cl_alpha: Lift slope coefficient in body-up direction.
            cy_beta: Sideslip slope coefficient in body-right direction.
            bank_restoring_coeff: Roll restoring moment per radian of bank error.
            alpha_restoring_coeff: Pitch restoring moment per radian of angle of attack.
            beta_restoring_coeff: Yaw restoring moment per radian of sideslip angle.
            roll_damping_coeff: Roll damping moment per rad/s of body roll rate.
            pitch_damping_coeff: Pitch damping moment per rad/s of body pitch rate.
            yaw_damping_coeff: Yaw damping moment per rad/s of body yaw rate.
            max_thrust: Maximum thrust in Newtons.
            max_roll_moment: Maximum roll moment in N*m.
            max_pitch_moment: Maximum pitch moment in N*m.
            max_yaw_moment: Maximum yaw moment in N*m.
            use_coriolis: If True, include Coriolis acceleration in world-frame translation.

        State layout:
            [x, y, z, vx, vy, vz, o11, ..., o33, w1, w2, w3]
        """

        initial_position = np.asarray(initial_position, dtype=float)
        if initial_position.shape != (3,):
            raise ValueError("initial_position must have shape (3,)")

        initial_velocity = np.asarray(initial_velocity, dtype=float)
        if initial_velocity.shape != (3,):
            raise ValueError("initial_velocity must have shape (3,)")

        initial_body_rates = (
            np.asarray(initial_body_rates, dtype=float)
            if initial_body_rates is not None
            else np.zeros(3)
        )
        if initial_body_rates.shape != (3,):
            raise ValueError("initial_body_rates must have shape (3,)")

        if initial_orientation is None:
            initial_orientation = self._derive_orientation_from_velocity(
                initial_position,
                initial_velocity,
            )
        else:
            initial_orientation = np.asarray(initial_orientation, dtype=float)
            if initial_orientation.shape != (3, 3):
                raise ValueError("initial_orientation must have shape (3, 3)")
            initial_orientation = project_to_rotation_matrix(initial_orientation)

        state = np.concatenate([
            initial_position,
            initial_velocity,
            initial_orientation.reshape(-1),
            initial_body_rates,
        ])
        super().__init__(state)

        self.mass = float(mass)
        self.rotational_mass = self._coerce_rotational_mass(rotational_mass)
        self.inv_rotational_mass = np.linalg.inv(self.rotational_mass)
        self.area = float(area)
        self.cd0 = float(cd0)
        self.cd_alpha = float(cd_alpha)
        self.cd_beta = float(cd_beta)
        self.cl_alpha = float(cl_alpha)
        self.cy_beta = float(cy_beta)
        self.bank_restoring_coeff = float(bank_restoring_coeff)
        self.alpha_restoring_coeff = float(alpha_restoring_coeff)
        self.beta_restoring_coeff = float(beta_restoring_coeff)
        self.roll_damping_coeff = float(roll_damping_coeff)
        self.pitch_damping_coeff = float(pitch_damping_coeff)
        self.yaw_damping_coeff = float(yaw_damping_coeff)
        self.max_thrust = float(max_thrust)
        self.max_roll_moment = float(max_roll_moment)
        self.max_pitch_moment = float(max_pitch_moment)
        self.max_yaw_moment = float(max_yaw_moment)
        self.use_coriolis = bool(use_coriolis)

        # All 0-100
        self.thrust_cmd = 0.0
        self.roll_cmd = 0.0
        self.pitch_cmd = 0.0
        self.yaw_cmd = 0.0

    def add_orientation_correction_event(self, engine, interval=1.0):
        """Schedule a recurring event that projects the committed orientation onto `SO(3)`."""
        event = self.OrientationCorrectionEvent(self, interval=interval)
        engine.schedule(engine.t, event.callback, name=event.name, interval=event.interval)
        return event

    def _derive_orientation_from_velocity(self, position, velocity):
        speed = np.linalg.norm(velocity)
        if speed < 1e-8:
            return np.eye(3)

        pos_norm = np.linalg.norm(position)
        local_vertical = position / pos_norm if pos_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        forward, right, up = build_aircraft_body_axes(velocity, local_vertical)
        return project_to_rotation_matrix(np.column_stack([forward, right, up]))

    def _coerce_rotational_mass(self, rotational_mass):
        if rotational_mass is None:
            rotational_mass = np.diag([8.0e4, 1.2e5, 1.0e5])
        rotational_mass = np.asarray(rotational_mass, dtype=float)
        if rotational_mass.shape == (3,):
            rotational_mass = np.diag(rotational_mass)
        if rotational_mass.shape != (3, 3):
            raise ValueError("rotational_mass must have shape (3,) or (3, 3)")
        return rotational_mass

    def get_orientation_slice(self):
        return slice(6, 15)

    @property
    def orientation(self):
        return self.get_state()[self.get_orientation_slice()].reshape((3,3))

    def get_omega_slice(self):
        return slice(15, 18)

    def _thrust_vector( self, forward ):
        # Thrust is applied along the body forward axis with a 0-100 percent command.
        return ( self.thrust_cmd / 100.0 ) * self.max_thrust * forward

    def _aerodynamic_angles(self, pos, vel, orientation):
        speed = np.linalg.norm(vel)
        if speed < 1e-6:
            return speed, 0.0, 0.0, 0.0

        v_hat = vel / speed
        velocity_body = orientation.T @ vel
        forward_speed = max(velocity_body[0], 1e-6)
        alpha = np.arctan2(-velocity_body[2], forward_speed)
        beta = np.arctan2(velocity_body[1], forward_speed)

        local_up = pos / max(np.linalg.norm(pos), 1e-6)
        up_reference = local_up - np.dot(local_up, v_hat) * v_hat
        body_up = orientation[:, 2] - np.dot(orientation[:, 2], v_hat) * v_hat

        up_reference_norm = np.linalg.norm(up_reference)
        body_up_norm = np.linalg.norm(body_up)
        if up_reference_norm < 1e-6 or body_up_norm < 1e-6:
            bank_error = 0.0
        else:
            up_reference /= up_reference_norm
            body_up /= body_up_norm
            bank_error = np.arctan2(
                np.dot(np.cross(up_reference, body_up), v_hat),
                np.clip(np.dot(up_reference, body_up), -1.0, 1.0),
            )

        return speed, alpha, beta, bank_error

    def _aerodynamic_force(self, pos, vel, orientation):
        # Use a simple body-axis aerodynamic model driven by air-relative attitude:
        # x: quadratic drag, y: sideslip force proportional to beta, z: lift proportional to alpha.
        speed, alpha, beta, _ = self._aerodynamic_angles(pos, vel, orientation)
        if speed < 1e-6 or self.area <= 0.0:
            return np.zeros(3)

        _, _, alt = ecef_to_lla(pos[0], pos[1], pos[2])
        rho = air_density(alt)
        dynamic_pressure = 0.5 * rho * (speed ** 2)

        # Drag grows with alpha^2 and beta^2 while lift and side force remain linear in
        # small-angle alpha/beta for a simple, tunable fixed-wing approximation.
        drag_coeff = self.cd0 + self.cd_alpha * (alpha ** 2) + self.cd_beta * (beta ** 2)
        body_force = dynamic_pressure * self.area * np.array([
            -drag_coeff,
            self.cy_beta * beta,
            self.cl_alpha * alpha,
        ])

        # Rotate the body-axis force back into ECEF/world coordinates for translation.
        return orientation @ body_force
    
    def _roll_moment( self, up, right ):
        """Applies force in up direction at position 1.0*right
        """
        magnitude = (self.roll_cmd / 100.0) * self.max_roll_moment
        return magnitude * np.outer( up, right )
    
    def _pitch_moment( self, up, forward ):
        """Applies force in up direction at position 1.0*forward
        """
        magnitude = (self.pitch_cmd / 100.0) * self.max_pitch_moment
        return magnitude * np.outer( up, forward )
    
    def _yaw_moment( self, right, forward ):
        """Applies force in right direction at position 1.0*forward
        """
        magnitude = (self.yaw_cmd / 100.0) * self.max_yaw_moment
        return magnitude * np.outer( right, forward )

    def _restoring_moment_components(self, pos, vel, orientation, omega_body):
        # The restoring model is purely aerodynamic: it vanishes at low airspeed and uses
        # alpha, beta, and bank error plus body-rate damping to oppose misalignment.
        speed, alpha, beta, bank_error = self._aerodynamic_angles(pos, vel, orientation)
        if speed < 1e-6:
            return np.zeros(3)

        return np.array([
            -self.bank_restoring_coeff * bank_error - self.roll_damping_coeff * omega_body[0],
            -self.alpha_restoring_coeff * alpha + self.pitch_damping_coeff * omega_body[1],
            self.beta_restoring_coeff * beta - self.yaw_damping_coeff * omega_body[2],
        ])

    def _restoring_moment(self, pos, vel, orientation, omega_body):
        # Reuse the same body-axis actuation geometry as the commanded roll/pitch/yaw
        # moments so passive stability and control inputs combine in one convention.
        forward = orientation[:, 0]
        right = orientation[:, 1]
        up = orientation[:, 2]
        roll_mag, pitch_mag, yaw_mag = self._restoring_moment_components(pos, vel, orientation, omega_body)
        return (
            roll_mag * np.outer(up, right)
            + pitch_mag * np.outer(up, forward)
            + yaw_mag * np.outer(right, forward)
        )
    

    def compute_state_derivative(self, t, state):
        # unpack current state
        pos = state[self.get_position_slice()]
        vel = state[self.get_velocity_slice()]
        orientation = state[self.get_orientation_slice()].reshape((3,3))
        orientation = project_to_rotation_matrix(orientation)
        omega_body = state[self.get_omega_slice()]
        omega = vector_to_skew_symmetric( omega_body )

        forward = orientation[:,0]
        right = orientation[:,1]
        up = orientation[:,2]

        # trivial derivatives
        dpos = vel
        dorientation = orientation @ omega
        if self.use_coriolis:
            # The attitude state evolves in ECEF, which is itself a rotating frame.
            # Subtract the frame rotation so `dorientation` reflects body motion
            # relative to Earth-fixed coordinates rather than inertial-space spin.
            coriolis = vector_to_skew_symmetric( coriolis_vector() )
            dorientation -= coriolis

        # Body acceleration
        body_force = self._thrust_vector( forward )
        body_force += self._aerodynamic_force(pos, vel, orientation)

        accel = gravity(pos)
        if self.use_coriolis:
            accel += centrifugal_acceleration( pos )
            accel += coriolis_acceleration( vel )

        dvel = accel + body_force / self.mass

        # Rotational acceleration
        body_moment = self._roll_moment( up, right )
        body_moment += self._pitch_moment( up, forward )
        body_moment += self._yaw_moment( right, forward )
        body_moment += self._restoring_moment(pos, vel, orientation, omega_body)

        domega = orientation.T @ body_moment
        domega = np.linalg.solve( self.rotational_mass.T, domega.T ).T
        domega = 0.5 * ( domega - domega.T )
        domega = skew_symmetric_to_vector( domega )

        return np.concatenate([dpos, dvel, dorientation.reshape(-1), domega])


class FixedWingAutopilot(Controller):
    def __init__(
        self,
        waypoints,
        target_speed=150.0,
        target_speeds=None,
        max_climb_rate=4.0,
        waypoint_radius=500.0,
        update_interval=0.1,
    ):
        """Create a waypoint-following autopilot for `FixedWingMover`.

        Parameters:
            waypoints: Sequence of ECEF waypoint positions with shape `(3,)`. This may
                be empty to represent a route with no active waypoint targets.
            target_speed: Default target speed in m/s used while flying toward every
                waypoint unless
                overridden by `target_speeds`.
            target_speeds: Optional per-waypoint speed overrides in m/s. If provided,
                its length must be `len(waypoints)`.
            max_climb_rate: Maximum commanded climb or descent rate in m/s.
            waypoint_radius: Distance in meters used to declare a waypoint reached.
            update_interval: Controller execution period in seconds.
        """
        super().__init__(update_interval=update_interval)

        if waypoint_radius <= 0.0:
            raise ValueError("waypoint_radius must be positive")
        if target_speed < 0.0:
            raise ValueError("target_speed must be non-negative")
        if max_climb_rate < 0.0:
            raise ValueError("max_climb_rate must be non-negative")

        self.waypoints = [np.asarray(wp, dtype=float) for wp in waypoints]
        if any(wp.shape != (3,) for wp in self.waypoints):
            raise ValueError("each waypoint must have shape (3,)")

        self.target_speed = float(target_speed)
        self.max_climb_rate = float(max_climb_rate)
        self.waypoint_radius = float(waypoint_radius)
        self.current_wp_idx = 0
        self.k_heading = 100.0 / np.radians(60.0)
        self.k_altitude = 0.05
        self.k_climb_rate = 12.0
        self.k_speed = 1.5
        self.hold_active = False
        self.hold_speed = None
        self.hold_altitude = None
        self.hold_horizontal_direction = None

        waypoint_count = len(self.waypoints)
        self.completed = waypoint_count == 0
        if target_speeds is None:
            self.target_speeds = np.full(waypoint_count, self.target_speed, dtype=float)
        else:
            self.target_speeds = np.asarray(target_speeds, dtype=float)
            if self.target_speeds.shape != (waypoint_count,):
                raise ValueError(f"target_speeds must have shape ({waypoint_count},)")
            if np.any(self.target_speeds < 0.0):
                raise ValueError("target_speeds must be non-negative")

    def _flight_condition(self, mover):
        pos = mover.position
        vel = mover.velocity
        speed = np.linalg.norm(vel)
        local_up = pos / max(np.linalg.norm(pos), 1e-6)
        alt = ecef_to_lla(pos[0], pos[1], pos[2])[2]
        return pos, vel, speed, local_up, alt

    def _heading_error_to_direction(self, mover, target_horizontal_direction, pos=None, vel=None, local_up=None):
        # Returns the signed horizontal heading error in radians from the current track
        # to `target_horizontal_direction`; positive means turn left, negative turn right.
        if target_horizontal_direction is None:
            return 0.0

        if pos is None or vel is None or local_up is None:
            pos, vel, _, local_up, _ = self._flight_condition(mover)

        target_horizontal_direction = np.asarray(target_horizontal_direction, dtype=float)
        target_norm = np.linalg.norm(target_horizontal_direction)
        if target_norm <= 1e-6:
            return 0.0
        target_horizontal_direction = target_horizontal_direction / target_norm

        current_horizontal = vel - np.dot(vel, local_up) * local_up
        current_horizontal_norm = np.linalg.norm(current_horizontal)
        if current_horizontal_norm > 1e-6:
            current_horizontal = current_horizontal / current_horizontal_norm
        else:
            # When the horizontal velocity nearly vanishes, use the projected body-forward
            # direction so guidance still has a meaningful heading reference.
            current_horizontal = mover.orientation[:, 0] - np.dot(mover.orientation[:, 0], local_up) * local_up
            current_horizontal_norm = np.linalg.norm(current_horizontal)
            if current_horizontal_norm <= 1e-6:
                return 0.0
            current_horizontal = current_horizontal / current_horizontal_norm

        return np.arctan2(
            np.dot(np.cross(current_horizontal, target_horizontal_direction), local_up),
            np.clip(np.dot(current_horizontal, target_horizontal_direction), -1.0, 1.0),
        )

    def _apply_guidance(self, mover, heading_error, vertical_error, target_speed, vel=None, speed=None, local_up=None, alt=None):
        if vel is None or speed is None or local_up is None or alt is None:
            _, vel, speed, local_up, alt = self._flight_condition(mover)

        desired_climb_rate = np.clip(
            self.k_altitude * vertical_error,
            -self.max_climb_rate,
            self.max_climb_rate,
        )
        actual_climb_rate = np.dot(vel, local_up)
        climb_rate_error = desired_climb_rate - actual_climb_rate

        drag_force_mag = np.linalg.norm(aerodynamic_drag_force(vel, alt, mover.cd0, mover.area))
        thrust_trim = 100.0 * drag_force_mag / max(mover.max_thrust, 1e-6)

        mover.roll_cmd = np.clip(self.k_heading * heading_error, -100.0, 100.0)
        mover.pitch_cmd = np.clip(self.k_climb_rate * climb_rate_error, -100.0, 100.0)
        mover.thrust_cmd = np.clip(thrust_trim + self.k_speed * (target_speed - speed), 0.0, 100.0)
        mover.yaw_cmd = 0.0

    def _enter_hold_mode(self, mover):
        if self.hold_active:
            return

        pos, vel, _, local_up, alt = self._flight_condition(mover)

        # Prefer the current horizontal flight direction so terminal hold continues from
        # the aircraft's actual track rather than snapping to its body heading.
        velocity_horizontal = vel - np.dot(vel, local_up) * local_up
        velocity_horizontal_norm = np.linalg.norm(velocity_horizontal)
        if velocity_horizontal_norm > 1e-6:
            hold_horizontal_direction = velocity_horizontal / velocity_horizontal_norm
        else:
            # If the aircraft is moving nearly straight up/down, fall back to the body
            # forward axis projected into the local horizontal plane.
            forward_horizontal = mover.orientation[:, 0] - np.dot(mover.orientation[:, 0], local_up) * local_up
            forward_horizontal_norm = np.linalg.norm(forward_horizontal)
            if forward_horizontal_norm > 1e-6:
                hold_horizontal_direction = forward_horizontal / forward_horizontal_norm
            else:
                # Final fallback for degenerate geometry: pick any stable local-horizontal
                # direction so hold mode still has a valid reference vector.
                fallback = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(fallback, local_up)) > 0.95:
                    fallback = np.array([0.0, 1.0, 0.0])
                hold_horizontal_direction = fallback - np.dot(fallback, local_up) * local_up
                hold_horizontal_direction /= max(np.linalg.norm(hold_horizontal_direction), 1e-6)

        # Hold speed comes from the last route target if one exists; otherwise the
        # default cruise target is used for empty-route hold mode.
        self.hold_active = True
        self.hold_speed = self.target_speeds[-1] if len(self.target_speeds) > 0 else self.target_speed
        self.hold_altitude = alt
        self.hold_horizontal_direction = hold_horizontal_direction

    def _update_hold_mode(self, mover):
        _, vel, speed, local_up, current_altitude = self._flight_condition(mover)
        heading_error = self._heading_error_to_direction(
            mover,
            self.hold_horizontal_direction,
        )
        target_altitude = current_altitude if self.hold_altitude is None else self.hold_altitude
        vertical_error = target_altitude - current_altitude
        target_speed = speed if self.hold_speed is None else self.hold_speed
        self._apply_guidance(
            mover,
            heading_error,
            vertical_error,
            target_speed,
            vel=vel,
            speed=speed,
            local_up=local_up,
            alt=current_altitude,
        )

    def _update_waypoint_guidance(self, mover):
        pos, vel, speed, local_up, alt = self._flight_condition(mover)
        wp_target = self.waypoints[self.current_wp_idx]
        target_speed = self.target_speeds[self.current_wp_idx]

        # Steer by comparing the current horizontal flight direction to the horizontal
        # direction from the aircraft to the active waypoint.
        rel_pos = wp_target - pos
        rel_horizontal = rel_pos - np.dot(rel_pos, local_up) * local_up
        rel_horizontal_norm = np.linalg.norm(rel_horizontal)
        if rel_horizontal_norm > 1e-6:
            desired_horizontal = rel_horizontal / rel_horizontal_norm
            heading_error = self._heading_error_to_direction(
                mover,
                desired_horizontal,
                pos=pos,
                vel=vel,
                local_up=local_up,
            )
        else:
            heading_error = 0.0

        # Convert altitude error into a bounded climb-rate target, then use pitch to
        # drive the actual climb rate toward that target.
        vertical_error = np.dot(rel_pos, local_up)
        self._apply_guidance(
            mover,
            heading_error,
            vertical_error,
            target_speed,
            vel=vel,
            speed=speed,
            local_up=local_up,
            alt=alt,
        )

    def update( self, t, engine ):
        """Adjusts thrust, roll, pitch, yaw commands to remain on course.
        """
        mover = self.platform.mover
        if not isinstance(mover, FixedWingMover):
            return

        # With no active waypoint, capture terminal hold targets once and keep flying
        # straight and level at the stored speed/altitude/heading targets.
        if self.current_wp_idx >= len(self.waypoints):
            self.completed = True
            if not self.hold_active:
                self._enter_hold_mode(mover)
            self._update_hold_mode(mover)
            return

        # Any live waypoint target takes precedence over hold mode.
        self.completed = False
        self.hold_active = False

        pos = mover.position
        # Consume any waypoint whose capture sphere already contains the aircraft. This
        # allows immediate progression through stacked or already-reached waypoints.
        while self.current_wp_idx < len(self.waypoints):
            if np.linalg.norm(self.waypoints[self.current_wp_idx] - pos) >= self.waypoint_radius:
                break
            reached_wp_idx = self.current_wp_idx
            self.current_wp_idx += 1
            engine.broker.publish("waypoint_reached", self.platform, reached_wp_idx)

        # Guidance-command logic only runs while there is still an active target waypoint.
        if self.current_wp_idx >= len(self.waypoints):
            self.completed = True
            if not self.hold_active:
                self._enter_hold_mode(mover)
            self._update_hold_mode(mover)
            return

        self._update_waypoint_guidance(mover)

        return




class RocketMover(TranslationalMover, IntegratedMover):
    """Rigid-body rocket mover with translational, attitude, and body-rate state."""

    REQUIRED_STAGE_FIELDS = (
        "dry_mass",
        "propellant_mass",
        "reference_area",
        "drag_coefficient",
        "rotational_mass",
        "angular_damping",
        "max_thrust",
        "max_steering_moment",
    )

    class OrientationCorrectionEvent(Event):
        """Periodically projects a mover's committed orientation back onto `SO(3)`."""

        def __init__(self, mover, interval=1.0):
            self.mover = mover
            mover_name = mover.platform.id if mover.platform is not None else str(id(mover))
            super().__init__(
                0.0,
                self.fix_orientation,
                name=f"RocketOrientationCorrection_{mover_name}",
                interval=interval,
            )

        def fix_orientation(self, engine):
            if self.mover._context is not engine.context:
                return
            if self.mover not in engine.context._index_map:
                return

            mover_slice = engine.context.get_state_slice(self.mover)
            orientation_slice = self.mover.get_orientation_slice()
            start = mover_slice.start + orientation_slice.start
            stop = mover_slice.start + orientation_slice.stop

            orientation = engine.context.committed_y[start:stop].reshape((3, 3))
            corrected = project_to_rotation_matrix(orientation)
            engine.context.committed_y[start:stop] = corrected.reshape(-1)

    def __init__(
        self,
        initial_position,
        initial_velocity,
        initial_orientation=None,
        initial_body_rates=None,
        stages=None,
        mass=10000.0,
        propellant_mass=0.0,
        rotational_mass=None,
        area=30.0,
        cd0=0.02,
        normal_force_coefficient=2.5,
        alignment_restoring_coefficient=0.0,
        max_thrust=80000.0,
        max_steering_moment=5.0e4,
        angular_damping=None,
        use_coriolis=True,
    ):
        """
        Parameters:
            initial_position: ECEF coordinates [X, Y, Z] in meters.
            initial_velocity: ECEF velocity [Vx, Vy, Vz] in m/s.
            initial_orientation: Optional mover-frame forward, right, up basis matrix in ECEF. If omitted,
                an orientation is derived from the initial velocity and local vertical.
            initial_body_rates: Optional body angular velocity vector in rad/s.
            stages: Optional sequence of stage dictionaries. The active stage contributes
                the integrated propellant state while attached upper stages contribute
                fixed attached mass until separated.
            mass: Dry or non-propellant mass carried by the rocket in kg.
            propellant_mass: Active propellant mass in kg stored in the state vector.
            rotational_mass: Body rotational-mass tensor as a `(3, 3)` tensor or `(3,)` principal moments.
            area: Reference area in m^2 used for drag.
            cd0: Zero-lift drag coefficient.
            normal_force_coefficient: Coefficient mapping axial-flow misalignment into
                a transverse aerodynamic force.
            alignment_restoring_coefficient: Coefficient scaling a passive aerodynamic
                moment that rotates the rocket longitudinal axis toward the airflow.
            max_thrust: Maximum thrust in Newtons.
            max_steering_moment: Maximum steering moment magnitude in N*m.
            angular_damping: Per-axis angular damping coefficients.
            use_coriolis: If True, include Coriolis acceleration in world-frame translation.

        State layout:
            [x, y, z, vx, vy, vz, o11, ..., o33, w1, w2, w3, m_prop]
        """

        initial_position = np.asarray(initial_position, dtype=float)
        if initial_position.shape != (3,):
            raise ValueError("initial_position must have shape (3,)")

        initial_velocity = np.asarray(initial_velocity, dtype=float)
        if initial_velocity.shape != (3,):
            raise ValueError("initial_velocity must have shape (3,)")

        initial_body_rates = (
            np.asarray(initial_body_rates, dtype=float)
            if initial_body_rates is not None
            else np.zeros(3)
        )
        if initial_body_rates.shape != (3,):
            raise ValueError("initial_body_rates must have shape (3,)")

        if initial_orientation is None:
            initial_orientation = self._derive_orientation_from_velocity(
                initial_position,
                initial_velocity,
            )
        else:
            initial_orientation = np.asarray(initial_orientation, dtype=float)
            if initial_orientation.shape != (3, 3):
                raise ValueError("initial_orientation must have shape (3, 3)")
            initial_orientation = project_to_rotation_matrix(initial_orientation)

        validated_stages = self._validate_stage_definitions(stages)
        if validated_stages:
            initial_propellant_mass = float(validated_stages[0]["propellant_mass"])
        else:
            initial_propellant_mass = float(propellant_mass)
            if initial_propellant_mass < 0.0:
                raise ValueError("propellant_mass must be non-negative")

        state = np.concatenate([
            initial_position,
            initial_velocity,
            initial_orientation.reshape(-1),
            initial_body_rates,
            np.array([initial_propellant_mass]),
        ])
        super().__init__(state)

        self.payload_mass = float(mass)
        if self.payload_mass < 0.0:
            raise ValueError("mass must be non-negative")

        self.base_rotational_mass = self._coerce_rotational_mass(rotational_mass)
        self.base_angular_damping = self._coerce_angular_damping(angular_damping)
        self.base_area = float(area)
        self.base_cd0 = float(cd0)
        self.base_normal_force_coefficient = self._validate_nonnegative_scalar(
            normal_force_coefficient,
            "normal_force_coefficient",
        )
        self.base_alignment_restoring_coefficient = self._validate_nonnegative_scalar(
            alignment_restoring_coefficient,
            "alignment_restoring_coefficient",
        )
        self.base_max_thrust = float(max_thrust)
        self.base_max_steering_moment = float(max_steering_moment)

        self.stages = validated_stages
        self.active_stage_index = 0
        self.attached_mass_excluding_active_propellant = self._compute_attached_mass_excluding_active_propellant(
            self.active_stage_index,
        )

        self.rotational_mass = self.base_rotational_mass.copy()
        self.inv_rotational_mass = np.linalg.inv(self.rotational_mass)
        self.area = self.base_area
        self.cd0 = self.base_cd0
        self.normal_force_coefficient = self.base_normal_force_coefficient
        self.alignment_restoring_coefficient = self.base_alignment_restoring_coefficient
        self.max_thrust = self.base_max_thrust
        self.max_steering_moment = self.base_max_steering_moment
        self.angular_damping = self.base_angular_damping.copy()
        self._apply_active_stage_properties()
        self.dry_mass = self.attached_mass_excluding_active_propellant
        self.mass = self.current_total_mass(initial_propellant_mass)
        self.use_coriolis = bool(use_coriolis)

        self.thrust_cmd = 0.0
        self.steer_cmd = 0.0
        self.steer_direction_body = np.array([0.0, 1.0, 0.0])

    def add_orientation_correction_event(self, engine, interval=1.0):
        """Schedule a recurring event that projects the committed orientation onto `SO(3)`."""
        event = self.OrientationCorrectionEvent(self, interval=interval)
        engine.schedule(engine.t, event.callback, name=event.name, interval=event.interval)
        return event

    def _derive_orientation_from_velocity(self, position, velocity):
        speed = np.linalg.norm(velocity)
        if speed < 1e-8:
            return np.eye(3)

        pos_norm = np.linalg.norm(position)
        local_vertical = position / pos_norm if pos_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        forward, right, up = build_aircraft_body_axes(velocity, local_vertical)
        return project_to_rotation_matrix(np.column_stack([forward, right, up]))

    def _coerce_rotational_mass(self, rotational_mass):
        if rotational_mass is None:
            rotational_mass = np.diag([6.0e4, 1.2e5, 1.2e5])
        rotational_mass = np.asarray(rotational_mass, dtype=float)
        if rotational_mass.shape == (3,):
            rotational_mass = np.diag(rotational_mass)
        if rotational_mass.shape != (3, 3):
            raise ValueError("rotational_mass must have shape (3,) or (3, 3)")
        return rotational_mass

    def _coerce_angular_damping(self, angular_damping):
        if angular_damping is None:
            return np.array([2.0e4, 4.0e4, 4.0e4])
        angular_damping = np.asarray(angular_damping, dtype=float)
        if angular_damping.shape != (3,):
            raise ValueError("angular_damping must have shape (3,)")
        return angular_damping

    def _validate_nonnegative_scalar(self, value, name):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _validate_positive_scalar(self, value, name):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than 0")
        return value

    def _validate_stage_definitions(self, stages):
        if stages is None:
            return []
        if not isinstance(stages, (list, tuple)):
            raise ValueError("stages must be a list or tuple of stage definitions")

        validated_stages = []
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise ValueError(f"stages[{index}] must be a dictionary")

            validated_stage = dict(stage)
            for field in self.REQUIRED_STAGE_FIELDS:
                if field not in validated_stage:
                    raise ValueError(f"stages[{index}] is missing required field '{field}'")

            validated_stage["dry_mass"] = self._validate_nonnegative_scalar(
                validated_stage["dry_mass"],
                f"stages[{index}].dry_mass",
            )
            validated_stage["propellant_mass"] = self._validate_nonnegative_scalar(
                validated_stage["propellant_mass"],
                f"stages[{index}].propellant_mass",
            )
            validated_stage["reference_area"] = self._validate_positive_scalar(
                validated_stage["reference_area"],
                f"stages[{index}].reference_area",
            )
            validated_stage["drag_coefficient"] = self._validate_nonnegative_scalar(
                validated_stage["drag_coefficient"],
                f"stages[{index}].drag_coefficient",
            )
            if "normal_force_coefficient" in validated_stage:
                validated_stage["normal_force_coefficient"] = self._validate_nonnegative_scalar(
                    validated_stage["normal_force_coefficient"],
                    f"stages[{index}].normal_force_coefficient",
                )
            if "alignment_restoring_coefficient" in validated_stage:
                validated_stage["alignment_restoring_coefficient"] = self._validate_nonnegative_scalar(
                    validated_stage["alignment_restoring_coefficient"],
                    f"stages[{index}].alignment_restoring_coefficient",
                )
            validated_stage["rotational_mass"] = self._coerce_rotational_mass(
                validated_stage["rotational_mass"],
            )
            validated_stage["angular_damping"] = self._coerce_angular_damping(
                validated_stage["angular_damping"],
            )
            validated_stage["max_thrust"] = self._validate_nonnegative_scalar(
                validated_stage["max_thrust"],
                f"stages[{index}].max_thrust",
            )
            validated_stage["max_steering_moment"] = self._validate_nonnegative_scalar(
                validated_stage["max_steering_moment"],
                f"stages[{index}].max_steering_moment",
            )

            has_burn_duration = "burn_duration" in validated_stage
            has_mass_flow_rate = "mass_flow_rate" in validated_stage
            if not has_burn_duration and not has_mass_flow_rate:
                raise ValueError(
                    f"stages[{index}] must define either 'burn_duration' or 'mass_flow_rate'"
                )
            if has_burn_duration:
                validated_stage["burn_duration"] = self._validate_positive_scalar(
                    validated_stage["burn_duration"],
                    f"stages[{index}].burn_duration",
                )
            if has_mass_flow_rate:
                validated_stage["mass_flow_rate"] = self._validate_positive_scalar(
                    validated_stage["mass_flow_rate"],
                    f"stages[{index}].mass_flow_rate",
                )

            if "thrust" not in validated_stage:
                raise ValueError(f"stages[{index}] must define 'thrust'")
            validated_stage["thrust"] = self._validate_nonnegative_scalar(
                validated_stage["thrust"],
                f"stages[{index}].thrust",
            )

            validated_stages.append(validated_stage)

        return validated_stages

    def _compute_attached_mass_excluding_active_propellant(self, active_stage_index):
        attached_mass = self.payload_mass
        if not self.stages:
            return attached_mass

        if active_stage_index < 0 or active_stage_index >= len(self.stages):
            raise ValueError("active_stage_index is out of range")

        current_stage = self.stages[active_stage_index]
        attached_mass += current_stage["dry_mass"]
        for future_stage in self.stages[active_stage_index + 1:]:
            attached_mass += future_stage["dry_mass"] + future_stage["propellant_mass"]
        return attached_mass

    def _apply_active_stage_properties(self):
        stage = self._active_stage()
        if stage is None:
            self.rotational_mass = self.base_rotational_mass.copy()
            self.angular_damping = self.base_angular_damping.copy()
            self.area = self.base_area
            self.cd0 = self.base_cd0
            self.normal_force_coefficient = self.base_normal_force_coefficient
            self.alignment_restoring_coefficient = self.base_alignment_restoring_coefficient
            self.max_thrust = self.base_max_thrust
            self.max_steering_moment = self.base_max_steering_moment
        else:
            self.rotational_mass = stage["rotational_mass"].copy()
            self.angular_damping = stage["angular_damping"].copy()
            self.area = float(stage["reference_area"])
            self.cd0 = float(stage["drag_coefficient"])
            self.normal_force_coefficient = float(
                stage.get("normal_force_coefficient", self.base_normal_force_coefficient)
            )
            self.alignment_restoring_coefficient = float(
                stage.get("alignment_restoring_coefficient", self.base_alignment_restoring_coefficient)
            )
            self.max_thrust = float(stage["max_thrust"])
            self.max_steering_moment = float(stage["max_steering_moment"])

        self.inv_rotational_mass = np.linalg.inv(self.rotational_mass)

    def _active_stage(self):
        if not self.stages or self.active_stage_index >= len(self.stages):
            return None
        return self.stages[self.active_stage_index]

    def _set_propellant_mass_state(self, propellant_mass):
        propellant_mass = self._validate_nonnegative_scalar(propellant_mass, "propellant_mass")
        propellant_mass_slice = self.get_propellant_mass_slice()
        self._initial_state[propellant_mass_slice] = propellant_mass

        self.mass = self.current_total_mass(propellant_mass)

    def _set_propellant_mass_state_in_engine(self, propellant_mass, engine):
        propellant_mass = self._validate_nonnegative_scalar(propellant_mass, "propellant_mass")
        self._set_propellant_mass_state(propellant_mass)

        if engine is None:
            raise ValueError("engine is required to rewrite committed propellant-mass state")

        context = engine.context
        if self not in context._index_map:
            raise ValueError("rocket mover must be registered with the supplied engine")

        propellant_mass_slice = self.get_propellant_mass_slice()
        state_slice = context.get_state_slice(self)
        committed_state = context.committed_y[state_slice].copy()
        committed_state[propellant_mass_slice] = propellant_mass
        context.committed_y[state_slice] = committed_state

        if context._integrating and context._integration_y is not None:
            integration_state = context._integration_y[state_slice].copy()
            integration_state[propellant_mass_slice] = propellant_mass
            context._integration_y[state_slice] = integration_state

    def _refresh_stage_configuration(self, propellant_mass):
        self.attached_mass_excluding_active_propellant = self._compute_attached_mass_excluding_active_propellant(
            self.active_stage_index,
        )
        self._apply_active_stage_properties()
        self.dry_mass = self.attached_mass_excluding_active_propellant
        self.mass = self.current_total_mass(propellant_mass)

    def _coerce_percent_command(self, value, name):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return np.clip(value, 0.0, 100.0)

    def _coerce_body_direction(self, direction, name):
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        if not np.all(np.isfinite(direction)):
            raise ValueError(f"{name} must contain only finite values")

        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError(f"{name} must have non-zero magnitude")
        return direction / norm

    def get_orientation_slice(self):
        return slice(6, 15)

    @property
    def orientation(self):
        return self.get_state()[self.get_orientation_slice()].reshape((3, 3))

    @property
    def forward_axis(self):
        return self.orientation[:, 0]

    @property
    def right_axis(self):
        return self.orientation[:, 1]

    @property
    def up_axis(self):
        return self.orientation[:, 2]

    def get_omega_slice(self):
        return slice(15, 18)

    @property
    def omega_body(self):
        return self.get_state()[self.get_omega_slice()]

    def get_propellant_mass_slice(self):
        return slice(18, 19)

    @property
    def propellant_mass(self):
        return float(self.get_state()[self.get_propellant_mass_slice()][0])

    @property
    def thrust_cmd(self):
        return self._thrust_cmd

    @thrust_cmd.setter
    def thrust_cmd(self, value):
        self._thrust_cmd = self._coerce_percent_command(value, "thrust_cmd")

    @property
    def steer_cmd(self):
        return self._steer_cmd

    @steer_cmd.setter
    def steer_cmd(self, value):
        self._steer_cmd = self._coerce_percent_command(value, "steer_cmd")

    @property
    def steer_direction_body(self):
        return self._steer_direction_body.copy()

    @steer_direction_body.setter
    def steer_direction_body(self, value):
        self._steer_direction_body = self._coerce_body_direction(value, "steer_direction_body")

    def steering_direction_perpendicular_to(self, forward_axis):
        """Return the commanded steering direction projected into the plane normal to `forward_axis`."""
        forward_axis = np.asarray(forward_axis, dtype=float)
        if forward_axis.shape != (3,):
            raise ValueError("forward_axis must have shape (3,)")

        forward_norm = np.linalg.norm(forward_axis)
        if forward_norm < 1e-12:
            raise ValueError("forward_axis must have non-zero magnitude")
        forward_axis = forward_axis / forward_norm

        direction = self._steer_direction_body - np.dot(self._steer_direction_body, forward_axis) * forward_axis
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-12:
            fallback = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(fallback, forward_axis)) > 0.95:
                fallback = np.array([0.0, 0.0, 1.0])
            direction = fallback - np.dot(fallback, forward_axis) * forward_axis
            direction_norm = np.linalg.norm(direction)

        return direction / max(direction_norm, 1e-12)

    def current_total_mass(self, propellant_mass=None):
        if propellant_mass is None:
            propellant_mass = self.propellant_mass
        propellant_mass = self._validate_nonnegative_scalar(propellant_mass, "propellant_mass")
        return self.attached_mass_excluding_active_propellant + propellant_mass

    def current_mass_flow_rate(self, t, propellant_mass=None):
        del t
        if propellant_mass is None:
            propellant_mass = self.propellant_mass
        propellant_mass = self._validate_nonnegative_scalar(propellant_mass, "propellant_mass")
        if propellant_mass <= 0.0:
            return 0.0

        throttle_fraction = self.thrust_cmd / 100.0
        if throttle_fraction <= 0.0:
            return 0.0

        stage = self._active_stage()
        if stage is None:
            return 0.0
        if "mass_flow_rate" in stage:
            return throttle_fraction * float(stage["mass_flow_rate"])
        return throttle_fraction * stage["propellant_mass"] / stage["burn_duration"]

    def current_stage_thrust(self, t, propellant_mass=None):
        del t
        if propellant_mass is None:
            propellant_mass = self.propellant_mass
        propellant_mass = self._validate_nonnegative_scalar(propellant_mass, "propellant_mass")
        if propellant_mass <= 0.0:
            return 0.0

        throttle_fraction = self.thrust_cmd / 100.0
        if throttle_fraction <= 0.0:
            return 0.0

        stage = self._active_stage()
        if stage is None:
            return throttle_fraction * self.max_thrust
        return throttle_fraction * float(stage["thrust"])

    def has_active_burn(self, propellant_mass=None):
        if propellant_mass is None:
            propellant_mass = self.propellant_mass
        propellant_mass = self._validate_nonnegative_scalar(propellant_mass, "propellant_mass")
        if propellant_mass <= 0.0:
            return False
        return bool(self.current_mass_flow_rate(0.0, propellant_mass=propellant_mass) > 0.0)

    def _air_data(self, pos, vel):
        speed = np.linalg.norm(vel)
        _, _, alt = ecef_to_lla(pos[0], pos[1], pos[2])
        rho = air_density(alt)
        dynamic_pressure = 0.5 * rho * (speed ** 2)
        return speed, alt, rho, dynamic_pressure

    def _thrust_force_world(self, orientation, t, propellant_mass=None):
        forward_axis = orientation[:, 0]
        thrust_magnitude = self.current_stage_thrust(t, propellant_mass=propellant_mass)
        return thrust_magnitude * forward_axis

    def _drag_force_world(self, pos, vel):
        _, alt, _, _ = self._air_data(pos, vel)
        return aerodynamic_drag_force(vel, alt, self.cd0, self.area)

    def _normal_aero_force_world(self, pos, vel, orientation):
        speed, _, _, dynamic_pressure = self._air_data(pos, vel)
        if speed < 1e-6 or self.area <= 0.0 or self.normal_force_coefficient <= 0.0:
            return np.zeros(3)

        forward_axis = orientation[:, 0]
        axial_velocity = np.dot(vel, forward_axis) * forward_axis
        transverse_velocity = vel - axial_velocity
        transverse_speed = np.linalg.norm(transverse_velocity)
        if transverse_speed < 1e-6:
            return np.zeros(3)

        misalignment = transverse_speed / speed
        force_direction = -transverse_velocity / transverse_speed
        force_magnitude = dynamic_pressure * self.area * self.normal_force_coefficient * misalignment
        return force_magnitude * force_direction

    def _steering_moment_body(self):
        body_forward_axis = np.array([1.0, 0.0, 0.0])
        steering_direction = self.steering_direction_perpendicular_to(body_forward_axis)
        steering_magnitude = (self.steer_cmd / 100.0) * self.max_steering_moment
        return steering_magnitude * steering_direction

    def _alignment_restoring_moment_body(self, pos, vel, orientation):
        if self.alignment_restoring_coefficient <= 0.0 or self.area <= 0.0:
            return np.zeros(3)

        speed, _, _, dynamic_pressure = self._air_data(pos, vel)
        if speed < 1e-6:
            return np.zeros(3)

        velocity_body = orientation.T @ vel
        velocity_hat_body = velocity_body / speed
        body_forward_axis = np.array([1.0, 0.0, 0.0])
        alignment_axis = np.cross(body_forward_axis, velocity_hat_body)
        alignment_axis[0] = 0.0
        alignment_axis_norm = np.linalg.norm(alignment_axis)
        if alignment_axis_norm < 1e-6:
            return np.zeros(3)

        restoring_magnitude = (
            dynamic_pressure * self.area * self.alignment_restoring_coefficient * alignment_axis_norm
        )
        return restoring_magnitude * (alignment_axis / alignment_axis_norm)

    def _angular_damping_moment_body(self, omega_body):
        omega_body = np.asarray(omega_body, dtype=float)
        if omega_body.shape != (3,):
            raise ValueError("omega_body must have shape (3,)")
        return -self.angular_damping * omega_body

    def _body_moment_body(self, pos, vel, orientation, omega_body):
        return (
            self._steering_moment_body()
            + self._alignment_restoring_moment_body(pos, vel, orientation)
            + self._angular_damping_moment_body(omega_body)
        )

    def _angular_acceleration_body(self, pos, vel, orientation, omega_body):
        omega_body = np.asarray(omega_body, dtype=float)
        if omega_body.shape != (3,):
            raise ValueError("omega_body must have shape (3,)")

        body_moment = self._body_moment_body(pos, vel, orientation, omega_body)
        angular_momentum = self.rotational_mass @ omega_body
        return self.inv_rotational_mass @ (body_moment - np.cross(omega_body, angular_momentum))

    def can_separate_stage(self, propellant_mass=None):
        if propellant_mass is None:
            propellant_mass = self.propellant_mass
        propellant_mass = self._validate_nonnegative_scalar(propellant_mass, "propellant_mass")
        if not self.stages:
            return False
        if self.active_stage_index + 1 >= len(self.stages):
            return False
        return propellant_mass <= 0.0

    def separate_stage(self, engine):
        current_propellant_mass = self.propellant_mass
        if not self.can_separate_stage(current_propellant_mass):
            raise RuntimeError("current stage cannot be separated")

        self.active_stage_index += 1
        next_stage = self._active_stage()
        next_propellant_mass = float(next_stage["propellant_mass"])
        self._refresh_stage_configuration(next_propellant_mass)
        self._set_propellant_mass_state_in_engine(next_propellant_mass, engine)
        return next_stage

    def advance_to_next_stage(self, engine):
        return self.separate_stage(engine)

    def compute_state_derivative(self, t, state):
        pos = state[self.get_position_slice()]
        vel = state[self.get_velocity_slice()]
        orientation = state[self.get_orientation_slice()].reshape((3, 3))
        orientation = project_to_rotation_matrix(orientation)
        omega_body = state[self.get_omega_slice()]
        propellant_mass = max(float(state[self.get_propellant_mass_slice()][0]), 0.0)
        total_mass = max(self.current_total_mass(propellant_mass), 1e-9)
        thrust_force = self._thrust_force_world(orientation, t, propellant_mass=propellant_mass)
        drag_force = self._drag_force_world(pos, vel)
        normal_force = self._normal_aero_force_world(pos, vel, orientation)

        dpos = vel
        dorientation = orientation @ vector_to_skew_symmetric(omega_body)
        if self.use_coriolis:
            frame_rotation = vector_to_skew_symmetric(coriolis_vector())
            dorientation -= frame_rotation

        accel = gravity(pos)
        if self.use_coriolis:
            accel += centrifugal_acceleration(pos)
            accel += coriolis_acceleration(vel)

        dvel = accel + (thrust_force + drag_force + normal_force) / total_mass
        domega = self._angular_acceleration_body(pos, vel, orientation, omega_body)
        dpropellant_mass = np.array([
            -self.current_mass_flow_rate(t, propellant_mass=propellant_mass),
        ])

        return np.concatenate([dpos, dvel, dorientation.reshape(-1), domega, dpropellant_mass])


class ExpendedStageMover(RocketMover):
    """Passive separated stage modeled with the rigid-body rocket aerodynamics."""

    DEFAULT_MASS = 2000.0
    DEFAULT_AREA = 2.5
    DEFAULT_CD0 = 0.15
    DEFAULT_ROTATIONAL_MASS = np.diag([3000.0, 18000.0, 18000.0])
    DEFAULT_ANGULAR_DAMPING = np.array([1.0e4, 3.0e4, 3.0e4])
    DEFAULT_NORMAL_FORCE_COEFFICIENT = 2.5
    DEFAULT_ALIGNMENT_RESTORING_COEFFICIENT = 0.0

    def __init__(
        self,
        initial_position,
        initial_velocity,
        initial_orientation=None,
        initial_body_rates=None,
        mass=DEFAULT_MASS,
        rotational_mass=None,
        area=DEFAULT_AREA,
        cd0=DEFAULT_CD0,
        normal_force_coefficient=DEFAULT_NORMAL_FORCE_COEFFICIENT,
        alignment_restoring_coefficient=DEFAULT_ALIGNMENT_RESTORING_COEFFICIENT,
        angular_damping=None,
        use_coriolis=True,
    ):
        if rotational_mass is None:
            rotational_mass = self.DEFAULT_ROTATIONAL_MASS.copy()
        if angular_damping is None:
            angular_damping = self.DEFAULT_ANGULAR_DAMPING.copy()

        super().__init__(
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            initial_orientation=initial_orientation,
            initial_body_rates=initial_body_rates,
            stages=None,
            mass=mass,
            propellant_mass=0.0,
            rotational_mass=rotational_mass,
            area=area,
            cd0=cd0,
            normal_force_coefficient=normal_force_coefficient,
            alignment_restoring_coefficient=alignment_restoring_coefficient,
            max_thrust=0.0,
            max_steering_moment=0.0,
            angular_damping=angular_damping,
            use_coriolis=use_coriolis,
        )

        self.thrust_cmd = 0.0
        self.steer_cmd = 0.0
        self.steer_direction_body = np.array([0.0, 1.0, 0.0])


class RocketController(Controller):
    """Simple phase-based controller for `RocketMover` ballistic missions."""

    BOOST_VERTICAL = "boost_vertical"
    PITCH_OVER = "pitch_over"
    POWERED_ASCENT = "powered_ascent"
    STAGE_SEPARATION = "stage_separation"
    BALLISTIC_COAST = "ballistic_coast"
    IMPACT = "impact"

    VALID_PHASES = {
        BOOST_VERTICAL,
        PITCH_OVER,
        POWERED_ASCENT,
        STAGE_SEPARATION,
        BALLISTIC_COAST,
        IMPACT,
    }

    def __init__(
        self,
        target_position_ecef=None,
        launch_azimuth=None,
        vertical_rise_time=0.0,
        pitch_over_duration=0.0,
        target_ascent_pitch=None,
        steer_kp=2.0,
        steer_kd=0.5,
        separation_delay=0.0,
        update_interval=0.1,
        initial_phase=None,
    ):
        """Create a minimal ascent-program controller for `RocketMover`.

        Parameters:
            target_position_ecef: Optional ECEF target position vector. When
                `launch_azimuth` is omitted, the controller derives an ascent
                azimuth from the current position toward this target.
            launch_azimuth: Optional ascent azimuth in radians. When provided,
                it overrides any azimuth derived from `target_position_ecef`.
            vertical_rise_time: Initial vertical-boost duration in seconds.
            pitch_over_duration: Pitch-over transition duration in seconds.
            target_ascent_pitch: Optional target ascent pitch angle in radians
                above local horizontal. Defaults to 45 degrees when omitted.
            steer_kp: Proportional gain applied to body-frame pointing error.
            steer_kd: Damping gain applied to transverse body rates.
            separation_delay: Delay between burnout and stage separation in seconds.
            update_interval: Controller execution period in seconds.
            initial_phase: Optional initial phase constant. Defaults to
                `BOOST_VERTICAL`.
        """
        super().__init__(update_interval=update_interval)
        phase = self.BOOST_VERTICAL if initial_phase is None else initial_phase
        if phase not in self.VALID_PHASES:
            raise ValueError(f"initial_phase must be one of {sorted(self.VALID_PHASES)}")

        self.target_position_ecef = self._coerce_optional_vector3(
            target_position_ecef,
            "target_position_ecef",
        )
        self.launch_azimuth = self._coerce_optional_finite_scalar(
            launch_azimuth,
            "launch_azimuth",
        )
        self.vertical_rise_time = self._coerce_nonnegative_scalar(
            vertical_rise_time,
            "vertical_rise_time",
        )
        self.pitch_over_duration = self._coerce_nonnegative_scalar(
            pitch_over_duration,
            "pitch_over_duration",
        )
        self.target_ascent_pitch = self._coerce_optional_finite_scalar(
            target_ascent_pitch,
            "target_ascent_pitch",
        )
        self.steer_kp = self._coerce_nonnegative_scalar(steer_kp, "steer_kp")
        self.steer_kd = self._coerce_nonnegative_scalar(steer_kd, "steer_kd")
        self.separation_delay = self._coerce_nonnegative_scalar(
            separation_delay,
            "separation_delay",
        )
        update_interval = self._coerce_positive_scalar(update_interval, "update_interval")
        self.update_interval = update_interval

        self.phase = phase
        self.phase_start_time = None
        self._published_burnout_stage_indices = set()
        self._published_separation_stage_indices = set()
        self._ballistic_coast_published = False
        self._ground_impact_published = False

    def _coerce_optional_vector3(self, value, name):
        if value is None:
            return None
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector

    def _coerce_optional_finite_scalar(self, value, name):
        if value is None:
            return None
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return scalar

    def _coerce_nonnegative_scalar(self, value, name):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        if scalar < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return scalar

    def _coerce_positive_scalar(self, value, name):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        if scalar <= 0.0:
            raise ValueError(f"{name} must be greater than 0")
        return scalar

    def _enter_phase(self, phase, t):
        if phase not in self.VALID_PHASES:
            raise ValueError(f"phase must be one of {sorted(self.VALID_PHASES)}")
        self.phase = phase
        self.phase_start_time = float(t)

    def _phase_elapsed(self, t):
        if self.phase_start_time is None:
            return 0.0
        return max(float(t) - self.phase_start_time, 0.0)

    def _local_enu_basis(self, position):
        position = np.asarray(position, dtype=float)
        position_norm = np.linalg.norm(position)
        if position.shape != (3,) or position_norm < 1e-8:
            return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])

        lat_deg, lon_deg, _ = ecef_to_lla(position[0], position[1], position[2])
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.array([
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ])
        up = position / position_norm
        return east, north, up

    def _current_horizontal_direction(self, mover, east, north, up):
        forward_horizontal = mover.forward_axis - np.dot(mover.forward_axis, up) * up
        horizontal_norm = np.linalg.norm(forward_horizontal)
        if horizontal_norm > 1e-8:
            return forward_horizontal / horizontal_norm
        return east

    def _resolve_launch_azimuth(self, mover):
        if self.launch_azimuth is not None:
            return self.launch_azimuth

        east, north, up = self._local_enu_basis(mover.position)
        if self.target_position_ecef is not None:
            rel = self.target_position_ecef - mover.position
            rel_horizontal = rel - np.dot(rel, up) * up
            rel_horizontal_norm = np.linalg.norm(rel_horizontal)
            if rel_horizontal_norm > 1e-8:
                east_component = np.dot(rel_horizontal, east)
                north_component = np.dot(rel_horizontal, north)
                return np.arctan2(east_component, north_component)

        current_horizontal = self._current_horizontal_direction(mover, east, north, up)
        return np.arctan2(np.dot(current_horizontal, east), np.dot(current_horizontal, north))

    def _horizontal_direction_from_azimuth(self, east, north, azimuth):
        horizontal_direction = np.cos(azimuth) * north + np.sin(azimuth) * east
        horizontal_norm = np.linalg.norm(horizontal_direction)
        if horizontal_norm < 1e-8:
            return north
        return horizontal_direction / horizontal_norm

    def _resolve_target_ascent_pitch(self):
        if self.target_ascent_pitch is None:
            return np.radians(45.0)
        return np.clip(self.target_ascent_pitch, -0.5 * np.pi, 0.5 * np.pi)

    def _forward_direction_from_pitch(self, horizontal_direction, up, pitch):
        pitch = np.clip(pitch, -0.5 * np.pi, 0.5 * np.pi)
        forward_direction = np.cos(pitch) * horizontal_direction + np.sin(pitch) * up
        forward_norm = np.linalg.norm(forward_direction)
        if forward_norm < 1e-8:
            return up
        return forward_direction / forward_norm

    def _advance_guidance_phase(self, t):
        if self.phase == self.BOOST_VERTICAL and self._phase_elapsed(t) >= self.vertical_rise_time:
            next_phase = self.PITCH_OVER if self.pitch_over_duration > 0.0 else self.POWERED_ASCENT
            self._enter_phase(next_phase, t)

        if self.phase == self.PITCH_OVER and self._phase_elapsed(t) >= self.pitch_over_duration:
            self._enter_phase(self.POWERED_ASCENT, t)

    def _desired_forward_direction(self, t, mover):
        east, north, up = self._local_enu_basis(mover.position)
        if self.phase == self.BOOST_VERTICAL:
            return up
        if self.phase not in (self.PITCH_OVER, self.POWERED_ASCENT):
            return None

        azimuth = self._resolve_launch_azimuth(mover)
        horizontal_direction = self._horizontal_direction_from_azimuth(east, north, azimuth)
        target_ascent_pitch = self._resolve_target_ascent_pitch()
        if self.phase == self.PITCH_OVER:
            duration = max(self.pitch_over_duration, 1e-9)
            pitch_progress = np.clip(self._phase_elapsed(t) / duration, 0.0, 1.0)
            pitch = (1.0 - pitch_progress) * (0.5 * np.pi) + pitch_progress * target_ascent_pitch
        else:
            pitch = target_ascent_pitch
        return self._forward_direction_from_pitch(horizontal_direction, up, pitch)

    def _powered_flight_phase(self):
        return self.phase in (self.BOOST_VERTICAL, self.PITCH_OVER, self.POWERED_ASCENT)

    def _publish_stage_burnout(self, engine, mover):
        stage_index = mover.active_stage_index
        if stage_index in self._published_burnout_stage_indices:
            return
        engine.broker.publish("stage_burnout", self.platform, stage_index)
        self._published_burnout_stage_indices.add(stage_index)

    def _publish_stage_separation(self, engine, stage_index):
        if stage_index in self._published_separation_stage_indices:
            return
        engine.broker.publish("stage_separation", self.platform, stage_index)
        self._published_separation_stage_indices.add(stage_index)

    def _publish_ballistic_coast_start(self, engine):
        if self._ballistic_coast_published:
            return
        engine.broker.publish("ballistic_coast_start", self.platform)
        self._ballistic_coast_published = True

    def _publish_ground_impact(self, engine):
        if self._ground_impact_published:
            return
        engine.broker.publish("ground_impact", self.platform)
        self._ground_impact_published = True

    def _has_ground_impact(self, mover):
        _, _, altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
        return altitude < 0.0

    def _enter_impact_phase(self, engine, mover, t):
        mover.thrust_cmd = 0.0
        mover.steer_cmd = 0.0
        self._enter_phase(self.IMPACT, t)
        self._publish_ground_impact(engine)

    def _burnout_detected(self, mover):
        if not mover.stages:
            return False
        propellant_mass = mover.propellant_mass
        if propellant_mass <= 0.0:
            return True
        stage = mover._active_stage()
        if stage is None:
            return True
        if "mass_flow_rate" in stage:
            return float(stage["mass_flow_rate"]) <= 0.0
        return (stage["propellant_mass"] / stage["burn_duration"]) <= 0.0

    def _begin_post_burnout_phase(self, engine, mover, t):
        mover.thrust_cmd = 0.0
        mover.steer_cmd = 0.0
        self._publish_stage_burnout(engine, mover)
        if mover.can_separate_stage():
            self._enter_phase(self.STAGE_SEPARATION, t)
            return
        self._publish_ballistic_coast_start(engine)
        self._enter_phase(self.BALLISTIC_COAST, t)

    def _update_stage_separation(self, t, engine, mover):
        mover.thrust_cmd = 0.0
        mover.steer_cmd = 0.0
        if self._phase_elapsed(t) < self.separation_delay:
            return

        if mover.can_separate_stage():
            separated_stage_index = mover.active_stage_index
            mover.advance_to_next_stage(engine)
            self._publish_stage_separation(engine, separated_stage_index)
            self._enter_phase(self.POWERED_ASCENT, t)
            return

        self._publish_ballistic_coast_start(engine)
        self._enter_phase(self.BALLISTIC_COAST, t)

    def _apply_pointing_guidance(self, mover, desired_forward):
        if desired_forward is None:
            mover.steer_cmd = 0.0
            return

        current_forward = mover.forward_axis
        current_forward = current_forward / max(np.linalg.norm(current_forward), 1e-8)
        desired_forward = np.asarray(desired_forward, dtype=float)
        desired_forward = desired_forward / max(np.linalg.norm(desired_forward), 1e-8)

        error_axis_world = np.cross(current_forward, desired_forward)
        error_axis_body = mover.orientation.T @ error_axis_world
        error_axis_body[0] = 0.0

        transverse_body_rates = np.asarray(mover.omega_body, dtype=float).copy()
        transverse_body_rates[0] = 0.0
        steering_command_body = self.steer_kp * error_axis_body - self.steer_kd * transverse_body_rates
        steering_command_body[0] = 0.0

        steering_magnitude = np.linalg.norm(steering_command_body)
        if steering_magnitude < 1e-8:
            mover.steer_cmd = 0.0
            return

        mover.steer_direction_body = steering_command_body
        mover.steer_cmd = 100.0 * steering_magnitude

    def initialize(self, engine):
        super().initialize(engine)
        if self.phase_start_time is None:
            self.phase_start_time = float(engine.t)

    def update(self, t, engine):
        """Advance the controller phase machine for a `RocketMover`.

        The controller uses a small ascent program: boost vertically, pitch over
        toward the launch azimuth, then hold the target ascent pitch while the
        staging logic decides when to separate and coast.
        """
        mover = self.platform.mover
        if not isinstance(mover, RocketMover):
            return

        if self.phase_start_time is None:
            self.phase_start_time = float(t)

        if self.phase != self.IMPACT and self._has_ground_impact(mover):
            self._enter_impact_phase(engine, mover, t)
            return

        if self.phase == self.IMPACT:
            mover.thrust_cmd = 0.0
            mover.steer_cmd = 0.0
            return

        if self.phase == self.STAGE_SEPARATION:
            self._update_stage_separation(t, engine, mover)
            return

        self._advance_guidance_phase(t)

        if self._powered_flight_phase() and self._burnout_detected(mover):
            self._begin_post_burnout_phase(engine, mover, t)
            return

        if self.phase in (self.STAGE_SEPARATION, self.BALLISTIC_COAST, self.IMPACT):
            mover.thrust_cmd = 0.0
            mover.steer_cmd = 0.0
            return

        if self.phase in (self.BOOST_VERTICAL, self.PITCH_OVER, self.POWERED_ASCENT):
            mover.thrust_cmd = 100.0
            desired_forward = self._desired_forward_direction(t, mover)
            self._apply_pointing_guidance(mover, desired_forward)
            return

        raise RuntimeError(f"unhandled rocket-controller phase: {self.phase}")
