import numpy as np
from mover_sim.core.mover import NewtonianMover, TranslationalMover, TranslationalNewtonianMover
from mover_sim.core.controller import Controller
from mover_sim.math.physics import aerodynamic_drag_force, air_density, coriolis_acceleration, gravity, GM
from mover_sim.math.coordinates import ecef_to_lla, ecef_to_enu, lla_to_ecef
from mover_sim.math.orientation import (
    build_aircraft_body_axes,
    normalize_quaternion,
    quaternion_derivative_from_body_rates,
    quaternion_from_basis,
    rotate_vector_by_quaternion,
)

class AircraftMover(TranslationalNewtonianMover):
    """
    A Newtonian mover representing a simplified 6-DOF style aircraft.
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


class Aircraft6DOFMover(TranslationalMover, NewtonianMover):
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
