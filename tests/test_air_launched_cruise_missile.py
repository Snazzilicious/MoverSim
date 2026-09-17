import h5py
import numpy as np

from examples.scenario_air_launched_cruise_missile import (
    AirLaunchedCruiseMissileController,
    AirLaunchedCruiseMissileMothershipController,
    AirLaunchedCruiseMissileMothershipMover,
    AirLaunchedCruiseMissileMover,
    MissileLaunchEvent,
    MothershipRTBEvent,
    _velocity_from_heading_speed,
    run_air_launched_cruise_missile_scenario,
)
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.math.coordinates import ecef_to_enu, ecef_to_lla, lla_to_ecef


def test_air_launched_velocity_from_heading_speed_matches_local_enu_command():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    heading = np.pi / 2.0
    speed = 200.0

    velocity = _velocity_from_heading_speed(position, heading, speed)
    sample_point = position + velocity
    east, north, up = ecef_to_enu(sample_point[0], sample_point[1], sample_point[2], 0.0, 0.0, 1000.0)
    velocity_enu = np.array([east, north, up], dtype=float)

    assert np.isclose(np.linalg.norm(velocity_enu), speed, atol=1e-7)
    assert np.allclose(velocity_enu / np.linalg.norm(velocity_enu), [1.0, 0.0, 0.0], atol=1e-7)


def test_air_launched_mothership_mover_uses_fixed_wing_state_layout():
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    velocity = _velocity_from_heading_speed(position, 0.0, 200.0)

    mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_body_rates=np.zeros(3),
    )
    state = mover.get_state()

    assert mover.get_state_dimension() == 18
    assert state.shape == (18,)
    assert mover.orientation.shape == (3, 3)
    assert np.allclose(mover.orientation, state[mover.get_orientation_slice()].reshape((3, 3)))
    assert np.allclose(state[mover.get_omega_slice()], np.zeros(3))


def test_air_launched_released_missile_mover_uses_fixed_wing_state_layout():
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    velocity = _velocity_from_heading_speed(position, 0.0, 220.0)

    mover = AirLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_body_rates=np.zeros(3),
    )
    state = mover.get_state()

    assert mover.get_state_dimension() == 18
    assert state.shape == (18,)
    assert mover.orientation.shape == (3, 3)
    assert np.allclose(mover.orientation, state[mover.get_orientation_slice()].reshape((3, 3)))
    assert np.allclose(state[mover.get_omega_slice()], np.zeros(3))


def test_air_launched_missile_controller_publishes_drop_events_and_transitions_on_time():
    engine = SimulationEngine()
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    velocity = _velocity_from_heading_speed(position, 0.0, 200.0)

    mover = AirLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_body_rates=np.zeros(3),
    )
    controller = AirLaunchedCruiseMissileController(
        cruise_speed=250.0,
        cruise_altitude=1200.0,
        cruise_heading=0.0,
        drop_duration=0.5,
    )
    platform = Platform("released_missile", mover, controller)
    engine.register_platform(platform)

    drop_events = []
    engine.broker.subscribe("missile_drop_start", lambda current_platform: drop_events.append(("start", current_platform.id)))
    engine.broker.subscribe("missile_drop_end", lambda current_platform: drop_events.append(("end", current_platform.id)))

    controller.initialize(engine)

    assert controller.phase == controller.DROP_PHASE
    assert drop_events == [("start", "released_missile")]

    controller.update(0.25, engine)
    assert controller.phase == controller.DROP_PHASE
    assert mover.thrust_cmd == 0.0
    assert drop_events == [("start", "released_missile")]

    controller.update(0.5, engine)
    assert controller.phase == controller.CRUISE_PHASE
    assert drop_events == [("start", "released_missile"), ("end", "released_missile")]

    controller.update(1.0, engine)
    assert drop_events == [("start", "released_missile"), ("end", "released_missile")]


def test_air_launched_missile_cruise_guidance_reduces_heading_and_altitude_error_after_drop():
    engine = SimulationEngine()
    engine.max_step = 0.05

    position = np.array(lla_to_ecef(0.0, 0.0, 1100.0), dtype=float)
    velocity = _velocity_from_heading_speed(position, np.pi / 2.0, 200.0)

    mover = AirLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_body_rates=np.zeros(3),
    )
    controller = AirLaunchedCruiseMissileController(
        cruise_speed=250.0,
        cruise_altitude=1000.0,
        cruise_heading=0.0,
        drop_duration=0.0,
        update_interval=0.05,
    )
    controller.phase = controller.CRUISE_PHASE
    controller.t_launch = 0.0

    platform = Platform("released_missile", mover, controller)
    engine.register_platform(platform)

    initial_heading_error = abs(
        controller._heading_error_to_direction(
            mover,
            controller._desired_cruise_horizontal_direction(mover),
        )
    )
    _, _, initial_altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    initial_altitude_error = abs(controller.cruise_altitude - initial_altitude)

    engine.run(5.0)

    final_heading_error = abs(
        controller._heading_error_to_direction(
            mover,
            controller._desired_cruise_horizontal_direction(mover),
        )
    )
    _, _, final_altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    final_altitude_error = abs(controller.cruise_altitude - final_altitude)

    assert final_heading_error < initial_heading_error
    assert final_altitude_error < initial_altitude_error


def test_missile_launch_event_spawns_released_missile_from_live_mothership_state_and_schedules_rtb():
    engine = SimulationEngine()
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    velocity = _velocity_from_heading_speed(position, 0.0, 200.0)
    rtb_position = np.array(lla_to_ecef(0.0, 0.001, 1500.0), dtype=float)

    mothership_mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_body_rates=np.zeros(3),
    )
    mothership_controller = AirLaunchedCruiseMissileMothershipController(
        cruise_speed=200.0,
        cruise_altitude=1500.0,
        cruise_heading=0.0,
        rtb_position_ecef=rtb_position,
    )
    mothership = Platform("mothership", mothership_mover, mothership_controller)
    engine.register_platform(mothership)

    mover_slice = engine.context.get_state_slice(mothership_mover)
    state = engine.context.committed_y[mover_slice].copy()
    state[:3] += np.array([10.0, 20.0, 30.0])
    state[3:6] = np.array([90.0, 80.0, 70.0])
    state[mothership_mover.get_omega_slice()] = np.array([0.1, 0.2, 0.3])
    engine.context.committed_y[mover_slice] = state

    launch_event = MissileLaunchEvent(
        missile_launch_time=1.0,
        mothership_platform=mothership,
        missile_cruise_speed=250.0,
        missile_cruise_altitude=1200.0,
        missile_cruise_heading=0.0,
        missile_drop_duration=0.5,
        mothership_rtb_position_ecef=rtb_position,
    )

    release_events = []
    engine.broker.subscribe("missile_release", lambda current_platform: release_events.append(current_platform.id))

    launch_event(engine)

    assert "released_missile" in engine.platforms
    missile = engine.platforms["released_missile"]
    assert release_events == ["released_missile"]
    assert np.allclose(missile.mover.position, state[:3])
    assert np.allclose(missile.mover.velocity, state[3:6])
    assert np.allclose(missile.mover.get_state()[missile.mover.get_omega_slice()], np.array([0.1, 0.2, 0.3]))
    assert np.isclose(engine.scheduler.peek_next_time(), 2.0)


def test_mothership_rtb_event_switches_controller_mode_and_publishes_event():
    engine = SimulationEngine()
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    velocity = _velocity_from_heading_speed(position, np.pi / 2.0, 200.0)
    rtb_position = np.array(lla_to_ecef(0.0, 0.001, 1500.0), dtype=float)

    mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_body_rates=np.zeros(3),
    )
    controller = AirLaunchedCruiseMissileMothershipController(
        cruise_speed=200.0,
        cruise_altitude=1500.0,
        cruise_heading=np.pi / 2.0,
        rtb_position_ecef=rtb_position,
        update_interval=0.05,
    )
    platform = Platform("mothership", mover, controller)
    engine.register_platform(platform)

    rtb_events = []
    engine.broker.subscribe("mothership_rtb_start", lambda current_platform: rtb_events.append(current_platform.id))

    rtb_event = MothershipRTBEvent(2.0, platform, rtb_position)
    rtb_event(engine)

    assert controller.mode == controller.RTB_MODE
    assert controller.hold_active is False
    assert rtb_events == ["mothership"]


def test_air_launched_mothership_cruise_guidance_holds_course_prior_to_rtb():
    engine = SimulationEngine()
    engine.max_step = 0.05

    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    velocity = _velocity_from_heading_speed(position, 0.0, 200.0)

    mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_body_rates=np.zeros(3),
    )
    controller = AirLaunchedCruiseMissileMothershipController(
        cruise_speed=200.0,
        cruise_altitude=1500.0,
        cruise_heading=0.0,
        update_interval=0.05,
    )
    platform = Platform("mothership", mover, controller)
    engine.register_platform(platform)

    initial_heading_error = abs(
        controller._heading_error_to_direction(
            mover,
            controller._desired_cruise_horizontal_direction(mover),
        )
    )

    engine.run(5.0)

    final_heading_error = abs(
        controller._heading_error_to_direction(
            mover,
            controller._desired_cruise_horizontal_direction(mover),
        )
    )

    assert initial_heading_error < 1e-9
    assert final_heading_error < np.radians(5.0)


def test_air_launched_mothership_rtb_guidance_holds_aligned_course_and_closes_distance():
    engine = SimulationEngine()
    engine.max_step = 0.05

    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    rtb_position = np.array(lla_to_ecef(0.01, 0.0, 1500.0), dtype=float)
    velocity = _velocity_from_heading_speed(position, 0.0, 200.0)

    mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_body_rates=np.zeros(3),
    )
    controller = AirLaunchedCruiseMissileMothershipController(
        cruise_speed=200.0,
        cruise_altitude=1500.0,
        cruise_heading=0.0,
        rtb_position_ecef=rtb_position,
        update_interval=0.05,
    )
    platform = Platform("mothership", mover, controller)
    engine.register_platform(platform)

    controller.enter_rtb(rtb_position)

    initial_heading_error = abs(
        controller._heading_error_to_direction(
            mover,
            controller._desired_rtb_horizontal_direction(mover),
        )
    )
    initial_distance = np.linalg.norm(rtb_position - mover.position)

    engine.run(5.0)

    final_heading_error = abs(
        controller._heading_error_to_direction(
            mover,
            controller._desired_rtb_horizontal_direction(mover),
        )
    )
    final_distance = np.linalg.norm(rtb_position - mover.position)

    assert initial_heading_error < 1e-9
    assert final_heading_error < np.radians(5.0)
    assert final_distance < initial_distance


def test_air_launched_hdf5_output_contains_mothership_and_missile_trajectory_groups_and_events(tmp_path):
    output_path = tmp_path / "air_launched_cruise_missile.h5"

    with h5py.File(output_path, "w") as h5:
        run_air_launched_cruise_missile_scenario(
            mothership_initial_position_ecef=lla_to_ecef(0.0, 0.0, 1500.0),
            mothership_cruise_speed=200.0,
            mothership_cruise_altitude=1500.0,
            mothership_cruise_heading=0.0,
            mothership_rtb_position_ecef=lla_to_ecef(0.0, 0.001, 1500.0),
            missile_launch_time=0.1,
            missile_cruise_speed=250.0,
            missile_cruise_altitude=1200.0,
            missile_cruise_heading=0.0,
            missile_drop_duration=0.1,
            t_end=3.0,
            sample_interval=0.1,
            output_group=h5.create_group("run"),
        )

    with h5py.File(output_path, "r") as h5:
        assert "trajectories" in h5["run"]
        assert "mothership" in h5["run"]["trajectories"]
        assert "released_missile" in h5["run"]["trajectories"]

        for group_name in ("mothership", "released_missile"):
            group = h5["run"]["trajectories"][group_name]
            assert "time" in group
            assert "state" in group
            assert "position" in group
            assert "velocity" in group
            assert "lla" in group
            assert group["state"].shape[1] == 18

        topics = [
            topic.decode("utf-8") if isinstance(topic, bytes) else topic
            for topic in h5["run"]["events"]["topic"][:]
        ]
        assert "platform_registered" in topics
        assert "missile_release" in topics
        assert "missile_drop_start" in topics
        assert "missile_drop_end" in topics
        assert "mothership_rtb_start" in topics


def test_air_launched_rejects_non_group_output_target():
    try:
        run_air_launched_cruise_missile_scenario(
            mothership_initial_position_ecef=lla_to_ecef(0.0, 0.0, 1500.0),
            mothership_cruise_speed=200.0,
            mothership_cruise_altitude=1500.0,
            mothership_cruise_heading=0.0,
            mothership_rtb_position_ecef=lla_to_ecef(0.0, 0.001, 1500.0),
            missile_launch_time=0.1,
            missile_cruise_speed=250.0,
            missile_cruise_altitude=1200.0,
            missile_cruise_heading=0.0,
            missile_drop_duration=0.1,
            t_end=1.0,
            sample_interval=0.1,
            output_group="not-a-group",
        )
    except ValueError as exc:
        assert str(exc) == "group must be an h5py.Group"
    else:
        raise AssertionError("Expected a ValueError for non-group output target")


def test_air_launched_rejects_reused_group(tmp_path):
    output_path = tmp_path / "air_reuse.h5"

    with h5py.File(output_path, "w") as h5:
        group = h5.create_group("run")
        run_air_launched_cruise_missile_scenario(
            mothership_initial_position_ecef=lla_to_ecef(0.0, 0.0, 1500.0),
            mothership_cruise_speed=200.0,
            mothership_cruise_altitude=1500.0,
            mothership_cruise_heading=0.0,
            mothership_rtb_position_ecef=lla_to_ecef(0.0, 0.001, 1500.0),
            missile_launch_time=0.1,
            missile_cruise_speed=250.0,
            missile_cruise_altitude=1200.0,
            missile_cruise_heading=0.0,
            missile_drop_duration=0.1,
            t_end=0.2,
            sample_interval=0.1,
            output_group=group,
        )

        try:
            run_air_launched_cruise_missile_scenario(
                mothership_initial_position_ecef=lla_to_ecef(0.0, 0.0, 1500.0),
                mothership_cruise_speed=200.0,
                mothership_cruise_altitude=1500.0,
                mothership_cruise_heading=0.0,
                mothership_rtb_position_ecef=lla_to_ecef(0.0, 0.001, 1500.0),
                missile_launch_time=0.1,
                missile_cruise_speed=250.0,
                missile_cruise_altitude=1200.0,
                missile_cruise_heading=0.0,
                missile_drop_duration=0.1,
                t_end=0.2,
                sample_interval=0.1,
                output_group=group,
            )
        except ValueError as exc:
            assert str(exc) == "output_group already contains logger-managed data"
        else:
            raise AssertionError("Expected reused output group to fail")


def test_air_launched_supports_sibling_groups_in_one_file(tmp_path):
    output_path = tmp_path / "air_siblings.h5"

    with h5py.File(output_path, "w") as h5:
        run_air_launched_cruise_missile_scenario(
            mothership_initial_position_ecef=lla_to_ecef(0.0, 0.0, 1500.0),
            mothership_cruise_speed=200.0,
            mothership_cruise_altitude=1500.0,
            mothership_cruise_heading=0.0,
            mothership_rtb_position_ecef=lla_to_ecef(0.0, 0.001, 1500.0),
            missile_launch_time=0.1,
            missile_cruise_speed=250.0,
            missile_cruise_altitude=1200.0,
            missile_cruise_heading=0.0,
            missile_drop_duration=0.1,
            t_end=0.2,
            sample_interval=0.1,
            output_group=h5.create_group("run_a"),
        )
        run_air_launched_cruise_missile_scenario(
            mothership_initial_position_ecef=lla_to_ecef(0.0, 0.0, 1500.0),
            mothership_cruise_speed=200.0,
            mothership_cruise_altitude=1500.0,
            mothership_cruise_heading=0.0,
            mothership_rtb_position_ecef=lla_to_ecef(0.0, 0.001, 1500.0),
            missile_launch_time=0.1,
            missile_cruise_speed=250.0,
            missile_cruise_altitude=1200.0,
            missile_cruise_heading=0.0,
            missile_drop_duration=0.1,
            t_end=0.2,
            sample_interval=0.1,
            output_group=h5.create_group("run_b"),
        )

    with h5py.File(output_path, "r") as h5:
        assert "trajectories" in h5["run_a"]
        assert "trajectories" in h5["run_b"]
