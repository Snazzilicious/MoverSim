import numpy as np

from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.math.coordinates import ecef_to_enu, ecef_to_lla, lla_to_ecef
from mover_sim.math.orientation import rotate_vector_by_quaternion
from mover_sim.scenario_air_launched_cruise_missile import (
    AirLaunchedCruiseMissileController,
    AirLaunchedCruiseMissileMothershipController,
    AirLaunchedCruiseMissileMothershipMover,
    AirLaunchedCruiseMissileMover,
    _begin_mothership_rtb,
    _make_missile_release_callback,
    _spawn_released_missile,
    _orientation_from_heading_pitch,
    _velocity_from_heading_speed,
    run_air_launched_cruise_missile_scenario,
)


def test_air_launched_orientation_from_heading_pitch_matches_local_enu_command():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    heading = np.pi / 2.0
    pitch = np.radians(10.0)

    quaternion = _orientation_from_heading_pitch(position, heading, pitch)
    forward_world = rotate_vector_by_quaternion([1.0, 0.0, 0.0], quaternion)
    sample_point = position + forward_world
    east, north, up = ecef_to_enu(sample_point[0], sample_point[1], sample_point[2], 0.0, 0.0, 1000.0)

    expected_forward_enu = np.array([np.cos(pitch), 0.0, np.sin(pitch)])
    actual_forward_enu = np.array([east, north, up], dtype=float)
    actual_forward_enu /= np.linalg.norm(actual_forward_enu)

    assert np.allclose(actual_forward_enu, expected_forward_enu, atol=1e-7)


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


def test_air_launched_mothership_mover_uses_13_element_state():
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    orientation = _orientation_from_heading_pitch(position, 0.0, 0.0)
    velocity = _velocity_from_heading_speed(position, 0.0, 200.0)

    mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
    )
    state = mover.get_state()

    assert mover.get_state_dimension() == 13
    assert state.shape == (13,)
    assert np.allclose(mover.position, state[:3])
    assert np.allclose(mover.velocity, state[3:6])
    assert np.allclose(mover.orientation, state[6:10])
    assert np.allclose(mover.body_rates, state[10:13])


def test_air_launched_released_missile_mover_uses_13_element_state():
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    orientation = _orientation_from_heading_pitch(position, 0.0, 0.0)
    velocity = _velocity_from_heading_speed(position, 0.0, 220.0)

    mover = AirLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
    )
    state = mover.get_state()

    assert mover.get_state_dimension() == 13
    assert state.shape == (13,)
    assert np.allclose(mover.position, state[:3])
    assert np.allclose(mover.velocity, state[3:6])
    assert np.allclose(mover.orientation, state[6:10])
    assert np.allclose(mover.body_rates, state[10:13])


def test_air_launched_missile_is_registered_only_after_release_time():
    engine = SimulationEngine()
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    orientation = _orientation_from_heading_pitch(position, 0.0, 0.0)
    velocity = _velocity_from_heading_speed(position, 0.0, 200.0)

    mothership_mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
    )
    mothership_controller = AirLaunchedCruiseMissileMothershipController(
        cruise_speed=200.0,
        cruise_altitude=1500.0,
        cruise_heading=0.0,
        rtb_position_ecef=position,
    )
    mothership = Platform("mothership", mothership_mover, mothership_controller)
    engine.register_platform(mothership)

    def spawn_missile(current_engine, current_mothership_platform):
        _spawn_released_missile(
            current_engine,
            current_mothership_platform,
            release_time=1.0,
            missile_cruise_speed=250.0,
            missile_cruise_altitude=1200.0,
            missile_cruise_heading=0.0,
            missile_drop_duration=0.5,
        )

    engine.schedule(1.0, _make_missile_release_callback(mothership, spawn_missile), "MissileRelease")

    engine.run(0.5)
    assert "released_missile" not in engine.platforms

    engine.run(1.1)
    assert "released_missile" in engine.platforms


def test_air_launched_missile_transitions_from_drop_to_ignite_on_time():
    engine = SimulationEngine()
    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    orientation = _orientation_from_heading_pitch(position, 0.0, 0.0)
    velocity = _velocity_from_heading_speed(position, 0.0, 200.0)

    mover = AirLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
    )
    controller = AirLaunchedCruiseMissileController(
        release_time=1.0,
        cruise_speed=250.0,
        cruise_altitude=1200.0,
        cruise_heading=0.0,
        drop_duration=0.5,
    )
    platform = Platform("released_missile", mover, controller)
    engine.register_platform(platform)
    controller.initialize(engine)

    controller.update(1.25, engine)
    assert controller.phase == controller.DROP_PHASE
    assert mover.thrust_cmd == 0.0

    controller.update(1.5, engine)
    assert controller.phase == controller.IGNITE_TRANSITION_PHASE


def test_air_launched_missile_guidance_reduces_heading_and_altitude_error_after_ignition():
    engine = SimulationEngine()
    engine.max_step = 0.05

    position = np.array(lla_to_ecef(0.0, 0.0, 1100.0), dtype=float)
    orientation = _orientation_from_heading_pitch(position, np.pi / 2.0, 0.0)
    velocity = _velocity_from_heading_speed(position, np.pi / 2.0, 200.0)

    mover = AirLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
    )
    controller = AirLaunchedCruiseMissileController(
        release_time=0.0,
        cruise_speed=250.0,
        cruise_altitude=1000.0,
        cruise_heading=0.0,
        drop_duration=0.0,
        update_interval=0.05,
    )
    controller.phase = controller.IGNITE_TRANSITION_PHASE

    platform = Platform("released_missile", mover, controller)
    engine.register_platform(platform)

    initial_heading_error = abs(controller._compute_heading_error(mover, controller.cruise_heading))
    _, _, initial_altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    initial_altitude_error = abs(controller.cruise_altitude - initial_altitude)

    engine.run(5.0)

    final_heading_error = abs(controller._compute_heading_error(mover, controller.cruise_heading))
    _, _, final_altitude = ecef_to_lla(mover.position[0], mover.position[1], mover.position[2])
    final_altitude_error = abs(controller.cruise_altitude - final_altitude)

    assert final_heading_error < initial_heading_error
    assert final_altitude_error < initial_altitude_error


def test_air_launched_mothership_switches_to_rtb_and_reaches_hold_state():
    engine = SimulationEngine()
    engine.max_step = 0.05

    position = np.array(lla_to_ecef(0.0, 0.0, 1500.0), dtype=float)
    rtb_position = np.array(lla_to_ecef(0.0, 0.001, 1500.0), dtype=float)
    orientation = _orientation_from_heading_pitch(position, np.pi / 2.0, 0.0)
    velocity = _velocity_from_heading_speed(position, np.pi / 2.0, 200.0)

    mover = AirLaunchedCruiseMissileMothershipMover(
        initial_position=position,
        initial_velocity=velocity,
        initial_orientation=orientation,
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

    _begin_mothership_rtb(engine, platform)
    assert controller.phase == controller.POST_RELEASE_RTB_PHASE

    engine.run(10.0)
    assert controller.phase == controller.RTB_HOLD_PHASE


def test_air_launched_scenario_stops_on_missile_ground_impact_and_logs_event(tmp_path):
    output_path = tmp_path / "air_launched_cruise_missile.h5"

    result = run_air_launched_cruise_missile_scenario(
        mothership_initial_position_ecef=lla_to_ecef(0.0, 0.0, 20.0),
        mothership_cruise_speed=200.0,
        mothership_cruise_altitude=20.0,
        mothership_cruise_heading=0.0,
        mothership_rtb_position_ecef=lla_to_ecef(0.0, 0.0, 20.0),
        missile_launch_time=0.1,
        missile_cruise_speed=250.0,
        missile_cruise_altitude=500.0,
        missile_cruise_heading=0.0,
        missile_drop_duration=0.0,
        t_end=10.0,
        sample_interval=0.1,
        output_path=output_path,
    )

    assert result["engine"].t < 10.0

    import h5py

    with h5py.File(output_path, "r") as h5:
        assert "events" in h5
        topics = [topic.decode("utf-8") if isinstance(topic, bytes) else topic for topic in h5["events"]["topic"][:]]
        assert "missile_ground_impact" in topics


def test_air_launched_hdf5_output_contains_mothership_and_missile_trajectory_groups(tmp_path):
    output_path = tmp_path / "air_launched_cruise_missile_groups.h5"

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
        output_path=output_path,
    )

    import h5py

    with h5py.File(output_path, "r") as h5:
        assert "trajectories" in h5
        assert "mothership" in h5["trajectories"]
        assert "released_missile" in h5["trajectories"]

        mothership_group = h5["trajectories"]["mothership"]
        missile_group = h5["trajectories"]["released_missile"]

        for group in (mothership_group, missile_group):
            assert "orientation" in group
            assert "body_rates" in group
            assert group["orientation"].shape[1] == 4
            assert group["body_rates"].shape[1] == 3


def test_air_launched_hdf5_event_table_contains_scenario_topics(tmp_path):
    output_path = tmp_path / "air_launched_cruise_missile_events.h5"

    run_air_launched_cruise_missile_scenario(
        mothership_initial_position_ecef=lla_to_ecef(0.0, 0.0, 20.0),
        mothership_cruise_speed=200.0,
        mothership_cruise_altitude=20.0,
        mothership_cruise_heading=0.0,
        mothership_rtb_position_ecef=lla_to_ecef(0.0, 0.0, 20.0),
        missile_launch_time=0.1,
        missile_cruise_speed=250.0,
        missile_cruise_altitude=500.0,
        missile_cruise_heading=0.0,
        missile_drop_duration=0.0,
        t_end=10.0,
        sample_interval=0.1,
        output_path=output_path,
    )

    import h5py

    with h5py.File(output_path, "r") as h5:
        topics = [topic.decode("utf-8") if isinstance(topic, bytes) else topic for topic in h5["events"]["topic"][:]]
        assert "platform_registered" in topics
        assert "missile_release" in topics
        assert "missile_drop_start" in topics
        assert "missile_drop_end" in topics
        assert "missile_ignite" in topics
        assert "mothership_rtb_start" in topics
        assert "missile_ground_impact" in topics
