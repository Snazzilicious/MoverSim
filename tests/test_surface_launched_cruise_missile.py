import h5py
import numpy as np

from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.math.coordinates import ecef_to_enu, ecef_to_lla, lla_to_ecef
from mover_sim.math.orientation import rotate_vector_by_quaternion
from mover_sim.scenario_surface_launched_cruise_missile import (
    SurfaceLaunchedCruiseMissileController,
    SurfaceLaunchedCruiseMissileMover,
    _initial_orientation_from_heading_pitch,
    run_surface_launched_cruise_missile_scenario,
)


def test_initial_orientation_from_heading_pitch_matches_local_enu_command():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    heading = np.pi / 2.0
    pitch = np.radians(20.0)

    quaternion = _initial_orientation_from_heading_pitch(position, heading, pitch)
    forward_world = rotate_vector_by_quaternion([1.0, 0.0, 0.0], quaternion)
    sample_point = position + forward_world
    east, north, up = ecef_to_enu(sample_point[0], sample_point[1], sample_point[2], 0.0, 0.0, 1000.0)

    expected_forward_enu = np.array([
        np.cos(pitch),
        0.0,
        np.sin(pitch),
    ])
    actual_forward_enu = np.array([east, north, up], dtype=float)
    actual_forward_enu /= np.linalg.norm(actual_forward_enu)

    assert np.allclose(actual_forward_enu, expected_forward_enu, atol=1e-7)


def test_surface_launched_cruise_missile_mover_uses_13_element_state():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    orientation = _initial_orientation_from_heading_pitch(position, 0.0, np.radians(10.0))

    mover = SurfaceLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=np.zeros(3),
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


def test_surface_launched_cruise_missile_controller_transitions_out_of_boost_on_time():
    engine = SimulationEngine()
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    orientation = _initial_orientation_from_heading_pitch(position, 0.0, np.radians(15.0))
    mover = SurfaceLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
    )
    controller = SurfaceLaunchedCruiseMissileController(
        cruise_speed=250.0,
        cruise_altitude=1000.0,
        cruise_heading=0.0,
        boost_duration=2.0,
        boost_acceleration=30.0,
        launch_pitch_angle=np.radians(15.0),
    )
    platform = Platform("missile", mover, controller)
    engine.register_platform(platform)
    controller.initialize(engine)

    controller.update(1.0, engine)
    assert controller.phase == controller.BOOST_PHASE
    assert mover.boost_acceleration_cmd == 30.0

    controller.update(2.0, engine)
    assert controller.phase == controller.TRANSITION_PHASE
    assert mover.boost_acceleration_cmd == 0.0


def test_surface_launched_cruise_missile_transition_guidance_reduces_heading_and_altitude_error():
    engine = SimulationEngine()
    engine.max_step = 0.05

    position = np.array(lla_to_ecef(0.0, 0.0, 1100.0), dtype=float)
    orientation = _initial_orientation_from_heading_pitch(position, np.pi / 2.0, 0.0)
    mover = SurfaceLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=np.array([0.0, 200.0, 0.0]),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
    )
    controller = SurfaceLaunchedCruiseMissileController(
        cruise_speed=250.0,
        cruise_altitude=1000.0,
        cruise_heading=0.0,
        boost_duration=2.0,
        boost_acceleration=30.0,
        launch_pitch_angle=0.0,
        update_interval=0.05,
    )
    controller.phase = controller.TRANSITION_PHASE

    platform = Platform("missile", mover, controller)
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


def test_surface_launched_cruise_missile_scenario_stops_on_ground_impact_and_logs_event(tmp_path):
    output_path = tmp_path / "surface_cruise_missile.h5"

    result = run_surface_launched_cruise_missile_scenario(
        initial_position_ecef=lla_to_ecef(0.0, 0.0, 10.0),
        cruise_speed=250.0,
        cruise_altitude=1000.0,
        cruise_heading=0.0,
        boost_duration=5.0,
        boost_acceleration=30.0,
        launch_pitch_angle=-np.pi / 2.0,
        t_end=10.0,
        sample_interval=0.1,
        output_path=output_path,
    )

    assert result["engine"].t < 10.0

    with h5py.File(output_path, "r") as h5:
        assert "events" in h5
        topics = [topic.decode("utf-8") if isinstance(topic, bytes) else topic for topic in h5["events"]["topic"][:]]
        assert "ground_impact" in topics


def test_surface_launched_cruise_missile_hdf5_output_contains_orientation_and_body_rates(tmp_path):
    output_path = tmp_path / "surface_cruise_missile_orientation.h5"

    run_surface_launched_cruise_missile_scenario(
        initial_position_ecef=lla_to_ecef(0.0, 0.0, 1000.0),
        cruise_speed=250.0,
        cruise_altitude=1200.0,
        cruise_heading=0.0,
        boost_duration=0.5,
        boost_acceleration=20.0,
        launch_pitch_angle=np.radians(10.0),
        t_end=1.0,
        sample_interval=0.1,
        output_path=output_path,
    )

    with h5py.File(output_path, "r") as h5:
        assert "trajectories" in h5
        group = h5["trajectories"]["surface_cruise_missile"]
        assert "orientation" in group
        assert "body_rates" in group
        assert group["orientation"].shape[1] == 4
        assert group["body_rates"].shape[1] == 3
