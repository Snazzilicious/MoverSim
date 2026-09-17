import h5py
import numpy as np

from examples.scenario_surface_launched_cruise_missile import (
    SurfaceLaunchedCruiseMissileController,
    SurfaceLaunchedCruiseMissileMover,
    _initial_orientation_from_heading_pitch,
    run_surface_launched_cruise_missile_scenario,
)
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.math.coordinates import ecef_to_enu, ecef_to_lla, lla_to_ecef


def test_initial_orientation_from_heading_pitch_matches_local_enu_command():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    heading = np.pi / 2.0
    pitch = np.radians(20.0)

    orientation = _initial_orientation_from_heading_pitch(position, heading, pitch)
    forward_world = orientation[:, 0]
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


def test_surface_launched_cruise_missile_mover_uses_fixed_wing_state_and_boost_fields():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    orientation = _initial_orientation_from_heading_pitch(position, 0.0, np.radians(10.0))

    mover = SurfaceLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        boost_thrust=1234.0,
    )
    state = mover.get_state()

    assert mover.get_state_dimension() == 18
    assert state.shape == (18,)
    assert mover.orientation.shape == (3, 3)
    assert np.allclose(mover.orientation, state[mover.get_orientation_slice()].reshape((3, 3)))
    assert np.allclose(state[mover.get_omega_slice()], np.zeros(3))
    assert mover.boost_thrust == 1234.0
    assert mover.boost_active is True


def test_surface_launched_cruise_missile_mover_switches_between_boost_and_fixed_wing_thrust():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    mover = SurfaceLaunchedCruiseMissileMover(
        initial_position=position,
        initial_velocity=np.zeros(3),
        initial_body_rates=np.zeros(3),
        boost_thrust=4321.0,
    )
    forward = np.array([1.0, 0.0, 0.0])

    mover.thrust_cmd = 50.0
    assert np.allclose(mover._thrust_vector(forward), 4321.0 * forward)

    mover.boost_active = False
    assert np.allclose(mover._thrust_vector(forward), 0.5 * mover.max_thrust * forward)


def test_surface_launched_cruise_missile_controller_publishes_boost_events_and_transitions_on_time():
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
        launch_pitch_angle=np.radians(15.0),
    )
    platform = Platform("missile", mover, controller)
    engine.register_platform(platform)

    boost_events = []
    engine.broker.subscribe("boost_start", lambda current_platform: boost_events.append(("start", current_platform.id)))
    engine.broker.subscribe("boost_end", lambda current_platform: boost_events.append(("end", current_platform.id)))

    controller.initialize(engine)

    assert controller.phase == controller.BOOST_PHASE
    assert mover.boost_active is True
    assert boost_events == [("start", "missile")]

    controller.update(1.0, engine)
    assert controller.phase == controller.BOOST_PHASE
    assert mover.boost_active is True
    assert boost_events == [("start", "missile")]

    controller.update(2.0, engine)
    assert controller.phase == controller.CRUISE_PHASE
    assert mover.boost_active is False
    assert boost_events == [("start", "missile"), ("end", "missile")]

    controller.update(3.0, engine)
    assert boost_events == [("start", "missile"), ("end", "missile")]


def test_surface_launched_cruise_missile_cruise_guidance_reduces_heading_and_altitude_error():
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
        boost_duration=0.0,
        launch_pitch_angle=0.0,
        update_interval=0.05,
    )
    controller.phase = controller.CRUISE_PHASE
    controller.t_launch = 0.0
    mover.boost_active = False

    platform = Platform("missile", mover, controller)
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


def test_surface_launched_cruise_missile_scenario_translates_boost_acceleration_to_thrust(tmp_path):
    output_path = tmp_path / "surface_translation.h5"

    with h5py.File(output_path, "w") as h5:
        result = run_surface_launched_cruise_missile_scenario(
            initial_position_ecef=lla_to_ecef(0.0, 0.0, 1000.0),
            cruise_speed=250.0,
            cruise_altitude=1200.0,
            cruise_heading=0.0,
            boost_duration=0.5,
            boost_acceleration=20.0,
            launch_pitch_angle=np.radians(10.0),
            t_end=0.1,
            sample_interval=0.1,
            output_group=h5.create_group("run"),
        )

    mover = result["platform"].mover
    assert np.isclose(mover.boost_thrust, 20.0 * mover.mass)


def test_surface_launched_cruise_missile_hdf5_output_contains_state_and_events(tmp_path):
    output_path = tmp_path / "surface_cruise_missile.h5"

    with h5py.File(output_path, "w") as h5:
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
            output_group=h5.create_group("run"),
        )

    with h5py.File(output_path, "r") as h5:
        assert "trajectories" in h5["run"]
        group = h5["run"]["trajectories"]["surface_cruise_missile"]
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
        assert "boost_start" in topics
        assert "boost_end" in topics


def test_surface_launched_cruise_missile_rejects_non_group_output_target():
    try:
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
            output_group="not-a-group",
        )
    except ValueError as exc:
        assert str(exc) == "group must be an h5py.Group"
    else:
        raise AssertionError("Expected a ValueError for non-group output target")


def test_surface_launched_cruise_missile_rejects_reused_group(tmp_path):
    output_path = tmp_path / "surface_reuse.h5"

    with h5py.File(output_path, "w") as h5:
        group = h5.create_group("run")
        run_surface_launched_cruise_missile_scenario(
            initial_position_ecef=lla_to_ecef(0.0, 0.0, 1000.0),
            cruise_speed=250.0,
            cruise_altitude=1200.0,
            cruise_heading=0.0,
            boost_duration=0.5,
            boost_acceleration=20.0,
            launch_pitch_angle=np.radians(10.0),
            t_end=0.1,
            sample_interval=0.1,
            output_group=group,
        )

        try:
            run_surface_launched_cruise_missile_scenario(
                initial_position_ecef=lla_to_ecef(0.0, 0.0, 1000.0),
                cruise_speed=250.0,
                cruise_altitude=1200.0,
                cruise_heading=0.0,
                boost_duration=0.5,
                boost_acceleration=20.0,
                launch_pitch_angle=np.radians(10.0),
                t_end=0.1,
                sample_interval=0.1,
                output_group=group,
            )
        except ValueError as exc:
            assert str(exc) == "output_group already contains logger-managed data"
        else:
            raise AssertionError("Expected reused output group to fail")


def test_surface_launched_cruise_missile_supports_sibling_groups_in_one_file(tmp_path):
    output_path = tmp_path / "surface_siblings.h5"

    with h5py.File(output_path, "w") as h5:
        run_surface_launched_cruise_missile_scenario(
            initial_position_ecef=lla_to_ecef(0.0, 0.0, 1000.0),
            cruise_speed=250.0,
            cruise_altitude=1200.0,
            cruise_heading=0.0,
            boost_duration=0.5,
            boost_acceleration=20.0,
            launch_pitch_angle=np.radians(10.0),
            t_end=0.1,
            sample_interval=0.1,
            output_group=h5.create_group("run_a"),
        )
        run_surface_launched_cruise_missile_scenario(
            initial_position_ecef=lla_to_ecef(0.0, 0.0, 1000.0),
            cruise_speed=250.0,
            cruise_altitude=1200.0,
            cruise_heading=0.0,
            boost_duration=0.5,
            boost_acceleration=20.0,
            launch_pitch_angle=np.radians(10.0),
            t_end=0.1,
            sample_interval=0.1,
            output_group=h5.create_group("run_b"),
        )

    with h5py.File(output_path, "r") as h5:
        assert "trajectories" in h5["run_a"]
        assert "trajectories" in h5["run_b"]
