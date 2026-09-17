import h5py
import numpy as np

from examples.scenario_ballistic_missile import (
    SCENARIO_EVENT_TOPICS,
    _derive_ascent_azimuth,
    _derive_rocket_controller_kwargs,
    _vertical_launch_orientation,
    run_ballistic_missile_scenario,
)
from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.math.coordinates import ecef_to_enu, lla_to_ecef
from mover_sim.models.aircraft_mover import ExpendedStageMover, RocketController, RocketMover


def _rocket_stage(
    *,
    dry_mass,
    propellant_mass,
    burn_duration,
    thrust,
    drag_coefficient,
    reference_area,
    separation_delay,
    rotational_mass=None,
    angular_damping=None,
    max_steering_moment=5.0e4,
):
    if rotational_mass is None:
        rotational_mass = np.diag([6.0e4, 1.2e5, 1.2e5])
    if angular_damping is None:
        angular_damping = np.array([2.0e4, 4.0e4, 4.0e4])
    return {
        "dry_mass": dry_mass,
        "propellant_mass": propellant_mass,
        "burn_duration": burn_duration,
        "thrust": thrust,
        "drag_coefficient": drag_coefficient,
        "reference_area": reference_area,
        "rotational_mass": rotational_mass,
        "angular_damping": angular_damping,
        "max_thrust": thrust,
        "max_steering_moment": max_steering_moment,
        "separation_delay": separation_delay,
    }


def test_ballistic_ascent_azimuth_points_toward_target_in_local_enu():
    initial_position = np.array(lla_to_ecef(0.0, 0.0, 0.0), dtype=float)
    target_position = np.array(lla_to_ecef(0.0, 0.01, 0.0), dtype=float)

    ascent_azimuth = _derive_ascent_azimuth(initial_position, target_position)

    assert np.isclose(ascent_azimuth, np.pi / 2.0, atol=1e-3)


def test_ballistic_vertical_launch_orientation_points_up_with_azimuth_aligned_roll_reference():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    orientation = _vertical_launch_orientation(position, np.pi / 2.0)

    forward_world = orientation[:, 0]
    right_world = orientation[:, 1]
    up_world = orientation[:, 2]
    sample_forward = position + forward_world
    sample_right = position + right_world
    _, _, forward_up = ecef_to_enu(sample_forward[0], sample_forward[1], sample_forward[2], 0.0, 0.0, 1000.0)
    right_east, right_north, _ = ecef_to_enu(sample_right[0], sample_right[1], sample_right[2], 0.0, 0.0, 1000.0)

    assert np.isclose(forward_up, 1.0, atol=1e-6)
    right_horizontal = np.array([right_east, right_north], dtype=float)
    right_horizontal /= np.linalg.norm(right_horizontal)
    assert np.allclose(right_horizontal, [0.0, -1.0], atol=1e-6)
    assert np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(up_world.reshape(3, 1).T @ up_world.reshape(3, 1)), 1.0, atol=1e-6)


def test_ballistic_controller_kwargs_translation_increases_ascent_shaping_with_peak_altitude():
    initial_position = np.array(lla_to_ecef(0.0, 0.0, 0.0), dtype=float)
    target_position = np.array(lla_to_ecef(0.1, 0.1, 0.0), dtype=float)
    stages = [_rocket_stage(
        dry_mass=1000.0,
        propellant_mass=500.0,
        burn_duration=1.0,
        thrust=10000.0,
        drag_coefficient=0.1,
        reference_area=1.0,
        separation_delay=0.25,
    )]

    low = _derive_rocket_controller_kwargs(initial_position, target_position, 12_000.0, stages)
    high = _derive_rocket_controller_kwargs(initial_position, target_position, 60_000.0, stages)

    assert np.allclose(low["target_position_ecef"], target_position)
    assert np.isclose(low["launch_azimuth"], high["launch_azimuth"])
    assert high["target_ascent_pitch"] > low["target_ascent_pitch"]
    assert high["vertical_rise_time"] > low["vertical_rise_time"]
    assert high["pitch_over_duration"] > low["pitch_over_duration"]
    assert np.isclose(low["separation_delay"], 0.25)


def test_ballistic_rocket_mover_uses_rocket_state_layout():
    position = np.array(lla_to_ecef(0.0, 0.0, 0.0), dtype=float)
    orientation = _vertical_launch_orientation(position, 0.0)
    mover = RocketMover(
        initial_position=position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        stages=[_rocket_stage(
            dry_mass=1000.0,
            propellant_mass=500.0,
            burn_duration=1.0,
            thrust=10000.0,
            drag_coefficient=0.1,
            reference_area=1.0,
            separation_delay=0.25,
        )],
    )
    state = mover.get_state()

    assert mover.get_state_dimension() == 19
    assert state.shape == (19,)
    assert mover.orientation.shape == (3, 3)
    assert np.allclose(mover.orientation, state[mover.get_orientation_slice()].reshape((3, 3)))
    assert np.allclose(state[mover.get_omega_slice()], np.zeros(3))
    assert np.isclose(mover.propellant_mass, 500.0)


def test_rocket_stage_separation_registers_passive_expended_stage_platform():
    engine = SimulationEngine()
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    orientation = _vertical_launch_orientation(position, 0.0)
    stages = [
        _rocket_stage(
            dry_mass=1000.0,
            propellant_mass=500.0,
            burn_duration=1.0,
            thrust=10000.0,
            drag_coefficient=0.1,
            reference_area=1.0,
            separation_delay=0.25,
        ),
        _rocket_stage(
            dry_mass=500.0,
            propellant_mass=250.0,
            burn_duration=1.0,
            thrust=8000.0,
            drag_coefficient=0.08,
            reference_area=0.8,
            separation_delay=0.1,
            rotational_mass=np.diag([3.0e4, 6.0e4, 6.0e4]),
            angular_damping=np.array([1.0e4, 2.0e4, 2.0e4]),
            max_steering_moment=2.5e4,
        ),
    ]
    mover = RocketMover(
        initial_position=position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        stages=stages,
        use_coriolis=False,
    )
    platform = Platform("ballistic_missile", mover)
    engine.register_platform(platform)

    mover._set_propellant_mass_state_in_engine(0.0, engine)
    spent_position = mover.position.copy()
    spent_velocity = mover.velocity.copy()
    mover.separate_stage(engine)

    assert "ballistic_missile_spent_stage_1" in engine.platforms
    spent_stage_platform = engine.platforms["ballistic_missile_spent_stage_1"]
    assert isinstance(spent_stage_platform.mover, ExpendedStageMover)
    assert np.allclose(spent_stage_platform.mover.position, spent_position)
    assert np.allclose(spent_stage_platform.mover.velocity, spent_velocity)
    assert spent_stage_platform.properties["separated_stage_index"] == 0


def test_ballistic_one_stage_scenario_runs_with_rocket_wrapper(tmp_path):
    output_path = tmp_path / "ballistic_one_stage.h5"
    with h5py.File(output_path, "w") as h5:
        result = run_ballistic_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
            target_position_ecef=lla_to_ecef(37.2, -121.8, 0.0),
            peak_altitude=20_000.0,
            stages=[_rocket_stage(
                dry_mass=1000.0,
                propellant_mass=500.0,
                burn_duration=0.2,
                thrust=10000.0,
                drag_coefficient=0.1,
                reference_area=1.0,
                separation_delay=0.1,
            )],
            t_end=0.6,
            sample_interval=0.05,
            output_group=h5.create_group("run"),
        )

    assert isinstance(result["platform"].mover, RocketMover)
    assert isinstance(result["platform"].controller, RocketController)

    with h5py.File(output_path, "r") as h5:
        assert "ballistic_missile" in h5["run"]["trajectories"]
        group = h5["run"]["trajectories"]["ballistic_missile"]
        assert group["state"].shape[1] == 19
        assert "position" in group
        assert "velocity" in group
        assert "lla" in group


def test_ballistic_two_stage_scenario_logs_active_and_spent_stage_trajectories_and_events(tmp_path):
    output_path = tmp_path / "ballistic_two_stage.h5"
    with h5py.File(output_path, "w") as h5:
        run_ballistic_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
            target_position_ecef=lla_to_ecef(37.5, -121.5, 0.0),
            peak_altitude=40_000.0,
            stages=[
                _rocket_stage(
                    dry_mass=1000.0,
                    propellant_mass=500.0,
                    burn_duration=0.2,
                    thrust=10000.0,
                    drag_coefficient=0.1,
                    reference_area=1.0,
                    separation_delay=0.1,
                ),
                _rocket_stage(
                    dry_mass=500.0,
                    propellant_mass=250.0,
                    burn_duration=0.2,
                    thrust=8000.0,
                    drag_coefficient=0.08,
                    reference_area=0.8,
                    separation_delay=0.1,
                    rotational_mass=np.diag([3.0e4, 6.0e4, 6.0e4]),
                    angular_damping=np.array([1.0e4, 2.0e4, 2.0e4]),
                    max_steering_moment=2.5e4,
                ),
            ],
            t_end=1.0,
            sample_interval=0.05,
            output_group=h5.create_group("run"),
        )

    with h5py.File(output_path, "r") as h5:
        trajectories = h5["run"]["trajectories"]
        assert "ballistic_missile" in trajectories
        assert "ballistic_missile_spent_stage_1" in trajectories
        for group_name in ("ballistic_missile", "ballistic_missile_spent_stage_1"):
            group = trajectories[group_name]
            assert "time" in group
            assert "state" in group
            assert "position" in group
            assert "velocity" in group
            assert "lla" in group
            assert group["state"].shape[1] == 19

        topics = [
            topic.decode("utf-8") if isinstance(topic, bytes) else topic
            for topic in h5["run"]["events"]["topic"][:]
        ]
        assert "platform_registered" in topics
        assert "stage_burnout" in topics
        assert "stage_separation" in topics
        assert "ballistic_coast_start" in topics


def test_ballistic_hdf5_event_table_contains_configured_topics_only(tmp_path):
    output_path = tmp_path / "ballistic_events.h5"
    with h5py.File(output_path, "w") as h5:
        run_ballistic_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
            target_position_ecef=lla_to_ecef(37.2, -121.8, 0.0),
            peak_altitude=20_000.0,
            stages=[_rocket_stage(
                dry_mass=1000.0,
                propellant_mass=500.0,
                burn_duration=0.2,
                thrust=10000.0,
                drag_coefficient=0.1,
                reference_area=1.0,
                separation_delay=0.1,
            )],
            t_end=0.6,
            sample_interval=0.05,
            output_group=h5.create_group("run"),
        )

    with h5py.File(output_path, "r") as h5:
        topics = {
            topic.decode("utf-8") if isinstance(topic, bytes) else topic
            for topic in h5["run"]["events"]["topic"][:]
        }
        assert topics <= set(SCENARIO_EVENT_TOPICS)
        assert "stage_burnout" in topics
        assert "ballistic_coast_start" in topics


def test_ballistic_rejects_non_group_output_target():
    try:
        run_ballistic_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
            target_position_ecef=lla_to_ecef(37.2, -121.8, 0.0),
            peak_altitude=20_000.0,
            stages=[_rocket_stage(
                dry_mass=1000.0,
                propellant_mass=500.0,
                burn_duration=0.2,
                thrust=10000.0,
                drag_coefficient=0.1,
                reference_area=1.0,
                separation_delay=0.1,
            )],
            t_end=0.6,
            sample_interval=0.05,
            output_group="not-a-group",
        )
    except ValueError as exc:
        assert str(exc) == "group must be an h5py.Group"
    else:
        raise AssertionError("Expected a ValueError for non-group output target")


def test_ballistic_rejects_reused_group(tmp_path):
    output_path = tmp_path / "ballistic_reuse.h5"
    with h5py.File(output_path, "w") as h5:
        group = h5.create_group("run")
        run_ballistic_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
            target_position_ecef=lla_to_ecef(37.2, -121.8, 0.0),
            peak_altitude=20_000.0,
            stages=[_rocket_stage(
                dry_mass=1000.0,
                propellant_mass=500.0,
                burn_duration=0.2,
                thrust=10000.0,
                drag_coefficient=0.1,
                reference_area=1.0,
                separation_delay=0.1,
            )],
            t_end=0.2,
            sample_interval=0.05,
            output_group=group,
        )

        try:
            run_ballistic_missile_scenario(
                initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
                target_position_ecef=lla_to_ecef(37.2, -121.8, 0.0),
                peak_altitude=20_000.0,
                stages=[_rocket_stage(
                    dry_mass=1000.0,
                    propellant_mass=500.0,
                    burn_duration=0.2,
                    thrust=10000.0,
                    drag_coefficient=0.1,
                    reference_area=1.0,
                    separation_delay=0.1,
                )],
                t_end=0.2,
                sample_interval=0.05,
                output_group=group,
            )
        except ValueError as exc:
            assert str(exc) == "output_group already contains logger-managed data"
        else:
            raise AssertionError("Expected reused output group to fail")


def test_ballistic_supports_sibling_groups_in_one_file(tmp_path):
    output_path = tmp_path / "ballistic_siblings.h5"
    with h5py.File(output_path, "w") as h5:
        run_ballistic_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
            target_position_ecef=lla_to_ecef(37.2, -121.8, 0.0),
            peak_altitude=20_000.0,
            stages=[_rocket_stage(
                dry_mass=1000.0,
                propellant_mass=500.0,
                burn_duration=0.2,
                thrust=10000.0,
                drag_coefficient=0.1,
                reference_area=1.0,
                separation_delay=0.1,
            )],
            t_end=0.2,
            sample_interval=0.05,
            output_group=h5.create_group("run_a"),
        )
        run_ballistic_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.0, -122.0, 0.0),
            target_position_ecef=lla_to_ecef(37.2, -121.8, 0.0),
            peak_altitude=20_000.0,
            stages=[_rocket_stage(
                dry_mass=1000.0,
                propellant_mass=500.0,
                burn_duration=0.2,
                thrust=10000.0,
                drag_coefficient=0.1,
                reference_area=1.0,
                separation_delay=0.1,
            )],
            t_end=0.2,
            sample_interval=0.05,
            output_group=h5.create_group("run_b"),
        )

    with h5py.File(output_path, "r") as h5:
        assert "trajectories" in h5["run_a"]
        assert "trajectories" in h5["run_b"]
