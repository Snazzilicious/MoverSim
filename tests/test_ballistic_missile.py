import h5py
import numpy as np

from mover_sim.core.engine import SimulationEngine
from mover_sim.core.platform import Platform
from mover_sim.math.coordinates import ecef_to_enu, lla_to_ecef
from mover_sim.math.orientation import rotate_vector_by_quaternion
from mover_sim.scenario_ballistic_missile import (
    BallisticMissileController,
    BallisticMissileMover,
    SCENARIO_EVENT_TOPICS,
    SpentStageMover,
    _derive_ascent_azimuth,
    _derive_ascent_program,
    _freeze_spent_stage,
    _orientation_from_ascent_azimuth,
    _spent_stage_has_ground_impact,
    _spawn_spent_stage,
    run_ballistic_missile_scenario,
)


def test_ballistic_ascent_azimuth_points_toward_target_in_local_enu():
    initial_position = np.array(lla_to_ecef(0.0, 0.0, 0.0), dtype=float)
    target_position = np.array(lla_to_ecef(0.0, 0.01, 0.0), dtype=float)

    ascent_azimuth = _derive_ascent_azimuth(initial_position, target_position)

    assert np.isclose(ascent_azimuth, np.pi / 2.0, atol=1e-3)


def test_ballistic_orientation_from_ascent_azimuth_matches_local_enu_command():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    ascent_azimuth = np.pi / 2.0
    ascent_pitch = np.radians(20.0)

    quaternion = _orientation_from_ascent_azimuth(position, ascent_azimuth, ascent_pitch)
    forward_world = rotate_vector_by_quaternion([1.0, 0.0, 0.0], quaternion)
    sample_point = position + forward_world
    east, north, up = ecef_to_enu(sample_point[0], sample_point[1], sample_point[2], 0.0, 0.0, 1000.0)

    expected_forward_enu = np.array([np.cos(ascent_pitch), 0.0, np.sin(ascent_pitch)])
    actual_forward_enu = np.array([east, north, up], dtype=float)
    actual_forward_enu /= np.linalg.norm(actual_forward_enu)

    assert np.allclose(actual_forward_enu, expected_forward_enu, atol=1e-7)


def test_ballistic_missile_mover_uses_13_element_state():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    orientation = _orientation_from_ascent_azimuth(position, 0.0, np.radians(60.0))

    mover = BallisticMissileMover(
        initial_position=position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        stages=[
            {
                "dry_mass": 1000.0,
                "propellant_mass": 500.0,
                "burn_duration": 10.0,
                "thrust": 10000.0,
                "drag_coefficient": 0.1,
                "reference_area": 1.0,
                "separation_delay": 1.0,
            }
        ],
    )
    state = mover.get_state()

    assert mover.get_state_dimension() == 13
    assert state.shape == (13,)
    assert np.allclose(mover.position, state[:3])
    assert np.allclose(mover.velocity, state[3:6])
    assert np.allclose(mover.orientation, state[6:10])
    assert np.allclose(mover.body_rates, state[10:13])


def test_spent_stage_mover_uses_13_element_state():
    position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    orientation = _orientation_from_ascent_azimuth(position, 0.0, np.radians(60.0))

    mover = SpentStageMover(
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


def test_ballistic_stage_1_burnout_and_separation_follow_timing():
    engine = SimulationEngine()
    initial_position = np.array(lla_to_ecef(0.0, 0.0, 0.0), dtype=float)
    target_position = np.array(lla_to_ecef(0.0, 0.01, 0.0), dtype=float)
    ascent_program = _derive_ascent_program(initial_position, target_position, 20000.0)
    orientation = _orientation_from_ascent_azimuth(
        initial_position,
        ascent_program["ascent_azimuth"],
        ascent_program["initial_ascent_pitch"],
    )
    stages = [
        {
            "dry_mass": 1000.0,
            "propellant_mass": 500.0,
            "burn_duration": 10.0,
            "thrust": 10000.0,
            "drag_coefficient": 0.1,
            "reference_area": 1.0,
            "separation_delay": 2.0,
        }
    ]

    mover = BallisticMissileMover(
        initial_position=initial_position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        stages=stages,
    )
    controller = BallisticMissileController(
        ascent_program=ascent_program,
        stages=stages,
        peak_altitude=20000.0,
    )
    platform = Platform("ballistic_missile", mover, controller)
    engine.register_platform(platform)

    controller.update(9.0, engine)
    assert controller.phase == controller.POWERED_ASCENT_STAGE_1

    controller.update(10.0, engine)
    assert controller.phase == controller.STAGE_1_SEPARATION

    controller.update(11.0, engine)
    assert controller.phase == controller.STAGE_1_SEPARATION

    controller.update(12.0, engine)
    assert controller.phase == controller.COAST_BALLISTIC


def test_spent_stage_is_registered_when_spawned_at_separation():
    engine = SimulationEngine()
    initial_position = np.array(lla_to_ecef(0.0, 0.0, 0.0), dtype=float)
    target_position = np.array(lla_to_ecef(0.0, 0.01, 0.0), dtype=float)
    ascent_program = _derive_ascent_program(initial_position, target_position, 20000.0)
    orientation = _orientation_from_ascent_azimuth(
        initial_position,
        ascent_program["ascent_azimuth"],
        ascent_program["initial_ascent_pitch"],
    )
    stage = {
        "dry_mass": 1000.0,
        "propellant_mass": 500.0,
        "burn_duration": 10.0,
        "thrust": 10000.0,
        "drag_coefficient": 0.1,
        "reference_area": 1.0,
        "separation_delay": 2.0,
    }

    mover = BallisticMissileMover(
        initial_position=initial_position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        stages=[stage],
    )
    controller = BallisticMissileController(
        ascent_program=ascent_program,
        stages=[stage],
        peak_altitude=20000.0,
    )
    platform = Platform("ballistic_missile", mover, controller)
    engine.register_platform(platform)

    _spawn_spent_stage(engine, platform, stage, "spent_stage_1")

    assert "spent_stage_1" in engine.platforms


def test_two_stage_ballistic_missile_reaches_ballistic_coast_after_final_burnout():
    engine = SimulationEngine()
    initial_position = np.array(lla_to_ecef(0.0, 0.0, 0.0), dtype=float)
    target_position = np.array(lla_to_ecef(0.0, 0.01, 0.0), dtype=float)
    ascent_program = _derive_ascent_program(initial_position, target_position, 40000.0)
    orientation = _orientation_from_ascent_azimuth(
        initial_position,
        ascent_program["ascent_azimuth"],
        ascent_program["initial_ascent_pitch"],
    )
    stages = [
        {
            "dry_mass": 1000.0,
            "propellant_mass": 500.0,
            "burn_duration": 10.0,
            "thrust": 10000.0,
            "drag_coefficient": 0.1,
            "reference_area": 1.0,
            "separation_delay": 2.0,
        },
        {
            "dry_mass": 500.0,
            "propellant_mass": 250.0,
            "burn_duration": 5.0,
            "thrust": 8000.0,
            "drag_coefficient": 0.08,
            "reference_area": 0.8,
            "separation_delay": 1.0,
        },
    ]

    mover = BallisticMissileMover(
        initial_position=initial_position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        stages=stages,
    )
    controller = BallisticMissileController(
        ascent_program=ascent_program,
        stages=stages,
        peak_altitude=40000.0,
    )
    platform = Platform("ballistic_missile", mover, controller)
    engine.register_platform(platform)

    controller.update(10.0, engine)
    assert controller.phase == controller.STAGE_1_SEPARATION

    controller.update(12.0, engine)
    assert controller.phase == controller.POWERED_ASCENT_STAGE_2

    controller.update(17.0, engine)
    assert controller.phase == controller.STAGE_2_SEPARATION

    controller.update(18.0, engine)
    assert controller.phase == controller.COAST_BALLISTIC


def test_spent_stage_retains_passive_rigid_body_attitude_propagation():
    engine = SimulationEngine()
    engine.max_step = 0.05

    position = np.array(lla_to_ecef(0.0, 0.0, 10000.0), dtype=float)
    orientation = _orientation_from_ascent_azimuth(position, 0.0, np.radians(45.0))
    mover = SpentStageMover(
        initial_position=position,
        initial_velocity=np.array([0.0, 0.0, -100.0]),
        initial_orientation=orientation.copy(),
        initial_body_rates=np.array([0.1, 0.2, -0.1]),
    )
    platform = Platform("spent_stage", mover)
    engine.register_platform(platform)

    initial_orientation = mover.orientation.copy()
    engine.run(0.5)

    assert not np.allclose(mover.orientation, initial_orientation)


def test_spent_stage_freezes_after_ground_impact_and_remains_registered():
    engine = SimulationEngine()
    position = np.array(lla_to_ecef(0.0, 0.0, -1.0), dtype=float)
    orientation = _orientation_from_ascent_azimuth(position, 0.0, np.radians(45.0))
    mover = SpentStageMover(
        initial_position=position,
        initial_velocity=np.array([10.0, 0.0, -50.0]),
        initial_orientation=orientation,
        initial_body_rates=np.array([0.1, 0.0, 0.0]),
    )
    platform = Platform("spent_stage", mover, properties={"phase": "separated_ballistic_fall", "frozen": False})
    engine.register_platform(platform)

    assert _spent_stage_has_ground_impact(platform)
    _freeze_spent_stage(platform)

    assert "spent_stage" in engine.platforms
    assert platform.properties["frozen"] is True
    assert platform.properties["phase"] == "ground_impact_frozen"
    assert np.allclose(mover.velocity, np.zeros(3))
    assert np.allclose(mover.body_rates, np.zeros(3))


def test_ballistic_scenario_stops_on_active_body_ground_impact_and_logs_event(tmp_path):
    output_path = tmp_path / "ballistic_missile.h5"

    with h5py.File(output_path, "w") as h5:
        result = run_ballistic_missile_scenario(
            initial_position_ecef=lla_to_ecef(0.0, 0.0, -1.0),
            target_position_ecef=lla_to_ecef(0.0, 0.01, 0.0),
            peak_altitude=20000.0,
            stages=[
                {
                    "dry_mass": 1000.0,
                    "propellant_mass": 500.0,
                    "burn_duration": 10.0,
                    "thrust": 10000.0,
                    "drag_coefficient": 0.1,
                    "reference_area": 1.0,
                    "separation_delay": 1.0,
                }
            ],
            t_end=10.0,
            sample_interval=0.1,
            output_group=h5.create_group("run"),
        )

    assert result["engine"].t < 10.0

    with h5py.File(output_path, "r") as h5:
        topics = [topic.decode("utf-8") if isinstance(topic, bytes) else topic for topic in h5["run"]["events"]["topic"][:]]
        assert "active_body_ground_impact" in topics


def test_ballistic_hdf5_output_contains_active_and_spent_stage_groups(tmp_path):
    from mover_sim.core.observer import HDF5Logger

    output_path = tmp_path / "ballistic_groups.h5"
    engine = SimulationEngine()
    initial_position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    target_position = np.array(lla_to_ecef(0.0, 0.01, 0.0), dtype=float)
    ascent_program = _derive_ascent_program(initial_position, target_position, 20000.0)
    orientation = _orientation_from_ascent_azimuth(
        initial_position,
        ascent_program["ascent_azimuth"],
        ascent_program["initial_ascent_pitch"],
    )
    stage = {
        "dry_mass": 1000.0,
        "propellant_mass": 500.0,
        "burn_duration": 10.0,
        "thrust": 10000.0,
        "drag_coefficient": 0.1,
        "reference_area": 1.0,
        "separation_delay": 1.0,
    }

    active_mover = BallisticMissileMover(
        initial_position=initial_position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        stages=[stage],
    )
    active_controller = BallisticMissileController(
        ascent_program=ascent_program,
        stages=[stage],
        peak_altitude=20000.0,
    )
    active_platform = Platform("ballistic_missile", active_mover, active_controller)
    engine.register_platform(active_platform)
    _spawn_spent_stage(engine, active_platform, stage, "spent_stage_1")

    with h5py.File(output_path, "w") as h5:
        HDF5Logger(
            engine,
            h5.create_group("run"),
            sample_interval=0.1,
            include_state=True,
            include_lla=True,
            include_events=True,
            event_topics=SCENARIO_EVENT_TOPICS,
        )
        engine.run(0.1)

    with h5py.File(output_path, "r") as h5:
        assert "trajectories" in h5["run"]
        assert "ballistic_missile" in h5["run"]["trajectories"]
        assert "spent_stage_1" in h5["run"]["trajectories"]

        for group_name in ("ballistic_missile", "spent_stage_1"):
            group = h5["run"]["trajectories"][group_name]
            assert "orientation" in group
            assert "body_rates" in group
            assert group["orientation"].shape[1] == 4
            assert group["body_rates"].shape[1] == 3


def test_ballistic_hdf5_event_table_contains_scenario_topics(tmp_path):
    from mover_sim.core.observer import HDF5Logger

    output_path = tmp_path / "ballistic_events.h5"
    engine = SimulationEngine()
    initial_position = np.array(lla_to_ecef(0.0, 0.0, 1000.0), dtype=float)
    target_position = np.array(lla_to_ecef(0.0, 0.01, 0.0), dtype=float)
    ascent_program = _derive_ascent_program(initial_position, target_position, 20000.0)
    orientation = _orientation_from_ascent_azimuth(
        initial_position,
        ascent_program["ascent_azimuth"],
        ascent_program["initial_ascent_pitch"],
    )
    stage = {
        "dry_mass": 1000.0,
        "propellant_mass": 500.0,
        "burn_duration": 10.0,
        "thrust": 10000.0,
        "drag_coefficient": 0.1,
        "reference_area": 1.0,
        "separation_delay": 1.0,
    }

    active_mover = BallisticMissileMover(
        initial_position=initial_position,
        initial_velocity=np.zeros(3),
        initial_orientation=orientation,
        initial_body_rates=np.zeros(3),
        stages=[stage],
    )
    active_controller = BallisticMissileController(
        ascent_program=ascent_program,
        stages=[stage],
        peak_altitude=20000.0,
    )
    active_platform = Platform("ballistic_missile", active_mover, active_controller)
    engine.register_platform(active_platform)
    spent_platform = _spawn_spent_stage(engine, active_platform, stage, "spent_stage_1")

    with h5py.File(output_path, "w") as h5:
        HDF5Logger(
            engine,
            h5.create_group("run"),
            sample_interval=0.1,
            include_state=True,
            include_lla=True,
            include_events=True,
            event_topics=SCENARIO_EVENT_TOPICS,
        )

        engine.broker.publish("platform_registered", active_platform)
        engine.broker.publish("platform_registered", spent_platform)
        engine.broker.publish("stage_1_burnout", active_platform)
        engine.broker.publish("stage_1_separation", active_platform)
        engine.broker.publish("ballistic_coast_start", active_platform)
        engine.broker.publish("spent_stage_ground_impact", spent_platform)
        engine.broker.publish("active_body_ground_impact", active_platform)
        engine.run(0.1)

    with h5py.File(output_path, "r") as h5:
        topics = [topic.decode("utf-8") if isinstance(topic, bytes) else topic for topic in h5["run"]["events"]["topic"][:]]
        assert "platform_registered" in topics
        assert "stage_1_burnout" in topics
        assert "stage_1_separation" in topics
        assert "ballistic_coast_start" in topics
        assert "spent_stage_ground_impact" in topics
        assert "active_body_ground_impact" in topics
