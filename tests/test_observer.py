import csv
import json

import h5py
import numpy as np

from mover_sim.core.engine import SimulationEngine
from mover_sim.core.observer import CSVLogger, HDF5Logger
from mover_sim.core.platform import Platform
from mover_sim.models.aircraft_mover import Aircraft6DOFMover, AircraftMover
from mover_sim.models.spline_mover import AircraftSplineMover
from mover_sim.math.coordinates import enu_to_ecef, lla_to_ecef


def _make_6dof_platform(platform_id="rigid"):
    pos = lla_to_ecef(0.0, 0.0, 1500.0)
    vel = np.array([0.0, 150.0, 0.0])
    mover = Aircraft6DOFMover(pos, vel, area=0.0, use_coriolis=False)
    return Platform(platform_id, mover)


def _make_point_mass_platform(platform_id="point"):
    pos = lla_to_ecef(0.0, 0.0, 1000.0)
    vel = np.array([0.0, 120.0, 0.0])
    mover = AircraftMover(pos, vel)
    return Platform(platform_id, mover)


def _make_aircraft_spline_platform(platform_id="spline"):
    times = [0.0, 5.0, 10.0]
    positions = [
        enu_to_ecef(0.0, 0.0, 1000.0, 0.0, 0.0, 0.0),
        enu_to_ecef(1000.0, 200.0, 1100.0, 0.0, 0.0, 0.0),
        enu_to_ecef(2000.0, 400.0, 1200.0, 0.0, 0.0, 0.0),
    ]
    mover = AircraftSplineMover(times, positions)
    return Platform(platform_id, mover)


def test_csv_logger_long_rows_preserve_translational_fields(tmp_path):
    engine = SimulationEngine()
    engine.max_step = 0.25
    platform = _make_point_mass_platform("point")
    engine.register_platform(platform)

    log_file = tmp_path / "telemetry.csv"
    CSVLogger(engine, str(log_file), log_interval=0.5)

    engine.run(1.0)

    with open(log_file, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 2
    row = rows[0]
    assert row["platform_id"] == "point"
    assert row["state_dim"] == "6"
    assert row["x"] != ""
    assert row["lat"] != ""
    assert row["vx"] != ""
    assert row["qw"] == ""
    assert row["p"] == ""
    assert len(json.loads(row["state_json"])) == 6


def test_csv_logger_supports_dynamic_registration_mixed_dimensions_and_orientation_rows(tmp_path):
    engine = SimulationEngine()
    engine.max_step = 0.25
    engine.register_platform(_make_point_mass_platform("point"))
    rigid_platform = _make_6dof_platform("rigid")

    log_file = tmp_path / "telemetry.csv"
    CSVLogger(engine, str(log_file), log_interval=0.5)

    engine.schedule(0.5, lambda eng: eng.register_platform(rigid_platform), "spawn_rigid")
    engine.run(1.5)

    with open(log_file, newline="") as f:
        rows = list(csv.DictReader(f))

    platform_ids = {row["platform_id"] for row in rows}
    assert {"point", "rigid"}.issubset(platform_ids)

    point_rows = [row for row in rows if row["platform_id"] == "point"]
    rigid_rows = [row for row in rows if row["platform_id"] == "rigid"]

    assert point_rows
    assert rigid_rows
    assert all(row["state_dim"] == "6" for row in point_rows)
    assert all(row["state_dim"] == "13" for row in rigid_rows)
    assert any(row["qw"] != "" for row in rigid_rows)
    assert any(row["p"] != "" for row in rigid_rows)
    assert all(row["qw"] == "" for row in point_rows)


def test_csv_logger_writes_event_csv(tmp_path):
    engine = SimulationEngine()
    engine.max_step = 0.25
    engine.register_platform(_make_point_mass_platform("point"))
    rigid_platform = _make_6dof_platform("rigid")

    log_file = tmp_path / "telemetry.csv"
    event_file = tmp_path / "telemetry.events.csv"
    CSVLogger(
        engine,
        str(log_file),
        log_interval=0.5,
        include_events=True,
        events_filepath=str(event_file),
    )

    engine.schedule(0.5, lambda eng: eng.register_platform(rigid_platform), "spawn_rigid")
    engine.run(1.0)

    with open(event_file, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert any(row["topic"] == "platform_registered" for row in rows)
    assert any(row["platform_id"] == "rigid" for row in rows)


def test_hdf5_logger_creates_per_platform_datasets_and_flushes_final_batch(tmp_path):
    engine = SimulationEngine()
    engine.max_step = 0.25
    platform = _make_point_mass_platform("point")
    engine.register_platform(platform)

    log_file = tmp_path / "telemetry.h5"
    HDF5Logger(engine, str(log_file), sample_interval=0.5, batch_size=100)

    engine.run(1.0)

    with h5py.File(log_file, "r") as h5:
        assert "trajectories" in h5
        assert "point" in h5["trajectories"]
        group = h5["trajectories"]["point"]
        assert group.attrs["state_dim"] == 6
        assert group["time"].shape[0] >= 2
        assert group["state"].shape[0] == group["time"].shape[0]
        assert group["state"].shape[1] == 6
        assert "position" in group
        assert "velocity" in group
        assert "lla" in group


def test_hdf5_logger_supports_mixed_dimensions_and_dynamic_registration(tmp_path):
    engine = SimulationEngine()
    engine.max_step = 0.25
    engine.register_platform(_make_point_mass_platform("point"))
    rigid_platform = _make_6dof_platform("rigid")

    log_file = tmp_path / "telemetry.h5"
    HDF5Logger(engine, str(log_file), sample_interval=0.5, batch_size=2)

    engine.schedule(0.5, lambda eng: eng.register_platform(rigid_platform), "spawn_rigid")
    engine.run(1.5)

    with h5py.File(log_file, "r") as h5:
        point = h5["trajectories"]["point"]
        rigid = h5["trajectories"]["rigid"]

        assert point.attrs["state_dim"] == 6
        assert rigid.attrs["state_dim"] == 13
        assert point["state"].shape[1] == 6
        assert rigid["state"].shape[1] == 13
        assert "orientation" not in point
        assert "body_rates" not in point
        assert "orientation" in rigid
        assert "body_rates" in rigid


def test_hdf5_logger_writes_event_table(tmp_path):
    engine = SimulationEngine()
    engine.max_step = 0.25
    engine.register_platform(_make_point_mass_platform("point"))
    rigid_platform = _make_6dof_platform("rigid")

    log_file = tmp_path / "telemetry.h5"
    HDF5Logger(engine, str(log_file), sample_interval=0.5, include_events=True, batch_size=2)

    engine.schedule(0.5, lambda eng: eng.register_platform(rigid_platform), "spawn_rigid")
    engine.run(1.0)

    with h5py.File(log_file, "r") as h5:
        assert "events" in h5
        events = h5["events"]
        assert events["time"].shape[0] >= 1
        topics = [topic.decode("utf-8") if isinstance(topic, bytes) else topic for topic in events["topic"][:]]
        platform_ids = [pid.decode("utf-8") if isinstance(pid, bytes) else pid for pid in events["platform_id"][:]]
        assert "platform_registered" in topics
        assert "rigid" in platform_ids


def test_hdf5_logger_records_orientation_and_body_rates_for_analytical_aircraft(tmp_path):
    engine = SimulationEngine()
    platform = _make_aircraft_spline_platform("spline")
    engine.register_platform(platform)

    log_file = tmp_path / "telemetry.h5"
    HDF5Logger(engine, str(log_file), sample_interval=0.5, batch_size=10)

    engine.run(1.0)

    with h5py.File(log_file, "r") as h5:
        group = h5["trajectories"]["spline"]
        assert group.attrs["state_dim"] == 13
        assert "orientation" in group
        assert "body_rates" in group
        assert group["orientation"].shape[1] == 4
        assert group["body_rates"].shape[1] == 3
