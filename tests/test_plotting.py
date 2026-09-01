import h5py
import matplotlib
import numpy as np

from mover_sim.math.coordinates import lla_to_ecef
from mover_sim.plotting import load_hdf5_run, plot_run_globe, plot_run_summary
from mover_sim.plotting.events import map_events_to_positions
from mover_sim.plotting.transforms import compute_speed, filter_events, select_platforms


matplotlib.use("Agg")


def _yaw_quaternion(yaw_degrees):
    yaw_radians = np.radians(yaw_degrees)
    return np.array([
        np.cos(yaw_radians / 2.0),
        0.0,
        0.0,
        np.sin(yaw_radians / 2.0),
    ])


def _write_run_file(
    path,
    *,
    include_events=True,
    include_second_platform=True,
    include_velocity=True,
    include_orientation=True,
    include_lla=True,
):
    alpha_time = np.array([0.0, 1.0, 2.0], dtype=float)
    alpha_lat = np.array([10.0, 10.05, 10.1], dtype=float)
    alpha_lon = np.array([20.0, 20.02, 20.04], dtype=float)
    alpha_alt = np.array([1000.0, 1100.0, 1200.0], dtype=float)
    alpha_x, alpha_y, alpha_z = lla_to_ecef(alpha_lat, alpha_lon, alpha_alt)
    alpha_position = np.column_stack([alpha_x, alpha_y, alpha_z])
    alpha_velocity = np.array([
        [100.0, 10.0, 2.0],
        [105.0, 11.0, 3.0],
        [110.0, 12.0, 4.0],
    ])
    alpha_orientation = np.vstack([
        _yaw_quaternion(0.0),
        _yaw_quaternion(10.0),
        _yaw_quaternion(20.0),
    ])
    alpha_body_rates = np.array([
        [0.0, 0.0, 0.01],
        [0.0, 0.0, 0.02],
        [0.0, 0.0, 0.03],
    ])

    with h5py.File(path, "w") as h5:
        run = h5.create_group("run")
        metadata = run.create_group("metadata")
        metadata.attrs["sample_interval"] = 1.0
        metadata.attrs["created_utc"] = "2026-09-01T00:00:00+00:00"
        metadata.attrs["schema_version"] = "1"

        trajectories = run.create_group("trajectories")
        alpha = trajectories.create_group("alpha")
        alpha.create_dataset("time", data=alpha_time)
        alpha.create_dataset("position", data=alpha_position)
        if include_velocity:
            alpha.create_dataset("velocity", data=alpha_velocity)
        if include_lla:
            alpha.create_dataset("lla", data=np.column_stack([alpha_lat, alpha_lon, alpha_alt]))
        if include_orientation:
            alpha.create_dataset("orientation", data=alpha_orientation)
            alpha.create_dataset("body_rates", data=alpha_body_rates)
        alpha.create_dataset("state", data=np.column_stack([alpha_position, alpha_velocity]))

        if include_second_platform:
            beta_time = np.array([1.0, 2.0], dtype=float)
            beta_lat = np.array([11.0, 11.1], dtype=float)
            beta_lon = np.array([21.0, 21.1], dtype=float)
            beta_alt = np.array([1500.0, 1550.0], dtype=float)
            beta_x, beta_y, beta_z = lla_to_ecef(beta_lat, beta_lon, beta_alt)
            beta_position = np.column_stack([beta_x, beta_y, beta_z])
            beta = trajectories.create_group("beta")
            beta.create_dataset("time", data=beta_time)
            beta.create_dataset("position", data=beta_position)
            beta.create_dataset("state", data=np.column_stack([beta_position, np.zeros((2, 3))]))

        if include_events:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            events = run.create_group("events")
            events.create_dataset("time", data=np.array([0.25, 1.6], dtype=float))
            events.create_dataset("topic", data=np.array(["spawn", "intercept"], dtype=object), dtype=string_dtype)
            events.create_dataset("platform_id", data=np.array(["alpha", "alpha"], dtype=object), dtype=string_dtype)
            events.create_dataset("payload_json", data=np.array(["{}", "{}"], dtype=object), dtype=string_dtype)


def test_load_hdf5_run_loads_valid_run(tmp_path):
    path = tmp_path / "run.h5"
    _write_run_file(path)

    run = load_hdf5_run(path)

    assert sorted(run.platforms) == ["alpha", "beta"]
    assert run.metadata["schema_version"] == "1"
    assert len(run.events) == 2
    assert run.platforms["alpha"].position_ecef.shape == (3, 3)


def test_load_hdf5_run_tolerates_missing_optional_datasets(tmp_path):
    path = tmp_path / "run_missing_optional.h5"
    _write_run_file(
        path,
        include_events=False,
        include_second_platform=False,
        include_velocity=False,
        include_orientation=False,
        include_lla=False,
    )

    run = load_hdf5_run(path)
    track = run.platforms["alpha"]

    assert run.events == []
    assert track.velocity_ecef is None
    assert track.orientation is None
    assert track.body_rates is None
    assert track.lla is None


def test_load_hdf5_run_preserves_multi_platform_time_ranges(tmp_path):
    path = tmp_path / "run_time_ranges.h5"
    _write_run_file(path)

    run = load_hdf5_run(path)

    assert np.allclose(run.platforms["alpha"].time, [0.0, 1.0, 2.0])
    assert np.allclose(run.platforms["beta"].time, [1.0, 2.0])


def test_select_platforms_filters_by_platform_id(tmp_path):
    path = tmp_path / "run_select.h5"
    _write_run_file(path)
    run = load_hdf5_run(path)

    platforms = select_platforms(run, ["beta"])

    assert list(platforms) == ["beta"]


def test_filter_events_filters_by_topic_and_platform(tmp_path):
    path = tmp_path / "run_events.h5"
    _write_run_file(path)
    run = load_hdf5_run(path)

    filtered = filter_events(run, event_topics=["intercept"], platform_ids=["alpha"])

    assert len(filtered) == 1
    assert filtered[0].topic == "intercept"


def test_compute_speed_returns_velocity_magnitude(tmp_path):
    path = tmp_path / "run_speed.h5"
    _write_run_file(path, include_second_platform=False)
    run = load_hdf5_run(path)

    speed = compute_speed(run.platforms["alpha"])

    assert np.allclose(speed, np.linalg.norm(run.platforms["alpha"].velocity_ecef, axis=1))


def test_map_events_to_positions_uses_nearest_platform_sample(tmp_path):
    path = tmp_path / "run_event_positions.h5"
    _write_run_file(path, include_second_platform=False)
    run = load_hdf5_run(path)

    mapped = map_events_to_positions(run, event_topics=["intercept"])

    assert len(mapped) == 1
    assert mapped[0]["sample_index"] == 2
    assert mapped[0]["sample_time"] == 2.0
    assert mapped[0]["position_ecef"].shape == (3,)


def test_plot_run_summary_returns_figure_object(tmp_path):
    path = tmp_path / "run_summary.h5"
    _write_run_file(path)
    run = load_hdf5_run(path)

    figure = plot_run_summary(run)

    assert figure.__class__.__name__ == "Figure"
    assert any(ax.get_title() == "Trajectory (ECEF)" for ax in figure.axes)
    assert any(ax.get_title() == "Orientation vs Time" for ax in figure.axes)


def test_plot_run_summary_omits_empty_panels(tmp_path):
    path = tmp_path / "run_sparse_summary.h5"
    _write_run_file(
        path,
        include_events=False,
        include_second_platform=False,
        include_velocity=False,
        include_orientation=False,
        include_lla=False,
    )
    run = load_hdf5_run(path)

    figure = plot_run_summary(run)
    titles = [ax.get_title() for ax in figure.axes]

    assert "Trajectory (ECEF)" in titles
    assert "Position vs Time" in titles
    assert "Velocity vs Time" not in titles
    assert "Orientation vs Time" not in titles


def test_plot_run_summary_does_not_fail_on_sparse_runs(tmp_path):
    path = tmp_path / "run_sparse.h5"
    _write_run_file(
        path,
        include_events=False,
        include_second_platform=False,
        include_velocity=False,
        include_orientation=False,
        include_lla=False,
    )
    run = load_hdf5_run(path)

    figure = plot_run_summary(run, sections=["trajectory", "position", "velocity", "orientation", "events"])

    assert figure.__class__.__name__ == "Figure"


def test_plot_run_globe_matplotlib_returns_figure(tmp_path):
    path = tmp_path / "run_globe.h5"
    _write_run_file(path)
    run = load_hdf5_run(path)

    figure = plot_run_globe(run, backend="matplotlib")

    assert figure.__class__.__name__ == "Figure"
    assert figure.axes[0].get_title() == "MoverSim Globe View"


def test_plot_run_globe_rejects_invalid_backend(tmp_path):
    path = tmp_path / "run_globe_invalid.h5"
    _write_run_file(path)
    run = load_hdf5_run(path)

    try:
        plot_run_globe(run, backend="bad-backend")
    except ValueError as exc:
        assert str(exc) == "backend must be one of: `auto`, `plotly`, `matplotlib`"
    else:
        raise AssertionError("Expected invalid backend to raise ValueError")
