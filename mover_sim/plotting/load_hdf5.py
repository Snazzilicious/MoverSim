"""HDF5 plotting loaders."""

from __future__ import annotations

from os import PathLike

import numpy as np

from mover_sim.plotting.models import EventRecord, PlatformTrack, RunData


TRACK_DATASET_MAP = {
    "state": "state",
    "position": "position_ecef",
    "velocity": "velocity_ecef",
    "lla": "lla",
    "orientation": "orientation",
    "body_rates": "body_rates",
}


def _import_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "load_hdf5_run requires h5py to be installed. "
            "Install it with `pip install h5py`."
        ) from exc
    return h5py


def _is_run_root(group):
    return "trajectories" in group


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_array(dataset):
    values = dataset[...]
    if getattr(values, "dtype", None) is not None and values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    return np.asarray(values)


def _resolve_run_group(group_or_path, h5py):
    if isinstance(group_or_path, h5py.Group):
        return group_or_path, None

    if not isinstance(group_or_path, (str, PathLike)):
        raise TypeError("group_or_path must be a file path or h5py.Group")

    h5_file = h5py.File(group_or_path, "r")
    root = h5_file

    if _is_run_root(root):
        return root, h5_file

    run_group_names = [
        name
        for name, value in root.items()
        if isinstance(value, h5py.Group) and _is_run_root(value)
    ]
    if len(run_group_names) == 1:
        return root[run_group_names[0]], h5_file
    if len(run_group_names) > 1:
        h5_file.close()
        raise ValueError(
            "HDF5 file contains multiple run groups; pass a specific h5py.Group instead"
        )

    h5_file.close()
    raise ValueError("HDF5 file does not contain a plotting-compatible run root")


def _load_metadata(run_group):
    metadata = {}
    if "metadata" not in run_group:
        return metadata

    for key, value in run_group["metadata"].attrs.items():
        metadata[str(key)] = _decode_scalar(value)
    return metadata


def _load_platform_track(platform_id, platform_group):
    if "time" not in platform_group:
        raise ValueError(f"trajectory group '{platform_id}' is missing required dataset 'time'")

    kwargs = {
        "platform_id": platform_id,
        "time": np.asarray(platform_group["time"][...], dtype=float),
    }
    for dataset_name, field_name in TRACK_DATASET_MAP.items():
        if dataset_name in platform_group:
            kwargs[field_name] = np.asarray(platform_group[dataset_name][...])
    return PlatformTrack(**kwargs)


def _decode_optional_string(value):
    decoded = _decode_scalar(value)
    if decoded == "":
        return None
    return decoded


def _load_events(run_group):
    if "events" not in run_group:
        return []

    events_group = run_group["events"]
    required = ["time", "topic", "platform_id", "payload_json"]
    missing = [name for name in required if name not in events_group]
    if missing:
        missing_text = ", ".join(repr(name) for name in missing)
        raise ValueError(f"events group is missing required datasets: {missing_text}")

    times = np.asarray(events_group["time"][...], dtype=float)
    topics = _read_array(events_group["topic"])
    platform_ids = _read_array(events_group["platform_id"])
    payloads = _read_array(events_group["payload_json"])

    event_count = len(times)
    for name, values in {
        "topic": topics,
        "platform_id": platform_ids,
        "payload_json": payloads,
    }.items():
        if len(values) != event_count:
            raise ValueError(
                f"events dataset '{name}' has length {len(values)}, expected {event_count}"
            )

    return [
        EventRecord(
            time=float(times[index]),
            topic=str(_decode_scalar(topics[index])),
            platform_id=_decode_optional_string(platform_ids[index]),
            payload_json=str(_decode_scalar(payloads[index])),
        )
        for index in range(event_count)
    ]


def load_hdf5_run(group_or_path):
    """Load one HDF5 run into a normalized plotting structure.

    Accepts either an `h5py.Group` representing one logger run root or a file path. When
    a file path is provided, the file must either be a run root itself or contain exactly
    one child group that is a run root.

    Example:
        ```python
        from mover_sim.plotting import load_hdf5_run

        run = load_hdf5_run("output/scenario_air_launched_cruise_missile.h5")
        ```
    """

    h5py = _import_h5py()
    run_group, opened_file = _resolve_run_group(group_or_path, h5py)
    try:
        if not _is_run_root(run_group):
            raise ValueError("run group is missing required group 'trajectories'")

        trajectories_group = run_group["trajectories"]
        platforms = {}
        for platform_id in sorted(trajectories_group.keys()):
            platform_group = trajectories_group[platform_id]
            platforms[platform_id] = _load_platform_track(platform_id, platform_group)

        return RunData(
            metadata=_load_metadata(run_group),
            platforms=platforms,
            events=_load_events(run_group),
        )
    finally:
        if opened_file is not None:
            opened_file.close()
