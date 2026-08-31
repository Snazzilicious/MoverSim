import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from mover_sim.math.coordinates import ecef_to_lla


class BaseTrajectoryLogger:
    """
    Shared trajectory logger base.

    This class owns engine broker subscriptions, sampling cadence checks, and buffered
    storage for per-platform trajectory records and optional event records. Concrete
    subclasses implement the output-format-specific open/write/close hooks.
    """

    def __init__(
        self,
        engine,
        sample_interval=1.0,
        include_state=True,
        include_lla=True,
        include_events=False,
        event_topics=None,
        batch_size=100,
    ):
        self.engine = engine
        self.sample_interval = sample_interval
        self.include_state = include_state
        self.include_lla = include_lla
        self.include_events = include_events
        self.event_topics = list(event_topics or [])
        self.batch_size = batch_size

        self.last_sample_time_by_platform = {}
        self.pending_records_by_platform = {}
        self.pending_events = []
        self._event_callbacks = {}
        self._opened = False

        self.engine.broker.subscribe("sim_start", self.on_sim_start)
        self.engine.broker.subscribe("position_updated", self.on_position_updated)
        self.engine.broker.subscribe("sim_end", self.on_sim_end)

        if self.include_events:
            for topic in self.event_topics:
                callback = self._make_event_callback(topic)
                self._event_callbacks[topic] = callback
                self.engine.broker.subscribe(topic, callback)

    def _make_event_callback(self, topic):
        def callback(*args, **kwargs):
            self.on_event(topic, *args, **kwargs)

        return callback

    def _get_due_platform_ids(self, t, force=False):
        due_platform_ids = []
        for platform_id in sorted(self.engine.platforms.keys()):
            last_t = self.last_sample_time_by_platform.get(platform_id, -float("inf"))
            if force or t - last_t >= self.sample_interval - 1e-9:
                due_platform_ids.append(platform_id)
        return due_platform_ids

    def _mark_platforms_sampled(self, platform_ids, t):
        for platform_id in platform_ids:
            self.last_sample_time_by_platform[platform_id] = t

    def buffer_platform_record(self, platform_id, record):
        """Append a trajectory record to the per-platform in-memory buffer."""
        self.pending_records_by_platform.setdefault(platform_id, []).append(record)

    def buffer_event_record(self, event_record):
        """Append an event record to the in-memory event buffer."""
        self.pending_events.append(event_record)

    def get_buffered_platform_records(self, platform_id):
        """Return the current buffered trajectory records for one platform."""
        return self.pending_records_by_platform.get(platform_id, [])

    def get_buffered_events(self):
        """Return the current buffered event records."""
        return self.pending_events

    def on_sim_start(self, t):
        self._open_once()
        self._on_sim_start(t)

    def on_position_updated(self, t):
        due_platform_ids = self._get_due_platform_ids(t)
        if due_platform_ids:
            self._sample_platforms(t, due_platform_ids)
            self._mark_platforms_sampled(due_platform_ids, t)

    def on_sim_end(self, t):
        due_platform_ids = [
            platform_id
            for platform_id in sorted(self.engine.platforms.keys())
            if t > self.last_sample_time_by_platform.get(platform_id, -float("inf")) + 1e-9
        ]
        if due_platform_ids:
            self._sample_platforms(t, due_platform_ids)
            self._mark_platforms_sampled(due_platform_ids, t)
        self._flush()
        self._close()
        self._opened = False

    def on_event(self, topic, *args, **kwargs):
        if not self.include_events:
            return

        event_record = {
            "time": self.engine.t,
            "topic": topic,
            "platform_id": getattr(args[0], "id", None) if args else None,
            "payload": {
                "args": [self._make_json_safe(arg) for arg in args],
                "kwargs": {key: self._make_json_safe(value) for key, value in kwargs.items()},
            },
        }
        self.buffer_event_record(event_record)
        if len(self.pending_events) >= self.batch_size:
            self._write_event_batch(self.pending_events)
            self.pending_events = []

    def _open_once(self):
        if not self._opened:
            self._open()
            self._opened = True

    def _make_json_safe(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [self._make_json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}
        if hasattr(value, "id"):
            return getattr(value, "id")
        return repr(value)

    def _slice_optional_state(self, state, mover, method_name, expected_size):
        if not hasattr(mover, method_name):
            return None

        state_slice = getattr(mover, method_name)()
        values = np.asarray(state[state_slice], dtype=float).copy()
        if values.shape != (expected_size,):
            return None
        return values

    def _extract_platform_record(self, t, platform_id):
        """Build a normalized record dict for one platform sample."""
        platform = self.engine.platforms[platform_id]
        mover = platform.mover

        state = np.asarray(mover.get_state(), dtype=float).copy()
        record = {
            "time": float(t),
            "platform_id": platform_id,
            "state": state if self.include_state else None,
            "state_dim": int(state.size),
            "position": None,
            "velocity": None,
            "lla": None,
            "orientation": None,
            "body_rates": None,
        }

        if hasattr(mover, "position"):
            position = np.asarray(mover.position, dtype=float).copy()
            if position.shape == (3,):
                record["position"] = position

        if hasattr(mover, "velocity"):
            velocity = np.asarray(mover.velocity, dtype=float).copy()
            if velocity.shape == (3,):
                record["velocity"] = velocity

        if self.include_lla and record["position"] is not None:
            pos = record["position"]
            record["lla"] = np.array(ecef_to_lla(pos[0], pos[1], pos[2]), dtype=float)

        record["orientation"] = self._slice_optional_state(state, mover, "get_orientation_slice", 4)
        record["body_rates"] = self._slice_optional_state(state, mover, "get_body_rate_slice", 3)
        return record

    def _on_sim_start(self, t):
        due_platform_ids = self._get_due_platform_ids(t, force=True)
        if due_platform_ids:
            self._sample_platforms(t, due_platform_ids)
            self._mark_platforms_sampled(due_platform_ids, t)

    def _sample_platforms(self, t, platform_ids):
        for platform_id in platform_ids:
            record = self._extract_platform_record(t, platform_id)
            self.buffer_platform_record(platform_id, record)
            if len(self.pending_records_by_platform[platform_id]) >= self.batch_size:
                self._write_platform_batch(platform_id, self.pending_records_by_platform[platform_id])
                self.pending_records_by_platform[platform_id] = []

    def _flush(self):
        for platform_id, records in self.pending_records_by_platform.items():
            if records:
                self._write_platform_batch(platform_id, records)
        self.pending_records_by_platform.clear()

        if self.pending_events:
            self._write_event_batch(self.pending_events)
            self.pending_events = []

    def _open(self):
        raise NotImplementedError

    def _write_platform_batch(self, platform_id, records):
        raise NotImplementedError

    def _write_event_batch(self, events):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError


class CSVLogger(BaseTrajectoryLogger):
    """
    Observer that writes one CSV row per platform sample.

    The output is a long/table format that supports mixed mover dimensions and platforms
    registered after simulation start.
    """

    def __init__(
        self,
        engine,
        filepath,
        log_interval=1.0,
        include_events=False,
        events_filepath=None,
        batch_size=100,
    ):
        """
        Parameters:
            engine: The SimulationEngine instance.
            filepath: Path to the output CSV file.
            log_interval: Minimum time interval (seconds) between logs.
            include_events: If True, write selected broker events to a separate CSV file.
            events_filepath: Optional path for the event CSV file.
            batch_size: Number of buffered records to accumulate before writing.
        """
        self.filepath = filepath
        self.events_filepath = events_filepath or str(Path(filepath).with_suffix(".events.csv"))
        self.file = None
        self.writer = None
        self.events_file = None
        self.events_writer = None
        super().__init__(
            engine,
            sample_interval=log_interval,
            include_events=include_events,
            event_topics=["platform_registered", "waypoint_reached", "intercept"],
            batch_size=batch_size,
        )

    def _open(self):
        """Initialize the CSV file and write the header."""
        self.file = open(self.filepath, mode="w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "time",
            "platform_id",
            "state_dim",
            "x",
            "y",
            "z",
            "lat",
            "lon",
            "alt",
            "vx",
            "vy",
            "vz",
            "qw",
            "qx",
            "qy",
            "qz",
            "p",
            "q",
            "r",
            "state_json",
        ])

        if self.include_events:
            self.events_file = open(self.events_filepath, mode="w", newline="")
            self.events_writer = csv.writer(self.events_file)
            self.events_writer.writerow(["time", "topic", "platform_id", "payload_json"])

    def _write_platform_batch(self, platform_id, records):
        """Write buffered trajectory records in long-row CSV form."""
        if not self.writer:
            return

        for record in records:
            pos = record["position"]
            vel = record["velocity"]
            lla = record["lla"]
            orientation = record["orientation"]
            body_rates = record["body_rates"]
            state_json = json.dumps(record["state"].tolist()) if record["state"] is not None else ""

            self.writer.writerow([
                record["time"],
                record["platform_id"],
                record["state_dim"],
                pos[0] if pos is not None else "",
                pos[1] if pos is not None else "",
                pos[2] if pos is not None else "",
                lla[0] if lla is not None else "",
                lla[1] if lla is not None else "",
                lla[2] if lla is not None else "",
                vel[0] if vel is not None else "",
                vel[1] if vel is not None else "",
                vel[2] if vel is not None else "",
                orientation[0] if orientation is not None else "",
                orientation[1] if orientation is not None else "",
                orientation[2] if orientation is not None else "",
                orientation[3] if orientation is not None else "",
                body_rates[0] if body_rates is not None else "",
                body_rates[1] if body_rates is not None else "",
                body_rates[2] if body_rates is not None else "",
                state_json,
            ])

    def _write_event_batch(self, events):
        """Write buffered event records to the optional event CSV file."""
        if not self.events_writer:
            return

        for event in events:
            self.events_writer.writerow([
                event["time"],
                event["topic"],
                event["platform_id"] if event["platform_id"] is not None else "",
                json.dumps(event["payload"]),
            ])

    def _close(self):
        """Close the CSV files and clear writer state."""
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
            self.writer = None
        if self.events_file:
            self.events_file.flush()
            self.events_file.close()
            self.events_file = None
            self.events_writer = None


class HDF5Logger(BaseTrajectoryLogger):
    """Structured trajectory logger that stores per-platform datasets in an HDF5 file."""

    DEFAULT_EVENT_TOPICS = ["platform_registered", "waypoint_reached", "intercept"]

    def __init__(
        self,
        engine,
        filepath,
        sample_interval=1.0,
        include_events=True,
        batch_size=100,
        compression="gzip",
        compression_level=4,
        include_state=True,
        include_lla=True,
        event_topics=None,
    ):
        """
        Parameters:
            engine: The SimulationEngine instance.
            filepath: Path to the output HDF5 file.
            sample_interval: Minimum time interval (seconds) between logs.
            include_events: If True, write selected broker events to the HDF5 file.
            batch_size: Number of buffered records to accumulate before writing.
            compression: Dataset compression algorithm or None.
            compression_level: Compression level passed to HDF5 when compression is enabled.
            include_state: If True, store the full mover state dataset.
            include_lla: If True, store derived geodetic coordinates when position is available.
            event_topics: Optional list of broker topics to log when include_events is enabled.
        """
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise ImportError("HDF5Logger requires h5py to be installed") from exc

        self.h5py = h5py
        self.filepath = filepath
        self.compression = compression
        self.compression_level = compression_level
        self.file = None
        self.trajectories_group = None
        self.events_group = None
        self.metadata_group = None
        super().__init__(
            engine,
            sample_interval=sample_interval,
            include_state=include_state,
            include_lla=include_lla,
            include_events=include_events,
            event_topics=event_topics or self.DEFAULT_EVENT_TOPICS,
            batch_size=batch_size,
        )

    def _dataset_kwargs(self):
        kwargs = {"chunks": True}
        if self.compression is not None:
            kwargs["compression"] = self.compression
            kwargs["compression_opts"] = self.compression_level
        return kwargs

    def _open(self):
        self.file = self.h5py.File(self.filepath, mode="w")
        self.trajectories_group = self.file.create_group("trajectories")
        self.metadata_group = self.file.create_group("metadata")
        self.metadata_group.attrs["sample_interval"] = self.sample_interval
        self.metadata_group.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        self.metadata_group.attrs["mover_sim_version"] = "unknown"
        self.metadata_group.attrs["schema_version"] = "1"

        if self.include_events:
            self.events_group = self.file.create_group("events")
            string_dtype = self.h5py.string_dtype(encoding="utf-8")
            kwargs = self._dataset_kwargs()
            self.events_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64, **kwargs)
            self.events_group.create_dataset("topic", shape=(0,), maxshape=(None,), dtype=string_dtype, **kwargs)
            self.events_group.create_dataset("platform_id", shape=(0,), maxshape=(None,), dtype=string_dtype, **kwargs)
            self.events_group.create_dataset("payload_json", shape=(0,), maxshape=(None,), dtype=string_dtype, **kwargs)

    def _ensure_platform_group(self, platform_id, records):
        if platform_id in self.trajectories_group:
            return self.trajectories_group[platform_id]

        group = self.trajectories_group.create_group(platform_id)
        first_record = records[0]
        state_dim = first_record["state_dim"]
        group.attrs["state_dim"] = state_dim
        group.attrs["schema_version"] = "1"
        group.attrs["field_descriptions"] = (
            "time, state, optional position, optional velocity, optional lla, "
            "optional orientation, optional body_rates"
        )

        kwargs = self._dataset_kwargs()
        group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64, **kwargs)
        if self.include_state and first_record["state"] is not None:
            group.create_dataset("state", shape=(0, state_dim), maxshape=(None, state_dim), dtype=np.float64, **kwargs)
        if first_record["position"] is not None:
            group.create_dataset("position", shape=(0, 3), maxshape=(None, 3), dtype=np.float64, **kwargs)
        if first_record["velocity"] is not None:
            group.create_dataset("velocity", shape=(0, 3), maxshape=(None, 3), dtype=np.float64, **kwargs)
        if first_record["lla"] is not None:
            group.create_dataset("lla", shape=(0, 3), maxshape=(None, 3), dtype=np.float64, **kwargs)
        if first_record["orientation"] is not None:
            group.create_dataset("orientation", shape=(0, 4), maxshape=(None, 4), dtype=np.float64, **kwargs)
        if first_record["body_rates"] is not None:
            group.create_dataset("body_rates", shape=(0, 3), maxshape=(None, 3), dtype=np.float64, **kwargs)
        return group

    def _append_dataset(self, dataset, values):
        if values.ndim == 1:
            old_size = dataset.shape[0]
            new_size = old_size + values.shape[0]
            dataset.resize((new_size,))
            dataset[old_size:new_size] = values
            return

        old_size = dataset.shape[0]
        new_size = old_size + values.shape[0]
        dataset.resize((new_size, values.shape[1]))
        dataset[old_size:new_size, :] = values

    def _write_platform_batch(self, platform_id, records):
        if not records:
            return

        group = self._ensure_platform_group(platform_id, records)

        times = np.array([record["time"] for record in records], dtype=float)
        self._append_dataset(group["time"], times)

        if "state" in group:
            states = np.vstack([record["state"] for record in records])
            self._append_dataset(group["state"], states)
        if "position" in group:
            positions = np.vstack([record["position"] for record in records])
            self._append_dataset(group["position"], positions)
        if "velocity" in group:
            velocities = np.vstack([record["velocity"] for record in records])
            self._append_dataset(group["velocity"], velocities)
        if "lla" in group:
            llas = np.vstack([record["lla"] for record in records])
            self._append_dataset(group["lla"], llas)
        if "orientation" in group:
            orientations = np.vstack([record["orientation"] for record in records])
            self._append_dataset(group["orientation"], orientations)
        if "body_rates" in group:
            body_rates = np.vstack([record["body_rates"] for record in records])
            self._append_dataset(group["body_rates"], body_rates)

    def _write_event_batch(self, events):
        if not self.events_group or not events:
            return

        times = np.array([event["time"] for event in events], dtype=float)
        topics = np.array([event["topic"] for event in events], dtype=object)
        platform_ids = np.array([
            event["platform_id"] if event["platform_id"] is not None else ""
            for event in events
        ], dtype=object)
        payloads = np.array([json.dumps(event["payload"]) for event in events], dtype=object)

        self._append_dataset(self.events_group["time"], times)
        self._append_dataset(self.events_group["topic"], topics)
        self._append_dataset(self.events_group["platform_id"], platform_ids)
        self._append_dataset(self.events_group["payload_json"], payloads)

    def _close(self):
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
            self.trajectories_group = None
            self.events_group = None
            self.metadata_group = None
