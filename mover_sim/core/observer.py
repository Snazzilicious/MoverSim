import csv
import json
from pathlib import Path

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
