import csv

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
        include_events=False,
        event_topics=None,
        batch_size=100,
    ):
        self.engine = engine
        self.sample_interval = sample_interval
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
            "args": args,
            "kwargs": kwargs,
        }
        self.buffer_event_record(event_record)

    def _open_once(self):
        if not self._opened:
            self._open()
            self._opened = True

    def _on_sim_start(self, t):
        due_platform_ids = self._get_due_platform_ids(t, force=True)
        if due_platform_ids:
            self._sample_platforms(t, due_platform_ids)
            self._mark_platforms_sampled(due_platform_ids, t)

    def _open(self):
        raise NotImplementedError

    def _sample_platforms(self, t, platform_ids):
        raise NotImplementedError

    def _flush(self):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError


class CSVLogger(BaseTrajectoryLogger):
    """
    Observer that writes the trajectories of all platforms in the simulation to a CSV file.

    Note: the header is fixed at simulation start, so platforms registered later will not
    appear in the CSV output yet.
    """

    def __init__(self, engine, filepath, log_interval=1.0):
        """
        Parameters:
            engine: The SimulationEngine instance.
            filepath: Path to the output CSV file.
            log_interval: Minimum time interval (seconds) between logs.
        """
        self.filepath = filepath
        self.file = None
        self.writer = None
        super().__init__(engine, sample_interval=log_interval)

    def _open(self):
        """Initialize the CSV file and write the header."""
        self.file = open(self.filepath, mode="w", newline="")
        self.writer = csv.writer(self.file)

        # TODO: Support platforms registered after sim_start. The current CSV format fixes
        # columns up front, so dynamically spawned platforms are omitted.
        header = ["time"]
        for plat_id in sorted(self.engine.platforms.keys()):
            header.extend([
                f"{plat_id}_x",
                f"{plat_id}_y",
                f"{plat_id}_z",
                f"{plat_id}_lat",
                f"{plat_id}_lon",
                f"{plat_id}_alt",
                f"{plat_id}_vx",
                f"{plat_id}_vy",
                f"{plat_id}_vz",
            ])
        self.writer.writerow(header)

    def _sample_platforms(self, t, platform_ids):
        """Write the current coordinates and velocities of all platforms to the file."""
        if not self.writer:
            return

        for plat_id in platform_ids:
            plat = self.engine.platforms[plat_id]
            pos = plat.mover.position
            vel = plat.mover.velocity
            lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
            self.buffer_platform_record(
                plat_id,
                {
                    "time": t,
                    "position": [pos[0], pos[1], pos[2]],
                    "lla": [lat, lon, alt],
                    "velocity": [vel[0], vel[1], vel[2]],
                },
            )

        row = [t]
        for plat_id in sorted(self.engine.platforms.keys()):
            plat = self.engine.platforms[plat_id]
            pos = plat.mover.position
            vel = plat.mover.velocity
            lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
            row.extend([
                pos[0],
                pos[1],
                pos[2],
                lat,
                lon,
                alt,
                vel[0],
                vel[1],
                vel[2],
            ])
        self.writer.writerow(row)

    def _flush(self):
        """Flush any buffered CSV output and clear in-memory buffers."""
        if self.file:
            self.file.flush()
        self.pending_records_by_platform.clear()
        self.pending_events.clear()

    def _close(self):
        """Close the CSV file and clear writer state."""
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
