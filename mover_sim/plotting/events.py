"""Event preparation helpers for plotting."""

from __future__ import annotations

import numpy as np

from mover_sim.plotting.models import PlatformTrack, RunData
from mover_sim.plotting.transforms import filter_events, select_platforms


def summarize_events(
    run: RunData,
    event_topics=None,
    platform_ids=None,
    max_events=None,
) -> list[dict[str, object]]:
    """Return plot-ready event summaries for text-panel rendering."""

    events = sorted(
        filter_events(run, event_topics=event_topics, platform_ids=platform_ids),
        key=lambda event: (event.time, event.topic, event.platform_id or ""),
    )
    if max_events is not None:
        events = events[:max_events]

    summaries = []
    for event in events:
        platform_label = event.platform_id if event.platform_id is not None else "-"
        summaries.append(
            {
                "time": float(event.time),
                "topic": event.topic,
                "platform_id": event.platform_id,
                "payload_json": event.payload_json,
                "label": f"{event.time:8.3f}  {event.topic}  {platform_label}",
            }
        )
    return summaries


def _nearest_sample_index(track: PlatformTrack, event_time: float) -> int | None:
    if track.time.size == 0:
        return None
    return int(np.argmin(np.abs(np.asarray(track.time, dtype=float) - float(event_time))))


def map_events_to_positions(
    run: RunData,
    event_topics=None,
    platform_ids=None,
) -> list[dict[str, object]]:
    """Return event records paired with nearest available platform positions.

    Each returned item always includes the event metadata. When a matching platform track
    with position samples exists, the nearest sample is reported. Otherwise the position-
    related fields are set to ``None``.
    """

    selected_platforms = select_platforms(run, platform_ids=platform_ids)
    mapped_events = []

    for event in filter_events(run, event_topics=event_topics, platform_ids=platform_ids):
        track = None if event.platform_id is None else selected_platforms.get(event.platform_id)
        sample_index = None
        sample_time = None
        position_ecef = None

        if track is not None and track.position_ecef is not None and len(track.position_ecef) > 0:
            sample_index = _nearest_sample_index(track, event.time)
            if sample_index is not None:
                sample_time = float(track.time[sample_index])
                position_ecef = np.asarray(track.position_ecef[sample_index], dtype=float).copy()

        mapped_events.append(
            {
                "time": float(event.time),
                "topic": event.topic,
                "platform_id": event.platform_id,
                "payload_json": event.payload_json,
                "sample_index": sample_index,
                "sample_time": sample_time,
                "position_ecef": position_ecef,
            }
        )

    return mapped_events
