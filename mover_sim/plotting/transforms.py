"""Plotting data transforms."""

from __future__ import annotations

import numpy as np

from mover_sim.plotting.models import EventRecord, PlatformTrack, RunData


DEFAULT_PLATFORM_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def select_platforms(run: RunData, platform_ids=None) -> dict[str, PlatformTrack]:
    """Return selected platform tracks in deterministic order.

    When ``platform_ids`` is omitted, all platforms are returned sorted by platform id.
    A missing requested platform id raises ``KeyError``.
    """

    if platform_ids is None:
        return {platform_id: run.platforms[platform_id] for platform_id in sorted(run.platforms)}

    selected = {}
    for platform_id in platform_ids:
        if platform_id not in run.platforms:
            raise KeyError(f"unknown platform_id: {platform_id}")
        selected[platform_id] = run.platforms[platform_id]
    return selected


def filter_events(
    run: RunData,
    event_topics=None,
    platform_ids=None,
) -> list[EventRecord]:
    """Return event records filtered by topic and platform id."""

    topic_filter = set(event_topics) if event_topics is not None else None
    platform_filter = set(platform_ids) if platform_ids is not None else None

    filtered = []
    for event in run.events:
        if topic_filter is not None and event.topic not in topic_filter:
            continue
        if platform_filter is not None and event.platform_id not in platform_filter:
            continue
        filtered.append(event)
    return filtered


def compute_speed(track: PlatformTrack) -> np.ndarray | None:
    """Return speed magnitude for one platform track when velocity is available."""

    if track.velocity_ecef is None:
        return None
    return np.linalg.norm(np.asarray(track.velocity_ecef, dtype=float), axis=1)


def compute_ecef_bounds(platforms: dict[str, PlatformTrack]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return axis-aligned ECEF bounds across the selected platforms."""

    positions = [
        np.asarray(track.position_ecef, dtype=float)
        for track in platforms.values()
        if track.position_ecef is not None and len(track.position_ecef) > 0
    ]
    if not positions:
        return None

    stacked = np.vstack(positions)
    return stacked.min(axis=0), stacked.max(axis=0)


def compute_ecef_equalized_bounds(platforms: dict[str, PlatformTrack]):
    """Return equalized ECEF bounds for 3D plotting.

    The return value is a dict containing ``mins``, ``maxs``, ``center``, and ``radius``.
    ``None`` is returned when no selected platform has position data.
    """

    bounds = compute_ecef_bounds(platforms)
    if bounds is None:
        return None

    mins, maxs = bounds
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    equalized_mins = center - radius
    equalized_maxs = center + radius
    return {
        "mins": equalized_mins,
        "maxs": equalized_maxs,
        "center": center,
        "radius": radius,
    }


def panel_data_availability(platforms: dict[str, PlatformTrack], events=None) -> dict[str, bool]:
    """Report which plot panels have usable data for the selected inputs."""

    tracks = list(platforms.values())
    return {
        "trajectory": any(track.position_ecef is not None and len(track.position_ecef) > 0 for track in tracks),
        "position": any(track.position_ecef is not None and len(track.position_ecef) > 0 for track in tracks),
        "velocity": any(track.velocity_ecef is not None and len(track.velocity_ecef) > 0 for track in tracks),
        "orientation": any(track.orientation is not None and len(track.orientation) > 0 for track in tracks),
        "body_rates": any(track.body_rates is not None and len(track.body_rates) > 0 for track in tracks),
        "events": bool(events),
    }


def assign_platform_colors(platforms: dict[str, PlatformTrack]) -> dict[str, str]:
    """Assign deterministic colors keyed by platform id."""

    color_map = {}
    for index, platform_id in enumerate(sorted(platforms)):
        color_map[platform_id] = DEFAULT_PLATFORM_COLORS[index % len(DEFAULT_PLATFORM_COLORS)]
    return color_map
