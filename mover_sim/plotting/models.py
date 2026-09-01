"""Normalized plotting data models.

These containers mirror the current ``HDF5Logger`` schema while keeping plotting code
independent from ``h5py`` objects. Optional trajectory datasets remain optional here,
matching the logger behavior for movers that do not expose every field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EventRecord:
    """One event row from the HDF5 ``/events`` group.

    This stays close to the logger schema so loaders can map structured event datasets
    directly into plotting data without reshaping payload content.
    """

    time: float
    topic: str
    platform_id: str | None
    payload_json: str


@dataclass
class PlatformTrack:
    """Per-platform trajectory data loaded from ``/trajectories/<platform_id>``.

    Arrays are stored exactly as loaded from the logger output. No resampling or time
    alignment is performed at this layer.
    """

    platform_id: str
    time: np.ndarray
    state: np.ndarray | None = None
    position_ecef: np.ndarray | None = None
    velocity_ecef: np.ndarray | None = None
    lla: np.ndarray | None = None
    orientation: np.ndarray | None = None
    body_rates: np.ndarray | None = None


@dataclass
class RunData:
    """Normalized representation of one logged simulation run.

    ``metadata`` maps the logger's ``/metadata`` attributes, ``platforms`` maps per-
    platform trajectory groups, and ``events`` contains the optional event table.
    """

    metadata: dict[str, object] = field(default_factory=dict)
    platforms: dict[str, PlatformTrack] = field(default_factory=dict)
    events: list[EventRecord] = field(default_factory=list)
