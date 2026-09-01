# Plotting Implementation Plan

## Goal

Implement an HDF5-first plotting package for `mover_sim` that provides:

- a small Python API
- a default composite static figure for one run
- a lightweight interactive globe view
- compatibility with the current `HDF5Logger` schema
- support for dynamic platform appearance and optional event data

## Phase 1: Establish Package Skeleton

1. Create a new package directory at `mover_sim/plotting/`.
2. Add initial module files:
   - `__init__.py`
   - `models.py`
   - `load_hdf5.py`
   - `transforms.py`
   - `events.py`
   - `static.py`
   - `globe.py`
3. In `__init__.py`, export only the intended public API surface:
   - `load_hdf5_run`
   - `plot_run_summary`
   - `plot_run_globe`
4. Keep the module layout minimal and avoid adding extra abstraction until real reuse appears.

## Phase 2: Define Normalized In-Memory Models

1. In `models.py`, define lightweight containers for:
   - `RunData`
   - `PlatformTrack`
   - `EventRecord`
2. Represent optional logger outputs directly with nullable fields rather than adapter subclasses.
3. Store per-platform arrays exactly as loaded, without forcing resampling or time alignment.
4. Keep event records close to the HDF5 schema:
   - `time`
   - `topic`
   - `platform_id`
   - `payload_json`
5. Add brief docstrings that describe how these models map to `HDF5Logger` output.

## Phase 3: Implement HDF5 Loader

1. In `load_hdf5.py`, implement `load_hdf5_run(group_or_path)`.
2. Accept either:
   - a file path
   - an `h5py.Group`
3. Read run metadata from `/metadata` when present.
4. Read all platform groups under `/trajectories`.
5. For each platform, load only datasets that exist:
   - `time`
   - `state`
   - `position`
   - `velocity`
   - `lla`
   - `orientation`
   - `body_rates`
6. Read `/events` if present and produce a list of `EventRecord` values.
7. If `/events` is absent, return an empty event collection instead of failing.
8. Avoid imposing any assumption that all platforms share the same sample count or time range.
9. Add validation only for truly required structure:
   - missing `/trajectories` should fail clearly
   - missing per-platform `time` should fail clearly
10. Keep loader errors precise and schema-focused.

## Phase 4: Add Transform Helpers

1. In `transforms.py`, implement helpers for selecting and preparing plot data.
2. Add `select_platforms(run, platform_ids=None)`.
3. Add `filter_events(run, event_topics=None, platform_ids=None)`.
4. Add `compute_speed(track)` using the magnitude of `velocity` when available.
5. Add extent helpers for ECEF plotting so axes can be scaled consistently.
6. Add logic to determine whether each static panel has usable data.
7. Keep platform color assignment deterministic across renderers.
8. Do not resample trajectories in the first version.

## Phase 5: Implement Event Utilities

1. In `events.py`, implement helpers for event display preparation.
2. Add a function to summarize events for text-panel rendering.
3. Add a function to map events to platform positions when possible.
4. Use nearest available sample time within the referenced platform track.
5. If an event has no platform id or the platform has no position data, return no marker position rather than failing.
6. Keep event-topic filtering centralized here or in `transforms.py` so static and globe renderers behave identically.

## Phase 6: Build Static Summary Renderer

1. In `static.py`, implement `plot_run_summary(run, platform_ids=None, event_topics=None, sections=None)`.
2. Use `matplotlib` and `GridSpec` for layout.
3. Build the default composite figure with these candidate sections:
   - 3D ECEF trajectory panel
   - position vs time panel
   - velocity vs time panel
   - orientation vs time panel
   - event summary text panel
4. Omit any panel that has no usable data for the selected platforms.
5. Plot all platforms by default.
6. Use consistent per-platform colors across every panel.
7. Add legends only where they improve readability.
8. Add event timeline markers to time-series plots when event data is available.
9. Add a compact event summary panel listing at least:
   - event time
   - topic
   - platform id when present
10. Keep the first layout functional and legible rather than highly polished.

## Phase 7: Implement 3D Trajectory Panel

1. Render trajectories in ECEF coordinates.
2. Use equalized axis scaling or the closest practical equivalent in `matplotlib` 3D.
3. Add simple labels and units.
4. Ensure the trajectory view still works when only one platform has position data.
5. If event positions can be inferred, plot them as markers on the trajectory panel.
6. Keep Earth rendering out of the static panel unless it materially improves readability.

## Phase 8: Implement Position, Velocity, and Orientation Panels

1. Position panel:
   - plot `x`, `y`, `z` versus time when position is available
2. Velocity panel:
   - plot `vx`, `vy`, `vz` versus time when velocity is available
   - consider optionally adding speed if it improves clarity without clutter
3. Orientation panel:
   - first version should plot quaternion components directly
4. If later needed, derived roll/pitch/yaw can be added as a separate enhancement.
5. Ensure these panels handle multiple platforms without breaking legends or colors.

## Phase 9: Build Interactive Globe Renderer

1. In `globe.py`, implement `plot_run_globe(run, platform_ids=None, event_topics=None)`.
2. Use `plotly` if approved; otherwise provide a simpler `matplotlib` 3D fallback.
3. Render a minimal wireframe Earth or equivalent lightweight globe context.
4. Plot platform trajectories in ECEF.
5. Add event markers when event positions can be inferred.
6. Add hover or inspect labels for:
   - platform id
   - event topic
   - event time
7. Limit first-version interactivity to view-and-inspect behavior.
8. Do not add playback controls or animation in the first pass.

## Phase 10: Public API Cleanup

1. Revisit `__init__.py` and ensure only the intended top-level functions are exposed.
2. Make sure each public function has a concise docstring with a minimal usage example.
3. Keep file writing outside the first API unless a clear need emerges during implementation.
4. Ensure optional dependencies fail cleanly with actionable error messages.

## Phase 11: Dependency Updates

1. Add `matplotlib` to dependency metadata if static plotting is a core supported feature.
2. If `plotly` is adopted, decide whether it should be:
   - a required dependency
   - an optional plotting extra
3. Prefer making the interactive globe dependency optional if that keeps the base install lighter.
4. Update any requirements files consistently with the chosen packaging approach.

## Phase 12: Test Coverage

1. Add tests for the loader:
   - loads a valid HDF5 run
   - tolerates missing optional datasets
   - tolerates missing `/events`
   - handles multiple platforms with different start times
2. Add tests for transforms:
   - platform filtering
   - event filtering
   - speed computation
   - event-to-position mapping
3. Add tests for static plotting:
   - returns a figure object
   - omits empty panels
   - does not fail on sparse runs
4. Add tests for globe plotting at the API level only if backend-specific assertions would be too brittle.
5. Prefer small synthetic HDF5 fixtures over large scenario outputs for most tests.

## Phase 13: Documentation and Examples

1. Add a short user-facing plotting section to the docs after the API is working.
2. Provide a minimal example loading an HDF5 run and generating a static summary figure.
3. Provide a second example for the interactive globe if that backend is included.
4. Document expected input schema in terms of current `HDF5Logger` output.
5. Explicitly note that CSV support is out of scope for the first version.

## Phase 14: Verification Pass

1. Run the full test suite.
2. Exercise the new plotting API against one or more existing scenario HDF5 outputs in `output/`.
3. Confirm behavior for:
   - runs with events
   - runs without events
   - multiple platforms
   - optional orientation/body-rate datasets
4. Review figure readability on both dense and sparse runs.
5. Trim any unnecessary abstractions introduced during implementation.

## Recommended Initial Milestone Split

### Milestone 1

- package skeleton
- normalized models
- HDF5 loader
- transform helpers
- loader and transform tests

### Milestone 2

- static summary renderer
- event panel and timeline markers
- static plotting tests

### Milestone 3

- interactive globe
- optional dependency wiring
- examples and docs

## Open Decisions To Resolve Before Coding

1. Whether the interactive globe will use `plotly` or stay within `matplotlib`.
2. Whether `matplotlib` should be a required dependency or a plotting extra.
3. Whether the static trajectory panel should include an Earth surface context or remain a pure ECEF trajectory view.
4. Whether first-version orientation plots should remain raw quaternions only.
