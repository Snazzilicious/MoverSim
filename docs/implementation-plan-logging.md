# Trajectory Logger Refactor Implementation Plan

## Goals

- Refactor `CSVLogger` around a format-agnostic trajectory collection layer.
- Add an `HDF5Logger` for structured, high-fidelity trajectory storage.
- Support dynamic platform registration.
- Support arbitrary-dimensional mover state while preserving useful translational convenience fields.
- Support optional event logging alongside trajectory logging.

## Design Summary

Use a two-layer design:

1. A shared logging core that owns sampling, buffering, broker subscriptions, and record construction.
2. Format-specific sinks for CSV and HDF5.

The shared core should operate on normalized per-platform records rather than a global wide row.

## Proposed Components

### `BaseTrajectoryLogger`

Responsibilities:

- subscribe to broker topics:
  - `sim_start`
  - `position_updated`
  - `sim_end`
  - optionally `platform_registered`
  - optionally user-configured event topics
- enforce `sample_interval`
- track `last_sample_time_by_platform`
- keep per-platform buffered trajectory records
- keep buffered event records
- manage lifecycle hooks:
  - `_open()`
  - `_write_platform_batch(platform_id, records)`
  - `_write_event_batch(events)`
  - `_close()`
- flush remaining buffered records on simulation end

Suggested constructor:

```python
BaseTrajectoryLogger(
    engine,
    sample_interval=1.0,
    include_state=True,
    include_lla=True,
    include_events=True,
    event_topics=None,
    batch_size=100,
)
```

### Record Builder

Represent each sample internally as a normalized dict:

```python
{
    "time": float,
    "platform_id": str,
    "state": np.ndarray,
    "state_dim": int,
    "position": np.ndarray | None,
    "velocity": np.ndarray | None,
    "lla": np.ndarray | None,
    "orientation": np.ndarray | None,
    "body_rates": np.ndarray | None,
}
```

Rules:

- always capture `time`, `platform_id`, `state`, and `state_dim`
- include translational fields only if the mover exposes them
- derive `orientation` and `body_rates` from mover slice helpers when present
- do not hardcode aircraft-specific class checks in the logger

### Event Records

Represent logged events as:

```python
{
    "time": float,
    "topic": str,
    "platform_id": str | None,
    "payload": dict | None,
}
```

Default event topics to support:

- `platform_registered`
- `waypoint_reached`
- `intercept`

Event capture should be optional and configurable.

## CSVLogger Design

Refactor `CSVLogger` into a long/table format logger.

### Output Shape

Use one row per platform sample, not one wide row for the entire simulation state.

Recommended columns:

- always:
  - `time`
  - `platform_id`
  - `state_dim`
- when available:
  - `x`, `y`, `z`
  - `lat`, `lon`, `alt`
  - `vx`, `vy`, `vz`
  - `qw`, `qx`, `qy`, `qz`
  - `p`, `q`, `r`
- for arbitrary state:
  - `state_json`

### CSV Decisions

- Keep CSV analysis-friendly and flat.
- Preserve full arbitrary state via serialized `state_json`.
- Do not attempt a dynamically widening `state_0 ... state_n` schema in the first pass.
- Write events to a separate CSV file if event logging is enabled.

Suggested constructor:

```python
CSVLogger(
    engine,
    filepath,
    sample_interval=1.0,
    include_events=False,
    events_filepath=None,
    batch_size=100,
)
```

## HDF5Logger Design

Add an `HDF5Logger` as the structured, high-fidelity sink.

### File Layout

```text
/trajectories
    /<platform_id>
        time           (N,)
        state          (N, D)
        position       (N, 3) optional
        velocity       (N, 3) optional
        lla            (N, 3) optional
        orientation    (N, 4) optional
        body_rates     (N, 3) optional
        attrs:
            state_dim
            schema_version
            field_descriptions
/events
    time               (M,)
    topic              (M,)
    platform_id        (M,)
    payload_json       (M,)
/metadata
    attrs:
        sample_interval
        created_utc
        mover_sim_version
```

### HDF5 Decisions

- Create platform datasets lazily on first sample.
- Use appendable, chunked datasets.
- Resize in batches rather than per row.
- Store arbitrary state natively in numeric datasets.
- Store events as a separate appendable table.

Suggested constructor:

```python
HDF5Logger(
    engine,
    filepath,
    sample_interval=1.0,
    include_events=True,
    batch_size=100,
    compression="gzip",
    compression_level=4,
)
```

## Step-by-Step Implementation Plan

### Phase 1: Refactor the observer module structure

1. Introduce a shared logger base class in `mover_sim/core/observer.py`.
2. Move broker subscription and lifecycle handling out of `CSVLogger` into the base class.
3. Add per-platform sampling time tracking.
4. Add per-platform buffered record storage.
5. Add buffered event storage.

### Phase 2: Build normalized record extraction

1. Add a private helper that extracts a normalized record from a platform at time `t`.
2. Always capture full state and state dimension.
3. Capture translational views only when exposed by the mover.
4. Capture orientation and body rates when slice helpers are present.
5. Derive `lat/lon/alt` only when position is available.

### Phase 3: Rebuild `CSVLogger`

1. Replace the current wide-row header model with one row per platform sample.
2. Emit stable columns for common translational and aircraft-derived fields.
3. Serialize arbitrary full state into `state_json`.
4. Support dynamic platform registration without header rewrites.
5. Optionally add a separate event CSV output.

### Phase 4: Add `HDF5Logger`

1. Add optional `h5py`-backed logging support in `mover_sim/core/observer.py`.
2. Create `/trajectories/<platform_id>` groups lazily.
3. Add appendable datasets for time and full state.
4. Add optional appendable derived datasets for translational and aircraft-specific fields.
5. Add `/events` storage for logged broker events.
6. Add `/metadata` attributes describing the file schema.

### Phase 5: Add tests

1. Add `CSVLogger` tests for:
   - long-row output format
   - dynamic platform registration
   - mixed mover dimensions
   - translational-only rows
   - orientation/body-rate rows
2. Add `HDF5Logger` tests for:
   - per-platform dataset creation
   - dataset append behavior
   - mixed state dimensions
   - event table creation
   - final flush on `sim_end`
3. Add regression tests ensuring the old translational logging fields remain available for translational movers.

### Phase 6: Documentation and examples

1. Update `docs/user-guide.md` to describe the new logger architecture.
2. Update `docs/architecture.md` with logger layering and HDF5 structure.
3. Add a README example showing `CSVLogger` and `HDF5Logger` usage.
4. Document that CSV is a flat export format and HDF5 is the preferred structured archive format.

## Recommended Implementation Order

1. Shared base logger and record builder.
2. CSV refactor.
3. HDF5 logger.
4. Tests.
5. Docs.

## Non-Goals for the First Pass

- generic plugin field registration
- dynamic CSV widening based on maximum observed state dimension
- schema migration support for older output files
- event payload normalization beyond conservative JSON serialization

## Key Design Choice

Treat CSV as a simple analysis/export format and HDF5 as the structured canonical logging
format.

That avoids forcing arbitrary-dimensional state into an awkward global CSV table while
still preserving an easy human-readable output path.
