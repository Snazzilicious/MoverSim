# Plotting Design

## Proposed Design

Implement a small `mover_sim.plotting` package with three layers:

1. `loaders`
2. `transforms`
3. `renderers`

This keeps HDF5 schema handling separate from plotting code and makes CSV support easy to add later.

## Package Shape

- `mover_sim/plotting/__init__.py`
- `mover_sim/plotting/load_hdf5.py`
- `mover_sim/plotting/models.py`
- `mover_sim/plotting/transforms.py`
- `mover_sim/plotting/static.py`
- `mover_sim/plotting/globe.py`
- `mover_sim/plotting/events.py`

## Core Data Model

Define a small normalized in-memory model independent of `h5py`:

```python
RunData
- metadata: dict
- platforms: dict[str, PlatformTrack]
- events: EventTable | None

PlatformTrack
- platform_id: str
- time: np.ndarray
- state: np.ndarray | None
- position_ecef: np.ndarray | None   # Nx3
- velocity_ecef: np.ndarray | None   # Nx3
- lla: np.ndarray | None             # Nx3
- orientation: np.ndarray | None     # Nx4
- body_rates: np.ndarray | None      # Nx3

EventRecord
- time: float
- topic: str
- platform_id: str | None
- payload_json: str
```

Why:
- Matches current HDF5 logger closely.
- Preserves optional fields cleanly.
- Makes dynamic spawn/despawn natural: each platform just has its own time vector.

## Loader Design

`load_hdf5_run(group_or_path) -> RunData`

Behavior:
- Accept either an `h5py.Group` or a file path.
- Read `/metadata`, `/trajectories/*`, and optional `/events`.
- Only materialize datasets that exist.
- Never require identical sample counts across platforms.
- Return empty `events` rather than failing when `/events` is absent.

This is the key compatibility boundary with `HDF5Logger`.

## Transform Layer

Keep plotting math out of the renderers.

Helpers:
- `select_platforms(run, platform_ids=None)`
- `filter_events(run, event_topics=None, platform_ids=None)`
- `compute_speed(track)` from `velocity`
- `ecef_extent(...)` for axis scaling
- `lla_extent(...)` if later needed
- `sample_event_positions(run, events)`:
  - for each event, map event time to nearest available platform sample
  - only when that platform has `position`

This layer also handles:
- missing optional datasets
- dynamic platform start times
- plot-ready label/color assignment

## Static Summary Renderer

`plot_run_summary(run, platform_ids=None, event_topics=None, sections=None) -> matplotlib.figure.Figure`

Default sections:
- 3D ECEF trajectory panel
- position vs time
- velocity vs time
- orientation vs time
- event text panel

Behavior:
- Plot all platforms by default.
- Omit panels with no usable data.
- Add event timeline markers to relevant time-series panels.
- Add a compact text summary panel listing:
  - time
  - topic
  - platform
- Use shared color mapping per platform across all panels.

Layout recommendation:
- `matplotlib` with `GridSpec`
- top-left: 3D trajectory
- top-right: event summary text
- lower rows: time-series panels

This meets the “one composite figure” requirement without becoming a GUI.

## Interactive Globe Renderer

`plot_run_globe(run, platform_ids=None, event_topics=None) -> FigureLike`

Recommendation:
- Use `plotly` for the interactive globe if acceptable.
- Fallback option: `matplotlib` 3D if you want fewer dependencies but much weaker interactivity.

Initial globe behavior:
- wireframe sphere representing Earth
- platform trajectories in ECEF
- optional event markers
- hover labels for platform ids and event topics

Why `plotly` is the strongest fit:
- lightweight enough compared with full GIS stacks
- easy rotate/zoom/inspect
- no GUI framework needed
- exports HTML cleanly if needed later

## Event Rendering

Separate event helpers from the main renderers.

Static:
- vertical markers on time-series axes
- text summary panel
- optional topic filtering

Interactive globe:
- mark event position if position can be inferred
- otherwise omit marker and keep event available in legend/metadata

Important rule:
- missing events never break plotting

## API Surface

Keep the first public API very small:

```python
from mover_sim.plotting import load_hdf5_run, plot_run_summary, plot_run_globe
```

Optionally:

```python
run = load_hdf5_run("output/scenario_ballistic_missile.h5")
fig = plot_run_summary(run)
globe = plot_run_globe(run, event_topics=["intercept"])
```

## Handling Dynamic Spawn/Despawn

This should be explicit in the design:
- each platform is plotted only over its own `time` range
- no assumption that all platforms exist at `t=0`
- no assumption that all platforms persist to run end
- event placement uses nearest sample within that platform’s own track

This is already compatible with current HDF5 per-platform groups.

## Dependencies

Suggested initial stack:
- required: `numpy`, `h5py`
- static plotting: `matplotlib`
- interactive globe: `plotly` if approved

If you want stricter minimalism:
- make `plotly` optional
- keep static summary functional with only `matplotlib`

## Testing Strategy

Add tests for:
- HDF5 loading with mixed optional datasets
- no-events runs
- multi-platform runs with different start times
- summary renderer omitting empty panels
- event filtering behavior
- event-to-position mapping

Focus test assertions on:
- returned figure object exists
- expected axes/panels are present or omitted
- no exception on sparse/mixed data

## Implementation Order

1. Build `RunData` loader for current HDF5 schema
2. Build transform helpers
3. Build static summary figure
4. Build event panel and timeline markers
5. Build interactive globe
6. Add docs/examples

## Open Questions

1. For the static trajectory panel, do you want true 3D ECEF as the default, or a globe-like projection that still looks geospatial but is easier to read in print?
  - "a globe-like projection that still looks geospatial but is easier to read"
2. Is adding `plotly` acceptable for the interactive globe, or should the first version stay entirely within `matplotlib`?
  - adding `plotly` is acceptable, ideally only for the interactive globe
3. Should `plot_run_summary()` also support writing directly to a file path, or should file saving remain the caller’s responsibility?
  - file saving should remain the caller's responsibility
4. For orientation plots, do you want raw quaternion components first, or a derived attitude view such as roll/pitch/yaw when possible?
  - "a derived attitude view such as roll/pitch/yaw"

These are the only design decisions that materially affect the first implementation pass.
