# Plotting Requirements

## Scope

- The first version targets `HDF5Logger` outputs as the primary supported input.
- The plotting utilities are a Python API, not a GUI application.
- The first version is centered on single-run visualization.
- The design must remain compatible with dynamic platform registration and future platform disappearance.

## Input Requirements

- The plotting layer shall accept a single HDF5 run root corresponding to the current logger schema:
  - `/trajectories/<platform_id>/time`
  - optional datasets such as `state`, `position`, `velocity`, `lla`, `orientation`, `body_rates`
  - optional `/events/*`
  - `/metadata`
- The plotting layer shall tolerate missing optional datasets per platform.
- The plotting layer shall render successfully when `/events` is absent or when requested event topics are not present.
- The plotting layer shall assume mixed platform shapes are valid:
  - different platforms may expose different optional datasets
  - platforms may begin appearing after simulation start
  - future platform disappearance must not be precluded

## Primary User Experience

- The main API shall be a small functional API.
- The main default output shall be one composite static figure for a single run.
- The utilities shall also provide a lightweight interactive globe view for inspection.
- All platforms in a run shall be plotted by default unless the caller filters them.

## Static Figure Requirements

- The default static figure shall combine:
  - a trajectory view
  - position-vs-time plots
  - velocity-vs-time plots
  - orientation plots when present
  - an event summary area
- Event display in the static figure shall include:
  - timeline markers on time-series plots
  - a text summary panel
- The static figure shall omit panels that have no usable data for the selected run/platforms.
- The static figure shall use an ECEF/globe-oriented trajectory view by default.

## Interactive Globe Requirements

- The interactive globe shall support basic view-and-inspect behavior only:
  - rotate
  - pan/zoom as supported by the backend
  - inspect plotted trajectories and markers
- The first version shall use a minimal wireframe Earth or similarly lightweight globe context.
- The first version does not require:
  - playback controls
  - time scrubbing
  - advanced analysis overlays
  - a full desktop or web GUI

## API Requirements

- The API shall return figure objects for programmatic use and testing.
- The API should separate `load run data` from `render figure` conceptually, even if exposed through a small functional surface.
- The API should support caller filtering by:
  - platform id
  - event topic
  - enabled plot sections

A plausible initial API shape would be:

```python
run = load_hdf5_run(group_or_path)
fig = plot_run_summary(run, platform_ids=None, event_topics=None)
globe = plot_run_globe(run, platform_ids=None, event_topics=None)
```

## Event Requirements

- Event rendering shall be driven from the logger's structured event records:
  - `time`
  - `topic`
  - `platform_id`
  - `payload_json`
- Event display shall be non-fatal:
  - missing events do not block plotting
  - unmatched filters do not block plotting
- The design should preserve room for future event types such as:
  - collisions
  - ground impacts
  - LOS-related events

## Non-Functional Requirements

- The dependency budget should remain light.
- The implementation should prefer common plotting dependencies and avoid a large application framework.
- The plotting utilities should be simple to invoke from scripts.
- The design should be robust to current logger semantics rather than requiring log format changes.

## Out Of Scope For V1

- CSV parity
- multi-run comparison plots
- movie generation
- rich textured globe rendering
- full GUI/dashboard behavior
- advanced overlays for collision volumes, LOS cones, or terrain interaction

## Design Implications From Current Logger Schema

- HDF5 is the right starting point because it already preserves per-platform structure and optional fields cleanly.
- The plotting layer should treat each platform as an independent trajectory source with optional kinematic/attitude channels.
- The event panel should be topic-aware but not tightly coupled to any one scenario.

## Remaining Decisions Worth Formalizing Later

- Exact figure layout for the composite summary
- Naming and signatures of the top-level plotting functions
- Whether loaders accept only `h5py.Group` or also file paths
- Whether the interactive globe backend should be `matplotlib` 3D, `plotly`, or another lightweight option
