## HDF5 Group Output Plan

### Goal

Change the scenario APIs and `HDF5Logger` so each scenario writes into a caller-provided `h5py.Group` instead of opening or naming an HDF5 file itself.

### Target Design

- The caller opens the HDF5 file.
- The caller creates a unique group for a single scenario run.
- The scenario function receives that group.
- `HDF5Logger` writes only beneath that group.
- The scenario and logger never create, close, or own the file handle.

### Intended Usage

```python
import h5py

from mover_sim.scenario_surface_launched_cruise_missile import run_surface_launched_cruise_missile_scenario

with h5py.File("runs.h5", "a") as h5:
    run_group = h5.create_group("surface_run_001")
    run_surface_launched_cruise_missile_scenario(
        initial_position_ecef=...,
        cruise_speed=...,
        cruise_altitude=...,
        cruise_heading=...,
        boost_duration=...,
        boost_acceleration=...,
        launch_pitch_angle=...,
        t_end=...,
        sample_interval=...,
        output_group=run_group,
    )
```

### Scope

In scope:
- `HDF5Logger`
- Scenario 1 `run_surface_launched_cruise_missile_scenario(...)`
- Scenario 2 `run_air_launched_cruise_missile_scenario(...)`
- Scenario 3 `run_ballistic_missile_scenario(...)`
- tests, docs, and examples that pass output destinations

Out of scope:
- changing trajectory schema
- changing event topic semantics
- changing platform IDs
- adding support for passing an `h5py.File`

### Required API Changes

#### `HDF5Logger`

Current shape:
- accepts a filesystem path string

New shape:
- accepts `group: h5py.Group`

Recommended constructor shape:

```python
HDF5Logger(
    engine,
    group,
    sample_interval=...,
    include_state=True,
    include_lla=True,
    include_events=True,
    event_topics=None,
    ...,
)
```

Validation rules:
- `group` must be an `h5py.Group`
- the group is treated as the root for exactly one scenario run
- the logger must fail fast if logger-managed child names already exist unless explicit overwrite behavior is added later

#### Scenario functions

Replace the last parameter:
- from `output_path`
- to `output_group`

Affected functions:
- `run_surface_launched_cruise_missile_scenario(...)`
- `run_air_launched_cruise_missile_scenario(...)`
- `run_ballistic_missile_scenario(...)`

New expectations:
- `output_group` must be a caller-created `h5py.Group`
- the scenario writes relative to that group only
- the scenario does not create directories or open files

### Group Layout Contract

Treat the provided group as the run root.

Recommended layout:

```text
<run_group>/
  trajectories/
    <platform_id>/...
  events/...
  attrs...
```

Recommended run-level metadata as attributes on the provided group:
- `scenario_name`
- `created_by`
- `schema_version`

Optional later metadata:
- serialized input parameters
- run timestamp
- git revision

### Collision Policy

Best-fit policy:
- require the provided group to be unused for logger-managed output
- fail if any of these already exist:
  - `trajectories`
  - `events`

Rationale:
- avoids accidental append/merge ambiguity
- keeps one group equal to one run
- leaves naming responsibility to the caller

### Implementation Steps

#### Step 1: Update `HDF5Logger` to accept a group

1. Change the constructor parameter from path-like input to `group`.
2. Validate that `group` is an `h5py.Group`.
3. Remove file-opening logic from the logger.
4. Remove any internal file-close behavior.
5. Update all internal path references to be relative to the provided group.

Verification:
- create a small unit test that passes a subgroup from an open file and verifies child datasets/groups are created beneath it

#### Step 2: Add empty-group / collision checks

1. During logger initialization, inspect the provided group.
2. If `trajectories` or `events` already exist, raise a clear error.
3. Keep the rule simple and strict for the first implementation.

Recommended error shape:
- `ValueError("output_group already contains logger-managed data")`

#### Step 3: Update Scenario 1 API

1. Rename `output_path` to `output_group`.
2. Replace path validation with HDF5 group validation.
3. Remove `output_path.parent.mkdir(...)`.
4. Construct `HDF5Logger(engine, output_group, ...)`.
5. Update the return payload from:
   - `output_path`
   - to `output_group`

#### Step 4: Update Scenario 2 API

Apply the same changes as Step 3 to:
- `run_air_launched_cruise_missile_scenario(...)`

#### Step 5: Update Scenario 3 API

Apply the same changes as Step 3 to:
- `run_ballistic_missile_scenario(...)`

#### Step 6: Add a shared HDF5 group validator

Add a small helper in each module or a shared utility, for example:

```python
def _validate_output_group(output_group):
    if not isinstance(output_group, h5py.Group):
        raise ValueError("output_group must be an h5py.Group")
    return output_group
```

If you want to keep the change minimal, duplicate this helper in the three scenario modules first and refactor later.

#### Step 7: Update tests

Update all scenario tests that currently do this:

```python
output_path = tmp_path / "run.h5"
run_...(..., output_path=output_path)
```

to this:

```python
with h5py.File(output_path, "w") as h5:
    group = h5.create_group("run")
    run_...(..., output_group=group)
```

Add explicit tests for:
1. passing a valid empty subgroup
2. failing on a non-group object
3. failing on a reused group that already contains `trajectories` or `events`
4. writing multiple scenario runs into sibling groups in one file

#### Step 8: Update examples

Update:
- `examples/scenario_surface_launched_cruise_missile.py`
- `examples/scenario_air_launched_cruise_missile.py`
- `examples/scenario_ballistic_missile.py`

Each example should:
1. open an HDF5 file with `h5py.File(..., "a" or "w")`
2. create one or more named groups
3. pass the group into the scenario function
4. keep file lifetime outside the scenario

#### Step 9: Update docs

Update references in:
- `docs/new_scenario_specs.md`
- any examples or guide material that mention `output_path`

Text updates needed:
- replace `output_path` with `output_group`
- explain that each run gets a caller-created group
- provide one short usage example with multiple runs in a shared file

### Test Plan

Minimum tests to add or update:

1. `HDF5Logger` writes beneath a provided subgroup
2. `HDF5Logger` rejects pre-populated logger-managed groups
3. Scenario 1 writes into a caller-created group
4. Scenario 2 writes into a caller-created group
5. Scenario 3 writes into a caller-created group
6. Two sibling run groups in one file both succeed
7. Reusing the same run group twice fails clearly

### Migration Notes

This is a breaking change.

All direct callers must change from:

```python
run_...(..., output_path=tmp_path / "file.h5")
```

to:

```python
with h5py.File(tmp_path / "file.h5", "w") as h5:
    run_group = h5.create_group("run_name")
    run_...(..., output_group=run_group)
```

### Recommended Order of Work

1. Update `HDF5Logger`
2. Add logger-focused tests for group validation and collision behavior
3. Update Scenario 1
4. Update Scenario 2
5. Update Scenario 3
6. Update scenario tests
7. Update example scripts
8. Update docs/specs
9. Run all scenario-focused test files

### Success Criteria

The change is complete when:
- all scenario `run_*` functions accept `output_group`
- `HDF5Logger` writes only beneath the provided group
- no scenario opens or closes an HDF5 file
- multiple runs can coexist in sibling groups of one file
- reused non-empty run groups fail clearly
- updated scenario tests pass
