"""Public plotting API for ``mover_sim``.

Example:
    ```python
    from mover_sim.plotting import load_hdf5_run, plot_run_summary

    run = load_hdf5_run("output/scenario_air_launched_cruise_missile.h5")
    fig = plot_run_summary(run)
    ```
"""

from mover_sim.plotting.globe import plot_run_globe
from mover_sim.plotting.load_hdf5 import load_hdf5_run
from mover_sim.plotting.static import plot_run_summary

__all__ = [
    "load_hdf5_run",
    "plot_run_summary",
    "plot_run_globe",
]
