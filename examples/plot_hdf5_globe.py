import argparse
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.plotting import load_hdf5_run, plot_run_globe


def _load_run(input_path, group_name=None):
    if group_name is None:
        return load_hdf5_run(str(input_path))

    with h5py.File(input_path, "r") as h5:
        return load_hdf5_run(h5[group_name])


def main():
    parser = argparse.ArgumentParser(description="Render a globe-style plot from an HDF5 run")
    parser.add_argument("input", help="Path to an HDF5 file written by HDF5Logger")
    parser.add_argument("--group", help="Run group name when the file contains multiple runs")
    parser.add_argument("--backend", default="auto", choices=["auto", "plotly", "matplotlib"], help="Plot backend")
    parser.add_argument("--output", help="Optional output path. HTML is used for plotly, PNG for matplotlib")
    args = parser.parse_args()

    input_path = Path(args.input)
    run = _load_run(input_path, group_name=args.group)
    globe = plot_run_globe(run, backend=args.backend)

    if hasattr(globe, "write_html"):
        output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_globe.html")
        globe.write_html(str(output_path))
    else:
        output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_globe.png")
        globe.savefig(output_path, dpi=150)

    print(f"Saved globe plot to {output_path}")


if __name__ == "__main__":
    main()
