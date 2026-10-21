#!/usr/bin/env python3
"""Generate the SWT transport figure from a downloaded cluster run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from swt_transport_job import resistivity_filename


# Edit this path to the downloaded run directory, or its data/ subdirectory.
DOWNLOADED_RESULTS = Path("~/Desktop/research/SWT_transport_q/run_11")

REQUIRED_DATA_FILES = (
    "tto_spin_wave_transport_summary.csv",
    "tto_spin_wave_scattering.hdf5",
)


def resolve_run_directory(path):
    """Accept either a downloaded run directory or its data subdirectory."""
    path = path.expanduser().resolve()
    if path.is_dir() and path.name == "data":
        run_directory = path.parent
    elif (path / "data").is_dir():
        run_directory = path
    else:
        raise ValueError(
            f"{path} is not a run directory containing data/, or a data directory"
        )

    data_directory = run_directory / "data"
    missing = [
        filename
        for filename in REQUIRED_DATA_FILES
        if not (data_directory / filename).is_file()
    ]
    if missing:
        raise ValueError(
            "Downloaded data is missing required files: " + ", ".join(missing)
        )
    return run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=DOWNLOADED_RESULTS,
        help="downloaded run directory; overrides DOWNLOADED_RESULTS",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="override the plot_root saved in the run manifest",
    )
    args = parser.parse_args()

    try:
        run_directory = resolve_run_directory(args.results)
    except ValueError as error:
        parser.error(str(error))

    project_root = Path(__file__).resolve().parent
    plotting_script = project_root / "05_plot_transport.py"
    if not plotting_script.is_file():
        parser.error(f"Plotting script does not exist: {plotting_script}")

    environment = os.environ.copy()
    environment["EIO_TTO_OUTPUT_ROOT"] = str(run_directory)
    manifest_path = run_directory / "manifest.json"
    if manifest_path.is_file():
        config = json.loads(manifest_path.read_text())["config"]
        plot_mode = config.get("resistivity_plot_mode", "rows")
        movie_fps = float(config.get("resistivity_movie_fps", 2.0))
        figure_name = resistivity_filename(config)
        output_directory = (
            args.output_directory
            if args.output_directory is not None
            else Path(config["plot_root"])
        )
    else:
        plot_mode, movie_fps = "rows", 2.0
        figure_name = "tto_spin_wave_interface_resistivity.pdf"
        output_directory = args.output_directory or run_directory
    output_directory = output_directory.expanduser().resolve()
    command = [
        sys.executable,
        str(plotting_script),
        "--theory",
        "spin-wave",
        "--plot-mode",
        plot_mode,
        "--movie-fps",
        str(movie_fps),
        "--output-directory",
        str(output_directory),
        "--output-name",
        figure_name,
    ]
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"Plot generation failed with exit code {completed.returncode}"
        )

    figure = output_directory / figure_name
    if not figure.is_file() or figure.stat().st_size == 0:
        raise SystemExit(f"Expected figure was not created: {figure}")
    print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
