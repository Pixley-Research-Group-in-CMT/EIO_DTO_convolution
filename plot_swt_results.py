#!/usr/bin/env python3
"""Generate the SWT transport figure from a downloaded cluster run."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


# Edit this path to the downloaded run directory, or its data/ subdirectory.
DOWNLOADED_RESULTS = Path("~/Desktop/research/SWT_transport/run_254")

REQUIRED_DATA_FILES = (
    "tto_spin_wave_transport_summary.csv",
    "tto_spin_wave_scattering.hdf5",
)
FIGURE_NAME = "tto_spin_wave_interface_resistivity.pdf"


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
    command = [
        sys.executable,
        str(plotting_script),
        "--theory",
        "spin-wave",
    ]
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"Plot generation failed with exit code {completed.returncode}"
        )

    figure = run_directory / "figures" / FIGURE_NAME
    if not figure.is_file() or figure.stat().st_size == 0:
        raise SystemExit(f"Expected figure was not created: {figure}")
    print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
