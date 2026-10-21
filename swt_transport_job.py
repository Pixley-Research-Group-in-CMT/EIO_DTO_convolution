"""Restartable SWT DSSF, interface transport, and resistivity plotting job."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def configured_output_directories(config):
    """Resolve the data/log run directory and shared plot root."""
    run_label = str(config.get("run_label", "")).strip()
    if (
        not run_label
        or run_label in (".", "..")
        or Path(run_label).name != run_label
    ):
        raise ValueError("run_label must be one nonempty directory name")
    if "log_root" not in config or "plot_root" not in config:
        raise ValueError("run_label requires log_root and plot_root")
    run = Path(config["log_root"]).expanduser().resolve() / run_label
    plots = Path(config["plot_root"]).expanduser().resolve()
    return run, plots


def resistivity_filename(config):
    """Name the selected resistivity output after the run."""
    suffix = ".gif" if config.get("resistivity_plot_mode", "rows") == "movie" else ".pdf"
    return f"{str(config['run_label']).strip()}{suffix}"


def nominal_phi_angles(config):
    """Return the endpoint-inclusive nominal grid with ``n_phi`` intervals."""
    import numpy as np

    return np.linspace(
        float(config["phi_start_degrees"]),
        float(config["phi_stop_degrees"]),
        int(config["n_phi"]) + 1,
    )


def validate(config):
    """Validate the user-editable SWT transport configuration."""
    import numpy as np
    from swt_transport_components import TRANSPORT_COMPONENTS
    from tto_surface_projection import validate_q_perp_sampling

    if config.get("resistivity_plot_mode", "rows") not in (
        "overlay", "rows", "movie"
    ):
        raise ValueError(
            "resistivity_plot_mode must be 'overlay', 'rows', or 'movie'"
        )
    movie_fps = config.get("resistivity_movie_fps", 2.0)
    if isinstance(movie_fps, bool) or not np.isfinite(movie_fps) or movie_fps <= 0:
        raise ValueError("resistivity_movie_fps must be finite and positive")
    if config["transport_component"] not in TRANSPORT_COMPONENTS:
        raise ValueError(
            "transport_component must be one of " + ", ".join(TRANSPORT_COMPONENTS)
        )
    for key in ("n_phi", "nk", "n_q_perp"):
        value = config[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(key + " must be a positive integer")
    fields = np.asarray(config["fields"], dtype=float)
    if (
        fields.ndim != 1
        or not len(fields)
        or not np.all(np.isfinite(fields))
        or np.any(fields < 0)
        or len(np.unique(fields)) != len(fields)
    ):
        raise ValueError("fields must be finite, nonnegative, and unique")
    finite_keys = (
        "J_pm_ratio", "J_pmpm_ratio", "g_z", "g_perp",
        "delta1_per_mev", "delta2_per_mev", "theta_degrees",
        "phi_offset_degrees", "phi_start_degrees", "phi_stop_degrees",
        "stability_tilt",
    )
    for key in finite_keys:
        if not np.isfinite(config[key]):
            raise ValueError(key + " must be finite")
    for key in (
        "j_zz_ev", "eta_ev", "spin_length", "regularization",
        "lattice_constant_angstrom",
    ):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    if config["stability_tilt"] < 0:
        raise ValueError("stability_tilt must be nonnegative")
    if config["phi_stop_degrees"] <= config["phi_start_degrees"]:
        raise ValueError("phi_stop_degrees must exceed phi_start_degrees")
    if not isinstance(config.get("reflect_negative_phi", False), bool):
        raise ValueError("reflect_negative_phi must be true or false")
    if config.get("reflect_negative_phi", False):
        angles = nominal_phi_angles(config)
        if np.any(angles < -1e-12) or not np.any(np.isclose(angles, 0.0)):
            raise ValueError(
                "negative-phi reflection requires a nonnegative grid containing zero"
            )
    validate_q_perp_sampling(config["q_perp_mode"], config["n_q_perp"])
    configured_output_directories(config)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "swt_transport_job.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    run, plot_root = configured_output_directories(config)
    run.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    lock = (run / ".job.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another job is already using {run}")

    source_names = (
        "swt_transport_job.py",
        "02_compute_tto_spin_wave_structure_factor.py",
        "03_compute_scattering.py",
        "04_compute_transport.py",
        "05_plot_transport.py",
        "01_generate_eio.py",
        "config.py",
        "eio_tb.py",
        "swt_transport_components.py",
        "tto_spin_wave.py",
        "tto_structure_factor.py",
        "tto_surface_projection.py",
    )
    source_files = [ROOT / name for name in source_names]
    manifest = {
        "config": config,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    manifest_path = run / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run configuration or source changed; choose a new run_label")
    atomic_json(manifest_path, manifest)

    archive = run / "source"
    archive.mkdir(exist_ok=True)
    for path in source_files:
        shutil.copy2(path, archive / path.name)

    data = run / "data"
    data.mkdir(exist_ok=True)
    environment = dict(os.environ, EIO_TTO_OUTPUT_ROOT=str(run), MPLBACKEND="Agg")

    def stage(name, command):
        marker = run / f"{name}.complete"
        if not marker.exists():
            subprocess.run(
                [sys.executable, "-u", *command],
                cwd=ROOT,
                env=environment,
                check=True,
            )
            marker.touch()

    eio = data / "surface_states_summary_nk24.hdf5"
    cached_eio = ROOT / "data" / eio.name
    if not eio.exists() and cached_eio.exists():
        shutil.copy2(cached_eio, eio)
    if not eio.exists():
        stage("eio", [str(ROOT / "01_generate_eio.py"), "--overwrite"])

    reflection = (
        "--reflect-negative-phi"
        if config.get("reflect_negative_phi", False)
        else "--no-reflect-negative-phi"
    )
    stage("dssf", [
        str(ROOT / "02_compute_tto_spin_wave_structure_factor.py"),
        "--j-pm", str(config["J_pm_ratio"]),
        "--j-pmpm", str(config["J_pmpm_ratio"]),
        "--g-z", str(config["g_z"]),
        "--g-perp", str(config["g_perp"]),
        "--delta1-per-mev", str(config["delta1_per_mev"]),
        "--delta2-per-mev", str(config["delta2_per_mev"]),
        "--transport-component", config["transport_component"],
        "--fields", *map(str, config["fields"]),
        "--theta", str(config["theta_degrees"]),
        "--phi-offset", str(config["phi_offset_degrees"]),
        "--phi-start", str(config["phi_start_degrees"]),
        "--phi-stop", str(config["phi_stop_degrees"]),
        # The numerical stage counts points; the JSON counts intervals like gMFT.
        "--n-phi", str(int(config["n_phi"]) + 1),
        reflection,
        "--j-zz-ev", str(config["j_zz_ev"]),
        "--eta-ev", str(config["eta_ev"]),
        "--dssf-nk", str(config["nk"]),
        "--q-perp-mode", config["q_perp_mode"],
        "--n-q-perp", str(config["n_q_perp"]),
        "--stability-tilt", str(config["stability_tilt"]),
        "--spin-length", str(config["spin_length"]),
        "--regularization", str(config["regularization"]),
        "--lattice-constant", str(config["lattice_constant_angstrom"]),
        "--overwrite",
    ])
    stage("scattering", [
        str(ROOT / "03_compute_scattering.py"),
        "--theory", "spin-wave", "--overwrite",
    ])
    stage("transport", [
        str(ROOT / "04_compute_transport.py"), "--theory", "spin-wave",
    ])
    stage("plot", [
        str(ROOT / "05_plot_transport.py"),
        "--theory", "spin-wave",
        "--plot-mode", config.get("resistivity_plot_mode", "rows"),
        "--movie-fps", str(config.get("resistivity_movie_fps", 2.0)),
        "--output-directory", str(plot_root),
        "--output-name", resistivity_filename(config),
    ])

    print(
        f"Completed data/logs: {run}\n"
        f"Completed resistivity output: {plot_root / resistivity_filename(config)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
