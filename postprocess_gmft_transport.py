"""Recover transport and plots from saved winners without rerunning gMFT.

Keeps the original manifest/source archive and saves a separate record of
the postprocessing code. Unstable DSSF points are skipped with warnings.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from gmft_transport_job import configured_output_directories, resistivity_filename

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    for name in ("manifest.json", "winning_saddles.csv", "mean_fields",
                 "data/surface_states_summary_nk24.hdf5"):
        if not (run / name).exists():
            raise FileNotFoundError(run / name)
    lock = (run / ".job.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another job is using {run}")
    config = json.loads((run / "manifest.json").read_text())["config"]
    plot_directory = configured_output_directories(config)[1]
    archive = run / "postprocessing_source"
    archive.mkdir(exist_ok=True)
    names = ("postprocess_gmft_transport.py", "02_compute_tto_structure_factor.py",
             "03_compute_scattering.py", "04_compute_transport.py", "config.py",
             "tto_general_gmft.py", "tto_structure_factor.py", "tto_surface_projection.py",
             "tto_spin_wave.py", "swt_transport_components.py", "plot_gmft_transport.py",
             "plot_gmft_spin_components.py", "plot_gmft_spin_movie.py")
    hashes = {}
    for name in names:
        path = ROOT / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, archive / name)
    (run / "postprocessing_manifest.json").write_text(json.dumps(dict(
        config=config, original_manifest="manifest.json", source_sha256=hashes), indent=2)+"\n")
    previous = run / "data/tto_dynamic_structure_factor.hdf5"
    backup = previous.with_name("tto_dynamic_structure_factor.before_reprocess.hdf5")
    if previous.exists() and not backup.exists():
        shutil.copy2(previous, backup)
    env = dict(os.environ, EIO_TTO_OUTPUT_ROOT=str(run), MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    commands = [
        ("dssf", ["02_compute_tto_structure_factor.py", "--gmft-folder", str(run / "mean_fields"),
            "--general-saddles", str(run), "--j-pm", str(config["J_pm_ratio"]),
            "--fields", *map(str, config["fields"]), "--j-zz-ev", str(config["j_zz_ev"]),
            "--eta-ev", str(config["eta_ev"]), "--dssf-nk", str(config["nk"]),
            "--q-perp-mode", config["q_perp_mode"], "--n-q-perp", str(config["n_q_perp"]),
            "--transfer-chunk-size", str(config["transfer_chunk_size"]),
            "--allow-missing", "--overwrite"]),
        ("scattering", ["03_compute_scattering.py", "--overwrite"]),
        ("transport", ["04_compute_transport.py"]),
        ("plot", [
            "plot_gmft_transport.py", str(run),
            "--output-directory", str(plot_directory),
        ]),
        ("spin_components", ["plot_gmft_spin_components.py", str(run)]),
    ]
    for name, _ in commands:
        marker = run / f"{name}.complete"
        if marker.exists():
            marker.unlink()
    for name, command in commands:
        subprocess.run([sys.executable, "-u", *command], cwd=ROOT, env=env, check=True)
        (run / f"{name}.complete").touch()
    print(
        f"Recovered transport data/logs: {run}\n"
        f"Resistivity plot: {plot_directory / resistivity_filename(config)}\n"
        f"Warnings: {run / 'warnings.txt'}"
    )


if __name__ == "__main__":
    main()
