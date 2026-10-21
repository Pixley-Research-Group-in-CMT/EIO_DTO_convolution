"""Restartable general-gMFT multistart scan followed by interface transport.

One spawned process per SCF trial; numerical libraries use one thread each.
All successful and failed trials are checkpointed atomically. Restarting with
the same configuration skips completed trials and complete transport stages.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import fcntl

ROOT = Path(__file__).resolve().parent


def nominal_phi_angles(config):
    """Return the endpoint-inclusive nominal scan with ``n_phi`` intervals."""
    import numpy as np
    return np.linspace(
        float(config.get("phi_start_degrees", 0.0)),
        float(config.get("phi_stop_degrees", 30.0)),
        int(config.get("n_phi", 30)) + 1,
    )


def phi_label(phi):
    """Return a stable, float-readable label for angle directories."""
    return format(float(phi), ".12g")


def configured_output_directories(config):
    """Resolve the data/log run directory and resistivity plot root.

    Saved manifests using the former explicit directory keys remain supported.
    """
    if "run_label" in config:
        run_label = str(config["run_label"]).strip()
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

    if "run_directory" not in config:
        raise ValueError("configuration requires run_label or run_directory")
    run = Path(config["run_directory"]).expanduser().resolve()
    plots = Path(config.get("plot_directory", run)).expanduser().resolve()
    return run, plots


def resistivity_filename(config):
    """Name the selected resistivity output after the run."""
    suffix = ".gif" if config.get("resistivity_plot_mode", "overlay") == "movie" else ".pdf"
    if "run_label" in config:
        return f"{str(config['run_label']).strip()}{suffix}"
    return "gmft_resistivity" + suffix


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate(config):
    import numpy as np
    if config.get("resistivity_plot_mode", "overlay") not in ("overlay", "movie"):
        raise ValueError("resistivity_plot_mode must be 'overlay' or 'movie'")
    movie_fps = config.get("resistivity_movie_fps", 2.0)
    if isinstance(movie_fps, bool) or not np.isfinite(movie_fps) or movie_fps <= 0:
        raise ValueError("resistivity_movie_fps must be finite and positive")
    if config.get("convergence_criterion", "full") not in ("full", "amplitudes"):
        raise ValueError("convergence_criterion must be 'full' or 'amplitudes'")
    if config["J_pmpm_ratio"] != 0:
        raise ValueError("The general gMFT Ansatz requires J_pmpm=0")
    for key in (
        "nk", "niter", "random_starts", "transfer_chunk_size", "n_q_perp",
        "n_phi",
    ):
        if isinstance(config[key], bool) or int(config[key]) != config[key] or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if not config["fields"] or len(set(config["fields"])) != len(config["fields"]):
        raise ValueError("fields must be nonempty and unique")
    if not np.isfinite(config["fields"]).all() or min(config["fields"]) < 0:
        raise ValueError("fields must be finite nonnegative mu_B |B|/Jzz values")
    for key in ("J_pm_ratio", "g_z", "g_perp", "delta1_per_mev", "delta2_per_mev", "theta_degrees"):
        if not np.isfinite(config[key]):
            raise ValueError(key + " must be finite")
    if not np.isfinite(config.get("phi_offset_degrees", .05)):
        raise ValueError("phi_offset_degrees must be finite")
    phi_start = config["phi_start_degrees"]
    phi_stop = config["phi_stop_degrees"]
    if not np.isfinite([phi_start, phi_stop]).all():
        raise ValueError("phi_start_degrees and phi_stop_degrees must be finite")
    if phi_start < 0 or phi_stop <= phi_start:
        raise ValueError(
            "phi scan must satisfy 0 <= phi_start_degrees < phi_stop_degrees"
        )
    for key in ("j_zz_ev", "eta_ev", "tol", "lambda_tol", "mix"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    if config["mix"] > 1:
        raise ValueError("mix must be <= 1")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    from tto_surface_projection import validate_q_perp_sampling
    validate_q_perp_sampling(config["q_perp_mode"], config["n_q_perp"])
    configured_output_directories(config)


def trial(task):
    config, run, field_index, phi_index, phi, start = task
    import numpy as np
    sys.path.insert(0, config["gmft_source"])
    from gmft_for_qsi.gmft_field_general import solve_gmft, local_zeeman_coefficients
    from gmft_for_qsi.gmft_field import polarized_x_initial_condition
    from gmft_for_qsi.lattice import magnetic_field_direction, gen_k_grid
    directory = (
        Path(run) / "trials" / f"field_{field_index}" / f"phi_{phi_label(phi)}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"start_{start}.json"
    if path.exists():
        return json.loads(path.read_text())
    seed = int(np.random.SeedSequence(
        [config["seed"], field_index, phi_index, start]
    ).generate_state(1)[0])
    evaluated_phi = phi + config.get("phi_offset_degrees", .05)
    row = dict(phi_degrees=phi, evaluated_phi_degrees=evaluated_phi,
               h_over_J_zz=config["fields"][field_index],
               J_pm_over_J_zz=config["J_pm_ratio"], start=start, seed=seed,
               initial_condition="polarized_x" if start == 0 else "random")
    params = dict(J_zz=1., J_pm=config["J_pm_ratio"], J_zpm=0.,
                  g_z=config["g_z"], g_perp=config["g_perp"],
                  delta_1=config["delta1_per_mev"] * config["j_zz_ev"] * 1000,
                  delta_2=config["delta2_per_mev"] * config["j_zz_ev"] * 1000,
                  h_ext=row["h_over_J_zz"] * magnetic_field_direction(
                      theta=config["theta_degrees"], phi=evaluated_phi, degrees=True))
    kwargs = {}
    if start == 0:
        kwargs["initial_s_z"], kwargs["initial_s_minus"] = polarized_x_initial_condition()
    try:
        result = solve_gmft(params, nk=config["nk"], niter=config["niter"],
                            mix=config["mix"], tol=config["tol"],
                            lambda_tol=config["lambda_tol"], random_seed=seed,
                            convergence_criterion=config.get("convergence_criterion", "full"), **kwargs)
        if not result["converged"] or not np.isfinite(result["energy"]):
            raise RuntimeError("SCF did not converge to a finite energy")
        _, k = gen_k_grid(nk=config["nk"])
        condensate = result["phi_condensate_density_matrices"] / len(k)
        occupied = np.trace(condensate, axis1=1, axis2=2).real > 0
        bp, bz = local_zeeman_coefficients(params)
        saddle = directory / f"start_{start}.npz"
        temporary = saddle.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, j_pm_ratio=params["J_pm"],
            lambda_ratio=result["lambda"], s_minus=result["s_minus"], s_z=result["s_z"],
            b_plus=bp, b_z=bz, rho=result["rho"], energy=result["energy"],
            condensate_k=k[occupied],
            condensate_diagonal=np.diagonal(condensate[occupied], axis1=1, axis2=2).real,
            condensate_density_matrices=condensate[occupied],
            initial_s_z=result["initial_s_z"], initial_s_minus=result["initial_s_minus"],
            nk=config["nk"], phi_degrees=phi, evaluated_phi_degrees=evaluated_phi,
            convergence_criterion=result["convergence_criterion"],
            residual_full=result["delta_full"], residual_amplitudes=result["delta_amplitudes"],
            fully_converged=result["converged_full"])
        temporary.replace(saddle)
        row.update(status="ok", energy=result["energy"], iterations=result["iterations_used"],
                   saddle_file=str(saddle.relative_to(run)),
                   convergence_criterion=result["convergence_criterion"],
                   residual_full=result["delta_full"], residual_amplitudes=result["delta_amplitudes"],
                   fully_converged=result["converged_full"])
    except Exception as exc:
        row.update(status="error", message=str(exc))
    atomic_json(path, row)
    return row


def select_winners(rows, fields, phis=None, phi_offset_degrees=.05):
    import numpy as np
    if phis is None:
        phis = np.linspace(0.0, 30.0, 31)
    winners, missing = [], []
    for field in fields:
        for phi in phis:
            phi = float(phi)
            candidates = [
                r for r in rows
                if np.isclose(r["h_over_J_zz"], field)
                and np.isclose(r["phi_degrees"], phi)
                and r["status"] == "ok"
            ]
            if candidates:
                winners.append(min(candidates, key=lambda r: (r["energy"], r["start"])))
            else:
                missing.append((field, phi))
                winners.append(dict(phi_degrees=phi,
                    evaluated_phi_degrees=phi+phi_offset_degrees,
                    h_over_J_zz=field, J_pm_over_J_zz=rows[0]["J_pm_over_J_zz"],
                    status="not_converged", energy=float("nan"), saddle_file="",
                    message="No converged initial condition"))
    if missing:
        print(f"Leaving {len(missing)} unconverged field-angle points blank: {missing}", flush=True)
    return winners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "gmft_transport_job.json")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    # Set these before importing numpy or starting workers.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    import numpy as np
    import pandas as pd
    if not args.scan_only:
        # Fail before the long multistart scan if a later stage lacks a dependency.
        import h5py
        import scipy
        import matplotlib
    config = json.loads(args.config.read_text())
    config.setdefault("phi_offset_degrees", .05)
    config.setdefault("phi_start_degrees", 0.0)
    config.setdefault("phi_stop_degrees", 30.0)
    config.setdefault("n_phi", 30)
    config.setdefault("convergence_criterion", "full")
    validate(config)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    source = Path(config["gmft_source"]).expanduser()
    config["gmft_source"] = str((ROOT / source).resolve())
    if not (Path(config["gmft_source"]) / "gmft_for_qsi/gmft_field_general.py").is_file():
        raise FileNotFoundError("Set gmft_source to the updated gMFT_for_QSI checkout")
    sys.path.insert(0, config["gmft_source"])
    import inspect
    from gmft_for_qsi.gmft_field import solve_gmft as scalar_solver
    if "convergence_criterion" not in inspect.signature(scalar_solver).parameters:
        raise RuntimeError("Update gmft_for_qsi/gmft_field.py to support convergence_criterion")
    run, plot_directory = configured_output_directories(config)
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".job.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another job is already using {run}")
    # Refuse resume after configuration OR numerical source changes.
    files = [ROOT / name for name in ("gmft_transport_job.py", "tto_general_gmft.py",
        "02_compute_tto_structure_factor.py", "tto_structure_factor.py", "tto_surface_projection.py",
        "03_compute_scattering.py", "04_compute_transport.py", "config.py", "eio_tb.py", "01_generate_eio.py",
        "plot_gmft_transport.py", "plot_gmft_spin_movie.py", "plot_gmft_spin_components.py",
        "tto_spin_wave.py", "swt_transport_components.py")]
    files += sorted((Path(config["gmft_source"]) / "gmft_for_qsi").glob("*.py"))
    manifest = {"config": config, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    saved = run / "manifest.json"
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise ValueError(
            "Run configuration or source changed; choose a new run_label "
            "(or legacy run_directory)"
        )
    atomic_json(saved, manifest)
    import shutil
    source_archive = run / "source"
    source_archive.mkdir(exist_ok=True)
    for path in files:
        target = source_archive / ("gmft_for_qsi" if path.parent.name == "gmft_for_qsi" else "EIO_TTO") / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    phis = nominal_phi_angles(config)
    tasks = [
        (config, str(run), field_index, phi_index, float(phi), start)
        for field_index in range(len(config["fields"]))
        for phi_index, phi in enumerate(phis)
        for start in range(config["random_starts"] + 1)
    ]
    print(f"{len(tasks)} trials, {args.workers} workers; checkpoints: {run}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for index, row in enumerate(pool.map(trial, tasks, chunksize=1), 1):
            rows.append(row)
            if index % (config["random_starts"] + 1) == 0 or index == len(tasks):
                print(f"Trials {index}/{len(tasks)}; converged {sum(r['status']=='ok' for r in rows)}", flush=True)
    pd.DataFrame(rows).to_csv(run / "all_trials.csv", index=False)
    winners = select_winners(
        rows, config["fields"], phis, config["phi_offset_degrees"]
    )
    table = pd.DataFrame(winners)
    table.to_csv(run / "winning_saddles.csv", index=False)
    for phi in phis:
        directory = run / "mean_fields" / f"phi_{phi_label(phi)}"
        directory.mkdir(parents=True, exist_ok=True)
        table[np.isclose(table.phi_degrees, phi)].to_csv(
            directory / "field_diagnostics_min_energy.csv", index=False
        )
    if args.scan_only:
        return
    env = dict(os.environ, EIO_TTO_OUTPUT_ROOT=str(run))
    data = run / "data"
    data.mkdir(exist_ok=True)
    def stage(name, command):
        marker = run / (name + ".complete")
        if not marker.exists():
            subprocess.run([sys.executable, "-u"] + command, cwd=ROOT, env=env, check=True)
            marker.touch()
    eio = data / "surface_states_summary_nk24.hdf5"
    cached = ROOT / "data" / eio.name
    if not eio.exists() and cached.exists():
        # Copy so subsequent replacement of the project cache cannot change a run.
        shutil.copy2(cached, eio)
    if not eio.exists():
        stage("eio", [str(ROOT / "01_generate_eio.py"), "--overwrite"])
    stage("dssf", [str(ROOT / "02_compute_tto_structure_factor.py"),
        "--gmft-folder", str(run / "mean_fields"), "--general-saddles", str(run),
        "--j-pm", str(config["J_pm_ratio"]), "--fields", *map(str, config["fields"]),
        "--j-zz-ev", str(config["j_zz_ev"]), "--eta-ev", str(config["eta_ev"]),
        "--dssf-nk", str(config["nk"]), "--q-perp-mode", config["q_perp_mode"],
        "--n-q-perp", str(config["n_q_perp"]),
        "--transfer-chunk-size", str(config["transfer_chunk_size"]),
        "--allow-missing", "--overwrite"])
    stage("scattering", [str(ROOT / "03_compute_scattering.py"), "--overwrite"])
    stage("transport", [str(ROOT / "04_compute_transport.py")])
    stage("plot", [
        str(ROOT / "plot_gmft_transport.py"), str(run),
        "--output-directory", str(plot_directory),
    ])
    stage("spin_components", [str(ROOT / "plot_gmft_spin_components.py"), str(run)])
    print(
        f"Completed data/logs: {run}\n"
        f"Completed resistivity output: {plot_directory / resistivity_filename(config)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
