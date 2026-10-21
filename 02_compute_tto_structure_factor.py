"""Evaluate the Sec. 4.1 TTO DSSF on the EIO scattering transfers."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    EIO_NK,
    GMFT_IN_PLANE_MIN,
    TTO_CUBIC_LATTICE_ANGSTROM,
    TTO_DSSF_BROADENING_EV,
    TTO_DSSF_NK,
    TTO_FIELD_RATIOS,
    TTO_INTERFACE_SUBLATTICES,
    TTO_J_PM_RATIO,
    TTO_J_ZZ_EV,
    TTO_N_Q_PERP,
    TTO_Q_PERP_MODE,
)
from tto_surface_projection import (
    Q_PERP_MODES,
    q_perp_samples_for_mode,
    surface_projected_dynamic_structure_factor,
)
from tto_structure_factor import (
    dynamic_structure_factor,
    eio_to_tto_reduced_momentum,
    load_gmft_scan,
    row_mean_field,
)
from tto_general_gmft import (
    load_saddle, dynamic_structure_factor as general_dssf, UnstableSpinonSpectrum,
)


def compute(args):
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    output_path = DATA_DIR / "tto_dynamic_structure_factor.hdf5"
    output_csv = DATA_DIR / "tto_gmft_conditions.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    with h5py.File(eio_path) as eio_h5:
        kf = np.squeeze(eio_h5["kf"][...])
        energies = np.squeeze(eio_h5["eigf"][...])

    # Paper convention: q = k-k' and omega = epsilon_k' - epsilon_k.
    q_slab = kf[:, None, :] - kf[None, :, :]
    q_reduced = eio_to_tto_reduced_momentum(q_slab, args.lattice_constant)
    energy_transfer = energies[None, :] - energies[:, None]
    conditions = load_gmft_scan(args.gmft_folder, args.j_pm, args.fields,
                                allow_missing=args.allow_missing)
    if "saddle_file" in conditions and not args.general_saddles:
        raise ValueError("General gMFT tables require --general-saddles; scalar fallback is invalid")
    components = ("continuum", "one_condensate", "two_condensate", "total")
    shape = (len(conditions), len(TTO_INTERFACE_SUBLATTICES), len(kf), len(kf))
    conditions["mf_status"] = conditions["status"]
    if "message" not in conditions:
        conditions["message"] = ""
    conditions["message"] = conditions["message"].fillna("").astype(str)
    warning_path = DATA_DIR.parent / "warnings.txt"
    warning_path.write_text(
        "gMFT transport warnings\n"
        "Skipped points are undefined (NaN), not zero.\n"
        "Fields below are bare mu_B|B|/Jzz; angles are degrees.\n\n")
    if "convergence_criterion" in conditions and conditions["convergence_criterion"].eq("amplitudes").any():
        with warning_path.open("a") as log:
            log.write("NOTE: Amplitude convergence was requested; complex phases and DSSF convergence "
                      "are not certified by this stopping criterion. Full residuals are saved with the winners.\n\n")

    def record_warning(row, stage, message):
        text = (f"WARNING [{stage}] h={row['h_over_J_zz']:.12g}, "
                f"nominal_phi={row['phi_degrees']:.12g}, "
                f"evaluated_phi={row.get('evaluated_phi_degrees', row['phi_degrees']):.12g}: "
                + str(message))
        with warning_path.open("a") as log:
            log.write(text + "\n")
        print(text, flush=True)
    # Write each condition directly, avoiding four full in-memory scan arrays.
    mode = "w" if args.overwrite else "x"
    with h5py.File(output_path, mode) as output:
        for component in components:
            output.create_dataset(component, shape=shape, dtype=float,
                                  fillvalue=np.nan, compression="gzip")
        for condition_index, (_, row) in enumerate(conditions.iterrows()):
            if row["status"] != "ok":
                record_warning(row, "MF", row.get("message") or "No converged initial condition")
                continue
            if args.general_saddles:
                saddle_path = Path(args.general_saddles) / str(row["saddle_file"])
                mean_field = load_saddle(saddle_path)
                if int(mean_field["nk"]) != args.dssf_nk:
                    raise ValueError("DSSF nk must match the winning general gMFT saddle grid")
                if not np.isclose(mean_field["j_pm_ratio"], args.j_pm):
                    raise ValueError("J_pm does not match the winning saddle")
                solver = general_dssf
                extra = {"chunk_size": args.transfer_chunk_size}
            else:
                mean_field, solver, extra = row_mean_field(row), dynamic_structure_factor, {}
            try:
                result = surface_projected_dynamic_structure_factor(
                    solver,
                    q_reduced,
                    energy_transfer,
                    mean_field,
                    q_perp_mode=args.q_perp_mode,
                    n_q_perp=args.n_q_perp,
                    j_zz_ev=args.j_zz_ev,
                    eta_ev=args.eta_ev,
                    nk=args.dssf_nk,
                    interface_sublattices=TTO_INTERFACE_SUBLATTICES,
                    **extra,
                )
            except UnstableSpinonSpectrum as exc:
                # Keep the whole condition undefined, even if earlier chunks
                # or perpendicular-momentum samples completed successfully.
                conditions.loc[condition_index, "status"] = "unstable_dssf"
                conditions.loc[condition_index, "message"] = str(exc)
                record_warning(row, "DSSF", exc)
                continue
            for component in components:
                output[component][condition_index] = result[component]
            if condition_index == 0 or (condition_index + 1) % 25 == 0:
                print(f"DSSF condition {condition_index + 1}/{len(conditions)}", flush=True)

        # Metadata is committed after all conditions have completed.
        output.create_dataset("kf", data=kf)
        output.create_dataset("q_slab", data=q_slab)
        output.create_dataset("q_tto_reduced", data=q_reduced)
        output.create_dataset(
            "q_perp_reduced_offsets",
            data=q_perp_samples_for_mode(args.q_perp_mode, args.n_q_perp),
        )
        output.create_dataset("energy_transfer_ev", data=energy_transfer)
        output.create_dataset("phi_degrees", data=conditions["phi_degrees"])
        output.create_dataset("h_over_J_zz", data=conditions["h_over_J_zz"])
        output.create_dataset("converged", data=conditions["status"].eq("ok"))
        output.create_dataset("mf_converged", data=conditions["mf_status"].eq("ok"))
        output.create_dataset("dssf_valid", data=conditions["status"].eq("ok"))
        if "evaluated_phi_degrees" in conditions:
            output.create_dataset("evaluated_phi_degrees", data=conditions["evaluated_phi_degrees"])
        output.attrs.update(
            theory="gMFT Sec. 4.1, Eqs. (92), (97)-(99)" if args.general_saddles else "scalar gMFT Sec. 4.1",
            momentum_convention="q = k - kprime",
            q_perp_mode=args.q_perp_mode,
            n_q_perp=args.n_q_perp,
            q_perp_reduced_direction=np.ones(3),
            surface_projection=(
                "q_perp=0 slice" if args.q_perp_mode == "slice" else
                "normalized integral over one [111] reciprocal period"
            ),
            energy_convention="omega = epsilon_kprime - epsilon_k",
            J_pm_over_J_zz=args.j_pm,
            J_zz_ev=args.j_zz_ev,
            lorentzian_eta_ev=args.eta_ev,
            dssf_nk=args.dssf_nk,
            tto_cubic_lattice_angstrom=args.lattice_constant,
            gmft_folder=str(args.gmft_folder),
            interface_sublattices=np.asarray(TTO_INTERFACE_SUBLATTICES),
            general_gmft=bool(args.general_saddles),
            allow_missing_gmft=args.allow_missing,
            phi_offset_degrees=(float((conditions["evaluated_phi_degrees"]
                                       - conditions["phi_degrees"]).iloc[0])
                                if "evaluated_phi_degrees" in conditions else 0.),
            transverse_contraction="Sxx+Syy=(S+-+S-+)/2",
        )
    conditions.to_csv(output_csv, index=False)
    print(f"Wrote {output_path}")
    print(f"Wrote {output_csv}")
    skipped = conditions["status"].ne("ok").sum()
    with warning_path.open("a") as log:
        log.write(f"\nSummary: {skipped}/{len(conditions)} points skipped.\n")
    print(f"Wrote {warning_path}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gmft-folder", type=str, default=str(GMFT_IN_PLANE_MIN))
parser.add_argument("--j-pm", type=float, default=TTO_J_PM_RATIO)
parser.add_argument("--fields", nargs="+", type=float, default=list(TTO_FIELD_RATIOS))
parser.add_argument("--j-zz-ev", type=float, default=TTO_J_ZZ_EV)
parser.add_argument("--eta-ev", type=float, default=TTO_DSSF_BROADENING_EV)
parser.add_argument("--dssf-nk", type=int, default=TTO_DSSF_NK)
parser.add_argument(
    "--q-perp-mode", choices=Q_PERP_MODES, default=TTO_Q_PERP_MODE
)
parser.add_argument("--n-q-perp", type=int, default=TTO_N_Q_PERP)
parser.add_argument(
    "--lattice-constant", type=float, default=TTO_CUBIC_LATTICE_ANGSTROM
)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--general-saddles", help="directory containing winning saddle NPZ files")
parser.add_argument("--transfer-chunk-size", type=int, default=128)
parser.add_argument("--allow-missing", action="store_true",
                    help="Keep unconverged conditions as NaN cells and compute the converged points")
if __name__ == "__main__":
    compute(parser.parse_args())
