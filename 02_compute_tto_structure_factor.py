"""Evaluate the Sec. 4.1 TTO DSSF on the EIO scattering transfers."""
import argparse

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
    conditions = load_gmft_scan(args.gmft_folder, args.j_pm, args.fields)
    components = ("continuum", "one_condensate", "two_condensate", "total")
    shape = (len(conditions), len(TTO_INTERFACE_SUBLATTICES), len(kf), len(kf))
    arrays = {component: np.empty(shape, dtype=float) for component in components}

    for condition_index, (_, row) in enumerate(conditions.iterrows()):
        result = surface_projected_dynamic_structure_factor(
            dynamic_structure_factor,
            q_reduced,
            energy_transfer,
            row_mean_field(row),
            q_perp_mode=args.q_perp_mode,
            n_q_perp=args.n_q_perp,
            j_zz_ev=args.j_zz_ev,
            eta_ev=args.eta_ev,
            nk=args.dssf_nk,
            interface_sublattices=TTO_INTERFACE_SUBLATTICES,
        )
        for component in components:
            arrays[component][condition_index] = result[component]
        if condition_index == 0 or (condition_index + 1) % 25 == 0:
            print(f"DSSF condition {condition_index + 1}/{len(conditions)}", flush=True)

    mode = "w" if args.overwrite else "x"
    with h5py.File(output_path, mode) as output:
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
        for component, values in arrays.items():
            output.create_dataset(component, data=values, compression="gzip")
        output.attrs.update(
            theory="gMFT Sec. 4.1, Eqs. (85)-(87)",
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
        )
    conditions.to_csv(output_csv, index=False)
    print(f"Wrote {output_path}")
    print(f"Wrote {output_csv}")


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
compute(parser.parse_args())
