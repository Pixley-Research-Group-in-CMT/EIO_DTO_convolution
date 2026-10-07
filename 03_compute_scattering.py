"""Contract a TTO DSSF with EIO surface states using Sec. 5 Eq. (102)."""
import argparse

import h5py
import numpy as np
import pandas as pd

from config import DATA_DIR, EIO_NK, ORBITAL_HOPPINGS


# Each interfacial TTO kagome site has two neighboring EIO kagome sites.
EIO_NEIGHBORS = {0: (1, 2), 1: (1, 3), 2: (2, 3)}


def local_eio_weights(projection, orbital_hoppings):
    """Sec. 5 local factor sum_{alpha,s}|<i alpha s|k><k'|i alpha s>|^2."""
    product_squared = np.abs(
        projection[:, None, :, :].conj() * projection[None, :, :, :]
    ) ** 2
    return np.einsum("ijxs,x->ij", product_squared, orbital_hoppings ** 2)


THEORIES = {
    "spinon": {
        "input": "tto_dynamic_structure_factor.hdf5",
        "output": "tto_dynamic_scattering",
        "components": ("continuum", "one_condensate", "two_condensate", "total"),
    },
    "spin-wave": {
        "input": "tto_spin_wave_structure_factor.hdf5",
        "output": "tto_spin_wave_scattering",
        "components": ("one_magnon", "two_magnon", "bragg", "total"),
    },
}


def compute(theory, overwrite=False):
    settings = THEORIES[theory]
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    dssf_path = DATA_DIR / settings["input"]
    output_path = DATA_DIR / f"{settings['output']}.hdf5"
    output_csv = DATA_DIR / f"{settings['output']}.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    hoppings = np.asarray(ORBITAL_HOPPINGS, dtype=float)
    with h5py.File(eio_path) as eio_h5, h5py.File(dssf_path) as dssf_h5:
        projections = {
            sublattice: eio_h5[f"proj_sl{sublattice}"][...]
            for sublattice in (1, 2, 3)
        }
        kf = np.squeeze(eio_h5["kf"][...])
        phi = dssf_h5["phi_degrees"][...]
        fields = dssf_h5["h_over_J_zz"][...]
        bond_weights = np.asarray([
            sum(
                local_eio_weights(projections[eio], hoppings)
                for eio in EIO_NEIGHBORS[tto]
            )
            for tto in range(3)
        ])
        components = settings["components"]
        scattering = {
            component: np.einsum(
                "cmij,mij->cij", dssf_h5[component][...], bond_weights,
                optimize=True,
            )
            for component in components
        }
        metadata = dict(dssf_h5.attrs)

    mode = "w" if overwrite else "x"
    with h5py.File(output_path, mode) as output:
        output.create_dataset("kf", data=kf)
        output.create_dataset("phi_degrees", data=phi)
        output.create_dataset("h_over_J_zz", data=fields)
        output.create_dataset("eio_bond_weights", data=bond_weights)
        for component, values in scattering.items():
            output.create_dataset(component, data=values)
        output.attrs.update(metadata)
        output.attrs["transport_contraction"] = "Sec. 5 Eq. (102)"
        output.attrs["interface_exchange_J_squared"] = 1.0

    nk = len(kf)
    frame_data = {
        "phi_degrees": np.repeat(phi, nk * nk),
        "h_over_J_zz": np.repeat(fields, nk * nk),
        "ik": np.tile(np.repeat(np.arange(nk), nk), len(phi)),
        "ikprime": np.tile(np.arange(nk), len(phi) * nk),
        "t2": scattering["total"].ravel(),
    }
    for component in components:
        if component != "total":
            frame_data[f"t2_{component}"] = scattering[component].ravel()
    frame = pd.DataFrame(frame_data)
    frame.to_csv(output_csv, index=False)
    print(f"Wrote {output_path}")
    print(f"Wrote {output_csv}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--theory", choices=tuple(THEORIES), default="spinon")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
compute(args.theory, overwrite=args.overwrite)
