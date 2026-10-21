"""Contract a TTO DSSF with EIO surface states using Sec. 5 Eq. (102)."""
import argparse

import h5py
import numpy as np
import pandas as pd

from config import DATA_DIR, EIO_NK, ORBITAL_HOPPINGS
from swt_transport_components import (
    combine_transport_components,
    transport_component_definition,
)
from tto_spin_wave import ONE_MAGNON_MODE_COMPONENTS


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
        "components": ("one_magnon", "two_magnon"),
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
        extra_coordinates = {name: dssf_h5[name][...] for name in
                             ("evaluated_phi_degrees", "converged", "mf_converged", "dssf_valid")
                             if name in dssf_h5}
        field_key = (
            "mu_B_B_over_J_zz"
            if "mu_B_B_over_J_zz" in dssf_h5
            else "h_over_J_zz"
        )
        fields = dssf_h5[field_key][...]
        bond_weights = np.asarray([
            sum(
                local_eio_weights(projections[eio], hoppings)
                for eio in EIO_NEIGHBORS[tto]
            )
            for tto in range(3)
        ])
        metadata = dict(dssf_h5.attrs)
        components = settings["components"]
        transport_component = None
        if theory == "spin-wave":
            transport_component = metadata.get("transport_component", "both")
            if isinstance(transport_component, bytes):
                transport_component = transport_component.decode("utf-8")
            if transport_component in ONE_MAGNON_MODE_COMPONENTS:
                if transport_component not in dssf_h5:
                    raise ValueError(
                        f"The DSSF file lacks {transport_component!r}; rerun "
                        "02_compute_tto_spin_wave_structure_factor.py with "
                        "the band-resolved solver"
                    )
                components = components + (transport_component,)
        scattering = {
            # Read one condition at a time: the band-resolved gMFT scan can
            # have hundreds of conditions and should not load the full DSSF.
            component: np.asarray([
                np.einsum("mij,mij->ij", dssf_h5[component][index],
                          bond_weights, optimize=True)
                for index in range(len(phi))
            ])
            for component in components
        }
        if theory == "spin-wave":
            scattering["total"] = combine_transport_components(
                scattering["one_magnon"],
                scattering["two_magnon"],
                transport_component,
                one_magnon_modes={
                    component: scattering[component]
                    for component in ONE_MAGNON_MODE_COMPONENTS
                    if component in scattering
                },
            )
            components = components + ("total",)
            metadata["transport_component"] = transport_component
            metadata["transport_total_definition"] = (
                transport_component_definition(transport_component)
            )
            metadata["bragg_included_in_transport"] = False

    mode = "w" if overwrite else "x"
    with h5py.File(output_path, mode) as output:
        output.create_dataset("kf", data=kf)
        output.create_dataset("phi_degrees", data=phi)
        for name, values in extra_coordinates.items():
            output.create_dataset(name, data=values)
        if theory == "spin-wave":
            output.create_dataset("mu_B_B_over_J_zz", data=fields)
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
    if theory == "spin-wave":
        frame_data["mu_B_B_over_J_zz"] = np.repeat(fields, nk * nk)
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
if __name__ == "__main__":
    args = parser.parse_args()
    compute(args.theory, overwrite=args.overwrite)
