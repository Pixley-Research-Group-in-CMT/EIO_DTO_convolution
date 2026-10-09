"""Generate [111] surface states from the symmetry-clean EIO model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from eio_tb import build_slab, hexagonal_momentum_grid, slab_hamiltonian
from eio_tb import transform_to_slab_coordinates


LATTICE = 5.1350 * np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
E_FERMI = 5.142059246119027


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--nk", type=int, choices=(24, 36), default=24)
parser.add_argument("--model", type=Path, default=ROOT / "model")
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
output = args.output_dir / f"surface_states_summary_nk{args.nk}.hdf5"
validation_path = args.output_dir / f"surface_validation_nk{args.nk}.json"
if output.exists() and not args.overwrite:
    raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

cells = np.loadtxt(args.model / "cellindices.txt", delimiter=",", dtype=int)
components = np.loadtxt(args.model / "ham0.txt", delimiter=",")
components = components.reshape(24, 24, len(cells), 2)
hoppings = components[..., 0] + 1j * components[..., 1]
centres = np.loadtxt(args.model / "wannier90_centres.xyz")
bulk = {
    "lattice_vectors": LATTICE,
    "wannier_centres": centres,
    "cell_indices": cells,
    "hoppings": np.moveaxis(hoppings, 2, 0),
}
slab = build_slab(bulk)
geometry = transform_to_slab_coordinates(bulk, slab)
k_grid = hexagonal_momentum_grid(geometry["reciprocal_vectors"], args.nk)

positions = geometry["wannier_centres"]
top = positions[:, 2] > 4
masks = [
    top & (positions[:, 0] < -0.5),
    top & (positions[:, 0] > -0.5) & (positions[:, 0] < 0.5),
    top & (positions[:, 0] > 0.5),
]
surface_indices = [
    (np.where(mask)[0] + 24 * 10).reshape(-1, 3).T for mask in masks
]
kagome_indices = np.concatenate(surface_indices, axis=1).reshape(-1)


def diagonalize(momentum):
    hamiltonian = slab_hamiltonian(
        momentum, slab["hoppings"], geometry["slab_vectors"], 24, 11
    )
    energies, states = np.linalg.eigh(hamiltonian)
    return hamiltonian, energies, states


records = {
    name: []
    for name in (
        "kf", "eigf", "band_indices", "proj_sl1", "proj_sl2", "proj_sl3", "v_kf"
    )
}
max_hermiticity_error = 0.0
for ik, momentum in enumerate(k_grid):
    hamiltonian, energies, states = diagonalize(momentum)
    max_hermiticity_error = max(
        max_hermiticity_error,
        float(np.max(np.abs(hamiltonian - hamiltonian.conj().T))),
    )
    surface_weight = np.sum(np.abs(states[kagome_indices]) ** 2, axis=0)
    surface_bands = np.where(surface_weight > 0.5)[0]
    if len(surface_bands):
        errors = np.abs(energies[surface_bands] - E_FERMI)
        if errors.min() < 0.01:
            closest_error = errors.min()
            band = np.where(np.abs(energies - E_FERMI) == closest_error)[0][0]
            records["kf"].append(momentum[None, :])
            records["eigf"].append(energies[band])
            records["band_indices"].append(band)
            for sublattice, indices in enumerate(surface_indices, 1):
                records[f"proj_sl{sublattice}"].append(states[indices, band])

            delta_k = 1e-4 * momentum
            _, energy_plus, _ = diagonalize(momentum + delta_k)
            _, energy_minus, _ = diagonalize(momentum - delta_k)
            direction = momentum / np.linalg.norm(momentum)
            velocity_plus = (
                (energy_plus[band] - energies[band]) / np.linalg.norm(delta_k)
                * direction
            )
            velocity_minus = (
                (energies[band] - energy_minus[band]) / np.linalg.norm(delta_k)
                * direction
            )
            records["v_kf"].append((velocity_plus + velocity_minus) / 2)

    if (ik + 1) % 200 == 0:
        print(f"nk{args.nk}: {ik + 1}/{len(k_grid)} momenta", flush=True)

arrays = {name: np.asarray(values) for name, values in records.items()}
mode = "w" if args.overwrite else "x"
with h5py.File(output, mode) as h5:
    for name, values in arrays.items():
        h5.create_dataset(name, data=values)
    h5.attrs.update(
        model="symmetry-clean 79-translation EIO model",
        efermi=E_FERMI,
        nk=args.nk,
        nlayer=11,
        nwann=24,
        surface_weight_cutoff=0.5,
        energy_window=0.01,
        velocity_convention="Archived radial finite difference; no hbar conversion",
    )

validation = {
    "nk": args.nk,
    "momenta": int(len(k_grid)),
    "selected_states": int(len(arrays["eigf"])),
    "max_hermiticity_error_eV": max_hermiticity_error,
    "model_sha256": {
        name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
        for name in ("cellindices.txt", "ham0.txt", "wannier90_centres.xyz")
    },
}
reference_path = (
    ROOT.parent / "reference/eio" / f"surface_states_summary_nk{args.nk}.hdf5"
)
if reference_path.exists():
    with h5py.File(reference_path) as reference_h5:
        reference = {name: reference_h5[name][...] for name in reference_h5}
    validation["archived_reference_errors"] = {
        "kf": float(np.max(np.abs(arrays["kf"] - reference["kf"]))),
        "eigf_eV": float(np.max(np.abs(arrays["eigf"] - reference["eigf"]))),
        "band_indices": int(np.max(np.abs(
            arrays["band_indices"] - reference["band_indices"]
        ))),
        "radial_velocity": float(np.max(np.abs(
            arrays["v_kf"] - reference["v_kf"]
        ))),
    }
    generated_spinors = np.concatenate([
        arrays[f"proj_sl{i}"].reshape(len(arrays["eigf"]), -1)
        for i in (1, 2, 3)
    ], axis=1)
    reference_spinors = np.concatenate([
        reference[f"proj_sl{i}"].reshape(len(reference["eigf"]), -1)
        for i in (1, 2, 3)
    ], axis=1)
    overlap = np.sum(generated_spinors.conj() * reference_spinors, axis=1)
    phase = np.exp(1j * np.angle(overlap))
    validation["archived_reference_errors"]["projected_spinors_phase_aligned"] = (
        float(np.max(np.abs(
            generated_spinors * phase[:, None] - reference_spinors
        )))
    )
validation_path.write_text(json.dumps(validation, indent=2) + "\n")
print(json.dumps(validation, indent=2))
print(f"Wrote {output}")
