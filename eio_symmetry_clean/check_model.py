"""Validate bulk and [111]-slab symmetries of the clean EIO model."""
import argparse
from itertools import permutations, product
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from eio_tb import build_slab, slab_hamiltonian, transform_to_slab_coordinates
from build_model import similarity, symmetry_matrices


LATTICE = 5.1350 * np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
SITE_OF_ORBITAL = np.r_[np.repeat(np.arange(4), 3), np.repeat(np.arange(4), 3)]


def load_model(model_dir):
    cells = np.loadtxt(model_dir / "cellindices.txt", delimiter=",", dtype=int)
    components = np.loadtxt(model_dir / "ham0.txt", delimiter=",")
    components = components.reshape(24, 24, len(cells), 2)
    hoppings = components[..., 0] + 1j * components[..., 1]
    centres = np.loadtxt(model_dir / "wannier90_centres.xyz")
    return cells, hoppings, centres


def hamiltonian(momentum, cells, hoppings):
    phases = np.exp(-1j * (cells @ LATTICE) @ momentum)
    return np.einsum("abr,r->ab", hoppings, phases)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, default=ROOT / "model")
parser.add_argument("--output", type=Path, default=ROOT / "validation.json")
args = parser.parse_args()

cells, hoppings, centres = load_model(args.model)
lookup = {tuple(cell): index for index, cell in enumerate(cells)}

hermiticity = 0.0
for index, cell in enumerate(cells):
    partner = lookup[tuple(-cell)]
    hermiticity = max(
        hermiticity,
        float(np.max(np.abs(
            hoppings[:, :, index] - hoppings[:, :, partner].conj().T
        ))),
    )

site_shifts = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
orbital_shifts = site_shifts[SITE_OF_ORBITAL]
inversion_matrix_error = 0.0
zero = np.zeros((24, 24), dtype=complex)
for index, cell in enumerate(cells):
    for row in range(24):
        for column in range(24):
            partner_cell = tuple(orbital_shifts[column] - orbital_shifts[row] - cell)
            partner = hoppings[:, :, lookup[partner_cell]] if partner_cell in lookup else zero
            inversion_matrix_error = max(
                inversion_matrix_error,
                float(abs(hoppings[row, column, index] - partner[row, column])),
            )

rng = np.random.default_rng(381)
momenta = rng.uniform(-0.45, 0.45, size=(8, 3)) * (2 * np.pi / 10.27)
cubic_error = 0.0
inversion_spectral_error = 0.0
generator_matrix_error = 0.0
c3, c2z, c2y, c2x, antiunitary = symmetry_matrices()
for momentum in momenta:
    h_reference = hamiltonian(momentum, cells, hoppings)
    reference = np.linalg.eigvalsh(h_reference)
    inverted = np.linalg.eigvalsh(hamiltonian(-momentum, cells, hoppings))
    inversion_spectral_error = max(
        inversion_spectral_error, float(np.max(np.abs(reference - inverted)))
    )
    for permutation in permutations(range(3)):
        base = np.eye(3)[list(permutation)]
        for signs in product((-1, 1), repeat=3):
            operation = np.diag(signs) @ base
            transformed = np.linalg.eigvalsh(
                hamiltonian(operation @ momentum, cells, hoppings)
            )
            cubic_error = max(
                cubic_error, float(np.max(np.abs(reference - transformed)))
            )

    generator_data = (
        (c3, momentum[[2, 0, 1]]),
        (c2z, momentum * [-1, -1, 1]),
        (c2y, momentum * [-1, 1, -1]),
        (c2x, momentum * [1, -1, -1]),
    )
    for unitary, transformed_momentum in generator_data:
        transformed_hamiltonian = hamiltonian(
            transformed_momentum, cells, hoppings
        )
        transformed_hamiltonian = similarity(
            unitary, transformed_hamiltonian[None, :, :]
        )[0]
        generator_matrix_error = max(
            generator_matrix_error,
            float(np.max(np.abs(h_reference - transformed_hamiltonian))),
        )

    reflected_momentum = -momentum[[1, 0, 2]]
    reflected = hamiltonian(reflected_momentum, cells, hoppings).conj()
    reflected = antiunitary @ reflected @ antiunitary.conj().T
    generator_matrix_error = max(
        generator_matrix_error, float(np.max(np.abs(h_reference - reflected)))
    )

time_reversal = np.block([
    [np.zeros((12, 12)), np.eye(12)],
    [-np.eye(12), np.zeros((12, 12))],
])
time_reversal_mismatch = 0.0
for momentum in momenta:
    h = hamiltonian(momentum, cells, hoppings)
    h_minus = hamiltonian(-momentum, cells, hoppings)
    transformed = time_reversal @ h_minus.conj() @ time_reversal.conj().T
    time_reversal_mismatch = max(
        time_reversal_mismatch,
        float(np.linalg.norm(h - transformed) / np.linalg.norm(h)),
    )

bulk = {
    "lattice_vectors": LATTICE,
    "wannier_centres": centres,
    "cell_indices": cells,
    "hoppings": np.moveaxis(hoppings, 2, 0),
}
slab = build_slab(bulk)
geometry = transform_to_slab_coordinates(bulk, slab)
rotation_120 = np.array([
    [-0.5, -np.sqrt(3) / 2, 0],
    [np.sqrt(3) / 2, -0.5, 0],
    [0, 0, 1],
])
slab_c3_error = 0.0
for in_plane in rng.uniform(-0.25, 0.25, size=(5, 2)):
    momentum = np.r_[in_plane, 0.0]
    h = slab_hamiltonian(
        momentum, slab["hoppings"], geometry["slab_vectors"], 24, 11
    )
    h_rotated = slab_hamiltonian(
        rotation_120 @ momentum,
        slab["hoppings"],
        geometry["slab_vectors"],
        24,
        11,
    )
    slab_c3_error = max(
        slab_c3_error,
        float(np.max(np.abs(np.linalg.eigvalsh(h) - np.linalg.eigvalsh(h_rotated)))),
    )

validation = {
    "translations": int(len(cells)),
    "real_space_hermiticity_error_eV": hermiticity,
    "real_space_inversion_error_eV": inversion_matrix_error,
    "inversion_spectral_error_eV": inversion_spectral_error,
    "all_48_cubic_spectral_error_eV": cubic_error,
    "magnetic_generator_matrix_error_eV": generator_matrix_error,
    "slab_c3_spectral_error_eV": slab_c3_error,
    "time_reversal_relative_mismatch": time_reversal_mismatch,
}
print(json.dumps(validation, indent=2))
if hermiticity > 1e-9 or inversion_matrix_error > 1e-9:
    raise ValueError("Real-space Hermiticity or inversion validation failed")
if (
    inversion_spectral_error > 1e-8
    or cubic_error > 1e-8
    or generator_matrix_error > 1e-8
    or slab_c3_error > 1e-8
):
    raise ValueError("Bulk or slab spectral symmetry validation failed")
if time_reversal_mismatch < 1e-3:
    raise ValueError("The AIAO model unexpectedly preserves ordinary time reversal")

args.output.write_text(json.dumps(validation, indent=2) + "\n")
