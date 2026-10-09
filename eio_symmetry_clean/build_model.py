"""Build a symmetry-clean EIO model from the archived 381-vector model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
LATTICE = 5.1350 * np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
])
SITE_FRACTIONAL = np.array([
    [0.125, 0.125, 0.125],
    [0.125, 0.625, 0.125],
    [0.625, 0.125, 0.125],
    [0.125, 0.125, 0.625],
])
SITE_OF_ORBITAL = np.r_[np.repeat(np.arange(4), 3), np.repeat(np.arange(4), 3)]


def load_model(model_dir):
    """Read the archived hopping tensor and Wannier centers."""
    cells = np.loadtxt(model_dir / "cellindices.txt", delimiter=",", dtype=int)
    components = np.loadtxt(model_dir / "ham0.txt", delimiter=",")
    components = components.reshape(24, 24, len(cells), 2)
    hoppings = components[..., 0] + 1j * components[..., 1]
    centres = np.loadtxt(model_dir / "wannier90_centres.xyz")
    return cells, hoppings, centres


def shell_cutoff(cells, hoppings):
    """Reproduce the MATLAB cutoff using ideal, symmetry-related Ir centers."""
    ideal_centres = (SITE_FRACTIONAL @ LATTICE)[SITE_OF_ORBITAL]
    displacements = ideal_centres[:, None, :] - ideal_centres[None, :, :]
    cutoff = 2 * np.linalg.norm(LATTICE[0])

    kept_cells = []
    kept_hoppings = []
    for cell, hopping in zip(cells, np.moveaxis(hoppings, 2, 0)):
        distances = np.linalg.norm(displacements + cell @ LATTICE, axis=2)
        # The historical strict inequality excludes the complete shell at Rcut.
        mask = distances < cutoff - 1e-8
        if np.any(mask):
            kept_cells.append(cell)
            kept_hoppings.append(np.where(mask, hopping, 0))

    return np.asarray(kept_cells), np.moveaxis(np.asarray(kept_hoppings), 0, 2)


def place_site_blocks(block, assignments):
    """Embed one six-component orbital-spin rotation into the four-site basis."""
    site_indices = [
        np.r_[3 * site:3 * site + 3, 12 + 3 * site:12 + 3 * site + 3]
        for site in range(4)
    ]
    full = np.zeros((24, 24), dtype=complex)
    for target, source in assignments:
        full[np.ix_(site_indices[target], site_indices[source])] = block
    return full


def symmetry_matrices():
    """Construct the orbital-spin matrices used by Shiang Fang's MATLAB code."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    spin = expm(-1j * np.pi / 3 * (sx + sy + sz) / np.sqrt(3))
    orbital = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    c3 = place_site_blocks(
        np.kron(spin, orbital), ((0, 0), (3, 1), (1, 2), (2, 3))
    )

    c2z_orbital = np.diag([1, -1, -1])
    c2z = place_site_blocks(
        np.kron(expm(-1j * np.pi / 2 * sz), c2z_orbital),
        ((2, 0), (0, 2), (3, 1), (1, 3)),
    )
    c2y_orbital = np.diag([-1, 1, -1])
    c2y = place_site_blocks(
        np.kron(expm(-1j * np.pi / 2 * sy), c2y_orbital),
        ((3, 0), (0, 3), (2, 1), (1, 2)),
    )
    c2x_orbital = np.diag([-1, -1, 1])
    c2x = place_site_blocks(
        np.kron(expm(-1j * np.pi / 2 * sx), c2x_orbital),
        ((1, 0), (0, 1), (3, 2), (2, 3)),
    )

    mirror_spin = expm(-1j * np.pi / 2 * (sx - sy) / np.sqrt(2))
    mirror_orbital = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    mirror = place_site_blocks(
        np.kron(mirror_spin, mirror_orbital),
        ((0, 0), (2, 2), (3, 1), (1, 3)),
    )
    time_reversal = np.kron(-1j * sy, np.eye(12))
    return c3, c2z, c2y, c2x, mirror @ time_reversal


def similarity(unitary, matrices):
    """Evaluate U-dagger H U for a batch of Hamiltonians."""
    return np.einsum(
        "ij,njk,kl->nil", unitary.conj().T, matrices, unitary, optimize=True
    )


def symmetrize(cells, hoppings, grid_size=13):
    """Apply the recovered cubic and AIAO projectors in the exported cell gauge."""
    grid_indices = np.indices((grid_size,) * 3).reshape(3, -1).T
    fractional_k = grid_indices / grid_size
    reciprocal = 2 * np.pi * np.linalg.inv(LATTICE).T
    momenta = fractional_k @ reciprocal

    phases = np.exp(-2j * np.pi * fractional_k @ cells.T)
    cell_hamiltonians = np.einsum("abr,nr->nab", hoppings, phases, optimize=True)

    lookup = np.empty((grid_size,) * 3, dtype=int)
    lookup[tuple(grid_indices.T)] = np.arange(len(grid_indices))
    inverse_reciprocal = np.linalg.inv(reciprocal)

    def mapped_indices(transformed_momenta):
        transformed_fractional = transformed_momenta @ inverse_reciprocal
        transformed_grid = np.rint(transformed_fractional * grid_size).astype(int)
        transformed_grid %= grid_size
        return lookup[tuple(transformed_grid.T)]

    c3, c2z, c2y, c2x, antiunitary = symmetry_matrices()
    c3a = mapped_indices(momenta[:, [2, 0, 1]])
    c3b = mapped_indices(momenta[:, [1, 2, 0]])
    c3_squared = c3 @ c3
    threefold_symmetric = (
        cell_hamiltonians
        + similarity(c3, cell_hamiltonians[c3a])
        + similarity(c3_squared, cell_hamiltonians[c3b])
    ) / 3

    k_c2z = momenta * np.array([-1, -1, 1])
    k_c2y = momenta * np.array([-1, 1, -1])
    k_c2x = momenta * np.array([1, -1, -1])
    cubic_symmetric = (
        threefold_symmetric
        + similarity(c2z, threefold_symmetric[mapped_indices(k_c2z)])
        + similarity(c2y, threefold_symmetric[mapped_indices(k_c2y)])
        + similarity(c2x, threefold_symmetric[mapped_indices(k_c2x)])
    ) / 4

    reflected_k = -momenta[:, [1, 0, 2]]
    reflected = cubic_symmetric[mapped_indices(reflected_k)].conj()
    reflected = np.einsum(
        "ij,njk,kl->nil",
        antiunitary,
        reflected,
        antiunitary.conj().T,
        optimize=True,
    )
    symmetric_hamiltonians = (cubic_symmetric + reflected) / 2

    inverse_phases = np.exp(2j * np.pi * fractional_k @ cells.T)
    symmetric_hoppings = np.einsum(
        "nab,nr->abr", symmetric_hamiltonians, inverse_phases, optimize=True
    ) / len(fractional_k)
    return symmetric_hoppings


def enforce_real_space_constraints(cells, hoppings):
    """Remove Fourier noise while enforcing Hermiticity and inversion exactly."""
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    hermitian = np.empty_like(hoppings)
    for index, cell in enumerate(cells):
        partner = lookup[tuple(-cell)]
        hermitian[:, :, index] = (
            hoppings[:, :, index] + hoppings[:, :, partner].conj().T
        ) / 2

    site_shifts = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    orbital_shifts = site_shifts[SITE_OF_ORBITAL]
    inversion_symmetric = np.empty_like(hermitian)
    zero = np.zeros((24, 24), dtype=complex)
    for index, cell in enumerate(cells):
        for row in range(24):
            for column in range(24):
                partner_cell = tuple(
                    orbital_shifts[column] - orbital_shifts[row] - cell
                )
                partner = lookup.get(partner_cell)
                if partner is None:
                    inversion_symmetric[row, column, index] = 0
                else:
                    inversion_symmetric[row, column, index] = (
                        hermitian[row, column, index]
                        + hermitian[row, column, partner]
                    ) / 2

    final = np.empty_like(inversion_symmetric)
    for index, cell in enumerate(cells):
        partner = lookup[tuple(-cell)]
        final[:, :, index] = (
            inversion_symmetric[:, :, index]
            + inversion_symmetric[:, :, partner].conj().T
        ) / 2
    final[np.abs(final) < 5e-13] = 0
    return final


def write_model(output_dir, cells, hoppings, centres, source_dir, grid_size):
    """Write the clean model in the same text format as the archived model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "cellindices.txt", cells, fmt="%d", delimiter=",")
    components = np.stack([hoppings.real, hoppings.imag], axis=-1).reshape(-1, 2)
    np.savetxt(output_dir / "ham0.txt", components, fmt="%.12f", delimiter=", ")
    np.savetxt(output_dir / "wannier90_centres.xyz", centres, fmt="%18.8f")

    try:
        source_label = str(source_dir.resolve().relative_to(ROOT.parent.resolve()))
    except ValueError:
        source_label = str(source_dir.resolve())
    metadata = {
        "source_model": source_label,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_dir.iterdir()) if path.is_file()
        },
        "output_sha256": {},
        "symmetry_grid": [grid_size] * 3,
        "cutoff_angstrom": float(2 * np.linalg.norm(LATTICE[0])),
        "cutoff_rule": "strictly inside the complete shell, using ideal Ir centers",
        "translations": int(len(cells)),
    }
    for name in ("cellindices.txt", "ham0.txt", "wannier90_centres.xyz"):
        path = output_dir / name
        metadata["output_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", type=Path, default=ROOT.parent / "model")
    parser.add_argument("--output", type=Path, default=ROOT / "model")
    parser.add_argument("--grid-size", type=int, default=13)
    args = parser.parse_args()

    input_cells, input_hoppings, input_centres = load_model(args.input_model)
    cut_cells, cut_hoppings = shell_cutoff(input_cells, input_hoppings)
    if len(cut_cells) != 79:
        raise ValueError(
            f"Expected 79 translations after cutoff, found {len(cut_cells)}"
        )
    cut_hoppings = enforce_real_space_constraints(cut_cells, cut_hoppings)
    clean_hoppings = symmetrize(cut_cells, cut_hoppings, args.grid_size)
    clean_hoppings = enforce_real_space_constraints(cut_cells, clean_hoppings)
    write_model(
        args.output,
        cut_cells,
        clean_hoppings,
        input_centres,
        args.input_model,
        args.grid_size,
    )
    print(
        f"Wrote symmetry-clean model with {len(cut_cells)} translations "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
