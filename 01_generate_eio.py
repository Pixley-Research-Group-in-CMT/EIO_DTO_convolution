"""Generate the EIO surface eigenstates used by the interface calculation."""
import argparse
import hashlib
import json

import h5py
import numpy as np

from config import (DATA_DIR, E_FERMI, EIO_REFERENCE, ENERGY_WINDOW, MODEL_DIR,
                    N_LAYER, N_WANN, SURFACE_WEIGHT_CUTOFF)
from eio_tb import (build_slab, hexagonal_momentum_grid, load_bulk_model,
                    slab_hamiltonian, transform_to_slab_coordinates,
                    truncate_hoppings)


def phase_aligned_spinor_error(generated, reference):
    """Compare projected spinors after removing one global phase per state."""
    generated = np.concatenate(
        [generated[f"proj_sl{i}"].reshape(len(generated["eigf"]), -1)
         for i in (1, 2, 3)], axis=1)
    reference = np.concatenate(
        [reference[f"proj_sl{i}"].reshape(len(reference["eigf"]), -1)
         for i in (1, 2, 3)], axis=1)
    overlap = np.sum(generated.conj() * reference, axis=1)
    phase = np.exp(1j * np.angle(overlap))
    return float(np.max(abs(generated * phase[:, None] - reference)))


def generate_surface_states(nk, overwrite=False):
    """Diagonalize the EIO [111] slab and retain near-Fermi top-surface states.

    The compact output contains only the momenta needed by the interface
    calculation, together with their energies, velocities, and orbital-spin
    amplitudes on the three interfacial EIO sublattices.
    """
    output = DATA_DIR / f"surface_states_summary_nk{nk}.hdf5"
    validation_output = DATA_DIR / f"eio_validation_nk{nk}.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    # Convert the bulk real-space Wannier model into an open [111] slab, then
    # sample its two-dimensional hexagonal Brillouin zone.
    bulk = load_bulk_model(
        MODEL_DIR / "cellindices.txt",
        MODEL_DIR / "ham0.txt",
        MODEL_DIR / "wannier90_centres.xyz",
    )
    cutoff = 2 * np.linalg.norm(bulk["lattice_vectors"][0])
    truncated_bulk = truncate_hoppings(bulk, cutoff)
    slab = build_slab(truncated_bulk)
    geometry = transform_to_slab_coordinates(bulk, slab)
    k_grid = hexagonal_momentum_grid(geometry["reciprocal_vectors"], nk)
    assert k_grid.shape == (nk * nk, 3)

    # On the top surface, the three x-coordinate groups are the three kagome
    # sublattices. Each group has shape (3 t2g orbitals, 2 spins).
    positions = geometry["wannier_centres"]
    top = positions[:, 2] > 4
    masks = [
        top & (positions[:, 0] < -0.5),
        top & (positions[:, 0] > -0.5) & (positions[:, 0] < 0.5),
        top & (positions[:, 0] > 0.5),
    ]
    top_cell_offset = N_WANN * (N_LAYER - 1)
    indices = [
        (np.where(mask)[0] + top_cell_offset).reshape(-1, 3).T
        for mask in masks
    ]
    assert all(index.shape == (3, 2) for index in indices)
    kagome_indices = np.concatenate(indices, axis=1).reshape(-1)

    records = {
        name: []
        for name in ("kf", "eigf", "band_indices", "proj_sl1", "proj_sl2",
                     "proj_sl3", "v_kf")
    }
    validation = {
        "nk": nk,
        "momenta": len(k_grid),
        "max_hermiticity_error": 0.0,
        "max_eigenpair_residual": 0.0,
        "max_orthonormality_error": 0.0,
    }

    def diagonalize(k):
        """Return the slab Hamiltonian and all eigenpairs at one momentum."""
        hamiltonian = slab_hamiltonian(
            k,
            slab["hoppings"],
            geometry["slab_vectors"],
            n_orbitals=N_WANN,
            n_layers=N_LAYER,
        )
        energies, states = np.linalg.eigh(hamiltonian)
        return hamiltonian, energies, states

    # Search the full nk-by-nk grid, but save only a top-surface state lying
    # within ENERGY_WINDOW of the fixed Fermi energy at a given momentum.
    for ik, k in enumerate(k_grid):
        hamiltonian, energies, states = diagonalize(k)
        projections = [states[index, :] for index in indices]
        surface_weight = np.sum(abs(states[kagome_indices, :]) ** 2, axis=0)
        validation["max_hermiticity_error"] = max(
            validation["max_hermiticity_error"],
            float(np.max(abs(hamiltonian - hamiltonian.conj().T))),
        )
        # Eigenpair checks are expensive, so evaluate them at representative
        # momenta while checking Hermiticity at every momentum.
        if ik in (0, len(k_grid) // 2, len(k_grid) - 1):
            validation["max_eigenpair_residual"] = max(
                validation["max_eigenpair_residual"],
                float(np.max(abs(hamiltonian @ states - states * energies))),
            )
            validation["max_orthonormality_error"] = max(
                validation["max_orthonormality_error"],
                float(np.max(abs(states.conj().T @ states - np.eye(len(energies))))),
            )

        surface_bands = np.where(surface_weight > SURFACE_WEIGHT_CUTOFF)[0]
        if len(surface_bands):
            energy_error = abs(energies[surface_bands] - E_FERMI)
            if energy_error.min() < ENERGY_WINDOW:
                # Preserve the exact state-selection convention of the old notebooks.
                closest_error = energy_error.min()
                band = np.where(abs(energies - E_FERMI) == closest_error)[0][0]
                records["kf"].append(k[None, :])
                records["eigf"].append(energies[band])
                records["band_indices"].append(band)
                for sublattice, projection in enumerate(projections, 1):
                    records[f"proj_sl{sublattice}"].append(projection[..., band])

                # This is the archived radial derivative, not a full band gradient.
                dk = 1e-4 * k
                if np.linalg.norm(dk) == 0:
                    raise ValueError("Archived radial velocity is undefined at Gamma")
                _, energy_plus, _ = diagonalize(k + dk)
                _, energy_minus, _ = diagonalize(k - dk)
                direction = k / np.linalg.norm(k)
                velocity_plus = (
                    (energy_plus[band] - energies[band]) / np.linalg.norm(dk)
                    * direction
                )
                velocity_minus = (
                    (energies[band] - energy_minus[band]) / np.linalg.norm(dk)
                    * direction
                )
                records["v_kf"].append((velocity_plus + velocity_minus) / 2)

        if (ik + 1) % 200 == 0:
            print(f"nk{nk}: {ik + 1}/{len(k_grid)} momenta", flush=True)

    arrays = {name: np.asarray(values) for name, values in records.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Non-finite values found in generated EIO data")

    # HDF5 stores the scientific data; the neighboring JSON records numerical
    # checks and comparisons against the archived nk=24 fingerprint.
    mode = "w" if overwrite else "x"
    with h5py.File(output, mode) as h5:
        for name, values in arrays.items():
            h5.create_dataset(name, data=values)
        h5.attrs.update(
            efermi=E_FERMI,
            nk=nk,
            nlayer=N_LAYER,
            nwann=N_WANN,
            surface_weight_cutoff=SURFACE_WEIGHT_CUTOFF,
            energy_window=ENERGY_WINDOW,
            velocity_convention="Archived radial finite difference; no hbar conversion",
        )

    validation["selected_states"] = len(arrays["eigf"])
    validation["model_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(MODEL_DIR.iterdir()) if path.is_file()
    }
    if nk == 24 and EIO_REFERENCE.exists():
        with h5py.File(EIO_REFERENCE) as reference_h5:
            reference = {name: reference_h5[name][...] for name in reference_h5}
        validation["reference_errors"] = {
            name: float(np.max(abs(arrays[name] - reference[name])))
            for name in ("kf", "eigf", "band_indices", "v_kf")
        }
        validation["reference_errors"]["projected_spinors_phase_aligned"] = (
            phase_aligned_spinor_error(arrays, reference)
        )

    validation_output.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)
    print(f"Wrote {output}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--nk", type=int, choices=(24, 36), default=24)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
DATA_DIR.mkdir(exist_ok=True)
generate_surface_states(args.nk, overwrite=args.overwrite)
