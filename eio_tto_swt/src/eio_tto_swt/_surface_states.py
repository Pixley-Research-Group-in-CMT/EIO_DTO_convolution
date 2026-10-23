from pathlib import Path
'Generate the EIO surface eigenstates used by the interface calculation.'
import hashlib
import json
import h5py
import numpy as np
from ._config import E_FERMI, ENERGY_WINDOW, N_LAYER, N_WANN, SURFACE_WEIGHT_CUTOFF
from .eio import build_slab, hexagonal_momentum_grid, load_bulk_model, slab_hamiltonian, transform_to_slab_coordinates, truncate_hoppings

def generate_surface_states(nk, overwrite=False, *, data_dir, model_dir, eio_model):
    """Diagonalize the EIO [111] slab and retain near-Fermi top-surface states.

    The compact output contains only the momenta needed by the interface
    calculation, together with their energies, velocities, and orbital-spin
    amplitudes on the three interfacial EIO sublattices.
    """
    DATA_DIR = Path(data_dir)
    MODEL_DIR = Path(model_dir)
    EIO_MODEL = eio_model
    output = DATA_DIR / f'surface_states_summary_nk{nk}.hdf5'
    validation_output = DATA_DIR / f'eio_validation_nk{nk}.json'
    if output.exists() and (not overwrite):
        raise FileExistsError(f'{output} exists; set overwrite=True to replace it')
    bulk = load_bulk_model(MODEL_DIR / 'cellindices.txt', MODEL_DIR / 'ham0.txt', MODEL_DIR / 'wannier90_centres.xyz')
    cutoff = 2 * np.linalg.norm(bulk['lattice_vectors'][0])
    truncated_bulk = truncate_hoppings(bulk, cutoff) if EIO_MODEL == 'original' else dict(bulk, hoppings=np.moveaxis(bulk['hoppings'], 2, 0))
    slab = build_slab(truncated_bulk)
    geometry = transform_to_slab_coordinates(bulk, slab)
    k_grid = hexagonal_momentum_grid(geometry['reciprocal_vectors'], nk)
    assert k_grid.shape == (nk * nk, 3)
    positions = geometry['wannier_centres']
    top = positions[:, 2] > 4
    masks = [top & (positions[:, 0] < -0.5), top & (positions[:, 0] > -0.5) & (positions[:, 0] < 0.5), top & (positions[:, 0] > 0.5)]
    top_cell_offset = N_WANN * (N_LAYER - 1)
    indices = [(np.where(mask)[0] + top_cell_offset).reshape(-1, 3).T for mask in masks]
    assert all((index.shape == (3, 2) for index in indices))
    kagome_indices = np.concatenate(indices, axis=1).reshape(-1)
    records = {name: [] for name in ('kf', 'eigf', 'band_indices', 'proj_sl1', 'proj_sl2', 'proj_sl3', 'v_kf')}
    validation = {'eio_model': EIO_MODEL, 'nk': nk, 'momenta': len(k_grid), 'max_hermiticity_error': 0.0, 'max_eigenpair_residual': 0.0, 'max_orthonormality_error': 0.0}

    def diagonalize(k):
        """Return the slab Hamiltonian and all eigenpairs at one momentum."""
        hamiltonian = slab_hamiltonian(k, slab['hoppings'], geometry['slab_vectors'], n_orbitals=N_WANN, n_layers=N_LAYER)
        (energies, states) = np.linalg.eigh(hamiltonian)
        return (hamiltonian, energies, states)
    for (ik, k) in enumerate(k_grid):
        (hamiltonian, energies, states) = diagonalize(k)
        projections = [states[index, :] for index in indices]
        surface_weight = np.sum(abs(states[kagome_indices, :]) ** 2, axis=0)
        validation['max_hermiticity_error'] = max(validation['max_hermiticity_error'], float(np.max(abs(hamiltonian - hamiltonian.conj().T))))
        if ik in (0, len(k_grid) // 2, len(k_grid) - 1):
            validation['max_eigenpair_residual'] = max(validation['max_eigenpair_residual'], float(np.max(abs(hamiltonian @ states - states * energies))))
            validation['max_orthonormality_error'] = max(validation['max_orthonormality_error'], float(np.max(abs(states.conj().T @ states - np.eye(len(energies))))))
        surface_bands = np.where(surface_weight > SURFACE_WEIGHT_CUTOFF)[0]
        if len(surface_bands):
            energy_error = abs(energies[surface_bands] - E_FERMI)
            if energy_error.min() < ENERGY_WINDOW:
                closest_error = energy_error.min()
                band = np.where(abs(energies - E_FERMI) == closest_error)[0][0]
                records['kf'].append(k[None, :])
                records['eigf'].append(energies[band])
                records['band_indices'].append(band)
                for (sublattice, projection) in enumerate(projections, 1):
                    records[f'proj_sl{sublattice}'].append(projection[..., band])
                dk = 0.0001 * k
                if np.linalg.norm(dk) == 0:
                    raise ValueError('Archived radial velocity is undefined at Gamma')
                (_, energy_plus, _) = diagonalize(k + dk)
                (_, energy_minus, _) = diagonalize(k - dk)
                direction = k / np.linalg.norm(k)
                velocity_plus = (energy_plus[band] - energies[band]) / np.linalg.norm(dk) * direction
                velocity_minus = (energies[band] - energy_minus[band]) / np.linalg.norm(dk) * direction
                records['v_kf'].append((velocity_plus + velocity_minus) / 2)
        if (ik + 1) % 200 == 0:
            print(f'nk{nk}: {ik + 1}/{len(k_grid)} momenta', flush=True)
    arrays = {name: np.asarray(values) for (name, values) in records.items()}
    if not all((np.isfinite(array).all() for array in arrays.values())):
        raise ValueError('Non-finite values found in generated EIO data')
    mode = 'w' if overwrite else 'x'
    with h5py.File(output, mode) as h5:
        for (name, values) in arrays.items():
            h5.create_dataset(name, data=values)
        h5.attrs.update(eio_model=EIO_MODEL, efermi=E_FERMI, nk=nk, nlayer=N_LAYER, nwann=N_WANN, surface_weight_cutoff=SURFACE_WEIGHT_CUTOFF, energy_window=ENERGY_WINDOW, velocity_convention='Archived radial finite difference; no hbar conversion')
    validation['selected_states'] = len(arrays['eigf'])
    validation['model_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(MODEL_DIR.iterdir()) if path.is_file()}
    validation_output.write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2), flush=True)
    print(f'Wrote {output}')
    return output
