"""Compare [111] slab spectra before and after restoring bulk inversion symmetry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.linalg import eigh


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from eio_tb import (build_slab, hexagonal_momentum_grid, load_bulk_model,
                    slab_hamiltonian, transform_to_slab_coordinates,
                    truncate_hoppings)


N_WANN = 24
N_LAYER = 11
ELECTRONS_PER_CELL = 20
BROADENING_EV = 0.002
EIGENVECTOR_WINDOW_EV = 3 * BROADENING_EV
MODEL_FILES = ("cellindices.txt", "ham0.txt", "wannier90_centres.xyz")


def prepare_slab(model_dir, apply_archived_cutoff):
    """Load one bulk model and reproduce its intended slab hopping cutoff."""
    bulk = load_bulk_model(*(model_dir / name for name in MODEL_FILES))
    if apply_archived_cutoff:
        cutoff = 2 * np.linalg.norm(bulk["lattice_vectors"][0])
        slab_bulk = truncate_hoppings(bulk, cutoff)
    else:
        slab_bulk = {
            **bulk,
            "hoppings": np.moveaxis(bulk["hoppings"], 2, 0),
        }

    slab = build_slab(slab_bulk)
    geometry = transform_to_slab_coordinates(bulk, slab)
    return bulk, slab_bulk, slab, geometry


def generate_spectrum(model, nk, output, overwrite):
    """Diagonalize the slab and save all energies and terminating-plane weights."""
    if output.exists() and not overwrite:
        print(f"Reusing {output}", flush=True)
        with h5py.File(output) as h5:
            saved = {name: h5[name][...] for name in (
                "k", "eigenvalues", "triangular_surface_weight",
                "kagome_surface_weight", "spectral_triangular",
                "spectral_kagome",
            )}
            saved["fermi_energy_eV"] = float(h5.attrs["fermi_energy_eV"])
            saved["near_fermi_eigenstates"] = len(
                h5["near_fermi_eigenvectors"]
            )
            return saved

    _, slab_bulk, slab, geometry = prepare_slab(
        model["directory"], model["apply_archived_cutoff"]
    )
    k_grid = hexagonal_momentum_grid(geometry["reciprocal_vectors"], nk)
    if k_grid.shape != (nk * nk, 3):
        raise ValueError(f"Expected {nk * nk} momenta, found {len(k_grid)}")

    # The lower boundary ends on the triangular plane in cell 0. The upper
    # boundary ends on the kagome plane in the final cell of the 11-cell slab.
    positions = geometry["wannier_centres"]
    triangular_indices = np.where(positions[:, 2] < 4)[0]
    kagome_indices = (
        np.where(positions[:, 2] > 4)[0] + N_WANN * (N_LAYER - 1)
    )
    if len(triangular_indices) != 6 or len(kagome_indices) != 18:
        raise ValueError("Unexpected triangular/kagome orbital count")

    n_bands = N_WANN * N_LAYER
    eigenvalues = np.empty((len(k_grid), n_bands))
    triangular_weight = np.empty_like(eigenvalues, dtype=np.float32)
    kagome_weight = np.empty_like(eigenvalues, dtype=np.float32)
    max_hermiticity_error = 0.0
    max_eigenpair_residual = 0.0
    check_points = {0, len(k_grid) // 2, len(k_grid) - 1}

    for ik, momentum in enumerate(k_grid):
        hamiltonian = slab_hamiltonian(
            momentum, slab["hoppings"], geometry["slab_vectors"],
            N_WANN, N_LAYER,
        )
        energies, states = eigh(
            hamiltonian, overwrite_a=True, check_finite=False, driver="evr"
        )
        eigenvalues[ik] = energies
        triangular_weight[ik] = np.sum(
            np.abs(states[triangular_indices]) ** 2, axis=0
        )
        kagome_weight[ik] = np.sum(
            np.abs(states[kagome_indices]) ** 2, axis=0
        )

        if ik in check_points:
            max_hermiticity_error = max(
                max_hermiticity_error,
                float(np.max(np.abs(hamiltonian - hamiltonian.conj().T))),
            )
            max_eigenpair_residual = max(
                max_eigenpair_residual,
                float(np.max(np.abs(hamiltonian @ states - states * energies))),
            )
        if (ik + 1) % 500 == 0:
            print(
                f"{model['key']}: {ik + 1}/{len(k_grid)} momenta", flush=True
            )

    # Match the archived filling convention: 20 occupied states per 24-state
    # bulk cell, multiplied by the 11 cells in the finite slab.
    occupied_states = len(k_grid) * ELECTRONS_PER_CELL * N_LAYER
    sorted_energies = np.sort(eigenvalues, axis=None)
    fermi_energy = float(np.mean(
        sorted_energies[occupied_states - 1:occupied_states + 1]
    ))
    gaussian = np.exp(-((eigenvalues - fermi_energy) / BROADENING_EV) ** 2)
    spectral_triangular = np.sum(gaussian * triangular_weight, axis=1)
    spectral_kagome = np.sum(gaussian * kagome_weight, axis=1)

    # Retain complete slab spinors for every mode that contributes appreciably
    # to the plotted spectrum. Modes at the edge of this window carry the
    # Gaussian factor exp(-9), while saving all spinors would take about 11 GB.
    state_k_index, state_band_index = np.where(
        np.abs(eigenvalues - fermi_energy) <= EIGENVECTOR_WINDOW_EV
    )
    near_fermi_states = np.empty(
        (len(state_k_index), n_bands), dtype=np.complex64
    )
    max_recomputed_energy_error = 0.0
    cursor = 0
    selected_momenta = np.unique(state_k_index)
    for count, ik in enumerate(selected_momenta, 1):
        bands = state_band_index[state_k_index == ik]
        hamiltonian = slab_hamiltonian(
            k_grid[ik], slab["hoppings"], geometry["slab_vectors"],
            N_WANN, N_LAYER,
        )
        energies, states = eigh(
            hamiltonian, subset_by_index=(int(bands[0]), int(bands[-1])),
            overwrite_a=True, check_finite=False, driver="evr",
        )
        n_states = len(bands)
        near_fermi_states[cursor:cursor + n_states] = states.T
        max_recomputed_energy_error = max(
            max_recomputed_energy_error,
            float(np.max(np.abs(energies - eigenvalues[ik, bands]))),
        )
        cursor += n_states
        if count % 500 == 0:
            print(
                f"{model['key']}: saved eigenvectors at "
                f"{count}/{len(selected_momenta)} momenta",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(output, mode) as h5:
        h5.create_dataset("k", data=k_grid)
        h5.create_dataset(
            "eigenvalues", data=eigenvalues, compression="gzip", shuffle=True
        )
        h5.create_dataset(
            "triangular_surface_weight", data=triangular_weight,
            compression="gzip", shuffle=True,
        )
        h5.create_dataset(
            "kagome_surface_weight", data=kagome_weight,
            compression="gzip", shuffle=True,
        )
        h5.create_dataset("spectral_triangular", data=spectral_triangular)
        h5.create_dataset("spectral_kagome", data=spectral_kagome)
        h5.create_dataset("near_fermi_k_index", data=state_k_index)
        h5.create_dataset("near_fermi_band_index", data=state_band_index)
        h5.create_dataset(
            "near_fermi_eigenvalues",
            data=eigenvalues[state_k_index, state_band_index],
        )
        h5.create_dataset(
            "near_fermi_eigenvectors", data=near_fermi_states,
            compression="gzip", shuffle=True,
        )
        h5.attrs.update(
            model=model["label"],
            nk=nk,
            nlayer=N_LAYER,
            nwann=N_WANN,
            electrons_per_bulk_cell=ELECTRONS_PER_CELL,
            fermi_energy_eV=fermi_energy,
            gaussian_broadening_eV=BROADENING_EV,
            eigenvector_energy_window_eV=EIGENVECTOR_WINDOW_EV,
            eigenvector_storage=(
                "Each row of near_fermi_eigenvectors is one normalized "
                "264-component slab eigenstate"
            ),
            triangular_surface="lower terminating plane, 6 orbital-spin components",
            kagome_surface="upper terminating plane, 18 orbital-spin components",
            apply_archived_cutoff=model["apply_archived_cutoff"],
            translations=len(slab_bulk["cell_indices"]),
            max_hermiticity_error_eV=max_hermiticity_error,
            max_eigenpair_residual_eV=max_eigenpair_residual,
            max_recomputed_energy_error_eV=max_recomputed_energy_error,
        )
        for name in MODEL_FILES:
            h5.attrs[f"sha256_{name}"] = hashlib.sha256(
                (model["directory"] / name).read_bytes()
            ).hexdigest()

    print(f"Wrote {output}", flush=True)
    return {
        "k": k_grid,
        "eigenvalues": eigenvalues,
        "triangular_surface_weight": triangular_weight,
        "kagome_surface_weight": kagome_weight,
        "spectral_triangular": spectral_triangular,
        "spectral_kagome": spectral_kagome,
        "fermi_energy_eV": fermi_energy,
        "near_fermi_eigenstates": len(state_k_index),
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--nk", type=int, default=100)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

output_dir = ROOT / "outputs"
figure_dir = ROOT / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
models = [
    {
        "key": "without_inversion",
        "label": "Archived cutoff model (inversion not restored)",
        "short_label": "Inversion not restored",
        "directory": ROOT.parent / "model",
        "apply_archived_cutoff": True,
    },
    {
        "key": "with_inversion",
        "label": "Symmetry-restored model (inversion restored)",
        "short_label": "Inversion restored",
        "directory": ROOT / "model",
        "apply_archived_cutoff": False,
    },
]

results = {}
for model in models:
    output = output_dir / f"slab_spectrum_{model['key']}_nk{args.nk}.hdf5"
    results[model["key"]] = generate_spectrum(
        model, args.nk, output, args.overwrite
    )

reference_k = results[models[0]["key"]]["k"]
if not np.allclose(reference_k, results[models[1]["key"]]["k"]):
    raise ValueError("The two models produced different momentum grids")

surface_names = ("triangular", "kagome")
intensity_scale = {
    surface: max(
        float(np.max(results[model["key"]][f"spectral_{surface}"]))
        for model in models
    )
    for surface in surface_names
}

reciprocal_length = 4 * np.pi / (5.1350 * np.sqrt(6))
radius = reciprocal_length / np.sqrt(3)
angles = np.arange(7) * np.pi / 3
hexagon = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
triangulation = mtri.Triangulation(reference_k[:, 0], reference_k[:, 1])

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.linewidth": 0.8,
})
levels = np.linspace(0, 1, 61)

# The comparison is the primary figure: rows are models and columns are the
# two physical terminations. Each column shares one absolute intensity scale.
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), constrained_layout=True)
for row, model in enumerate(models):
    for column, surface in enumerate(surface_names):
        ax = axes[row, column]
        normalized = (
            results[model["key"]][f"spectral_{surface}"]
            / intensity_scale[surface]
        )
        contour = ax.tricontourf(
            triangulation, normalized, levels=levels, cmap="hot", extend="max"
        )
        ax.plot(hexagon[:, 0], hexagon[:, 1], color="white", lw=1.0)
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.set_xlim(-1.04 * radius, 1.04 * radius)
        ax.set_ylim(-0.92 * radius, 0.92 * radius)
        ax.tick_params(colors="black", direction="out", length=2.5)
        ax.set_title(f"{surface.capitalize()} termination")
        if column == 0:
            ax.set_ylabel(
                model["short_label"] + "\n" +
                r"$k_y$ ($\mathrm{\AA}^{-1}$)"
            )
        if row == 1:
            ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
        ax.text(
            0.025, 0.97, chr(ord("a") + 2 * row + column),
            transform=ax.transAxes, va="top", color="white", weight="bold",
        )
colorbar = fig.colorbar(contour, ax=axes, shrink=0.82, pad=0.02)
colorbar.solids.set_rasterized(False)
colorbar.set_ticks(np.linspace(0, 1, 6))
colorbar.set_label(r"$A_{\mathrm{surf}}(\mathbf{k},E_F)/A_{\max}$")
fig.suptitle(
    "EIO [111] slab surface spectral function\n"
    f"11 cells, {args.nk}$\\times${args.nk} momenta, "
    f"Gaussian broadening {1000 * BROADENING_EV:.0f} meV",
    fontsize=10,
)
comparison_pdf = figure_dir / f"slab_spectral_comparison_nk{args.nk}.pdf"
fig.savefig(comparison_pdf, bbox_inches="tight")
plt.close(fig)

# Save one identically scaled two-panel figure per model for convenient reuse.
for model in models:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    for column, surface in enumerate(surface_names):
        ax = axes[column]
        normalized = (
            results[model["key"]][f"spectral_{surface}"]
            / intensity_scale[surface]
        )
        contour = ax.tricontourf(
            triangulation, normalized, levels=levels, cmap="hot", extend="max"
        )
        ax.plot(hexagon[:, 0], hexagon[:, 1], color="white", lw=1.0)
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.set_xlim(-1.04 * radius, 1.04 * radius)
        ax.set_ylim(-0.92 * radius, 0.92 * radius)
        ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
        ax.set_title(f"{surface.capitalize()} termination")
        ax.text(
            0.025, 0.97, chr(ord("a") + column), transform=ax.transAxes,
            va="top", color="white", weight="bold",
        )
    colorbar = fig.colorbar(contour, ax=axes, shrink=0.84, pad=0.02)
    colorbar.solids.set_rasterized(False)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_label(r"$A_{\mathrm{surf}}(\mathbf{k},E_F)/A_{\max}$")
    fig.suptitle(
        f"{model['label']}\n"
        f"$E_F={results[model['key']]['fermi_energy_eV']:.6f}$ eV, "
        f"Gaussian broadening {1000 * BROADENING_EV:.0f} meV",
        fontsize=10,
    )
    model_pdf = figure_dir / f"slab_spectral_{model['key']}_nk{args.nk}.pdf"
    fig.savefig(model_pdf, bbox_inches="tight")
    plt.close(fig)

summary = {
    "nk": args.nk,
    "n_momenta": len(reference_k),
    "n_bands": N_WANN * N_LAYER,
    "gaussian_broadening_eV": BROADENING_EV,
    "fermi_energy_eV": {
        model["key"]: results[model["key"]]["fermi_energy_eV"]
        for model in models
    },
    "fermi_energy_shift_meV": 1000 * (
        results["with_inversion"]["fermi_energy_eV"]
        - results["without_inversion"]["fermi_energy_eV"]
    ),
    "near_fermi_eigenstates": {
        model["key"]: results[model["key"]]["near_fermi_eigenstates"]
        for model in models
    },
    "surface_intensity_scale": intensity_scale,
    "surface_map_comparison": {
        surface: {
            "pearson_correlation": float(np.corrcoef(
                results[models[0]["key"]][f"spectral_{surface}"],
                results[models[1]["key"]][f"spectral_{surface}"],
            )[0, 1]),
            "normalized_rms_difference": float(np.sqrt(np.mean((
                results[models[0]["key"]][f"spectral_{surface}"]
                / intensity_scale[surface]
                - results[models[1]["key"]][f"spectral_{surface}"]
                / intensity_scale[surface]
            ) ** 2))),
            "normalized_max_difference": float(np.max(np.abs(
                results[models[0]["key"]][f"spectral_{surface}"]
                / intensity_scale[surface]
                - results[models[1]["key"]][f"spectral_{surface}"]
                / intensity_scale[surface]
            ))),
        }
        for surface in surface_names
    },
    "figures": [
        comparison_pdf.name,
        *(f"slab_spectral_{model['key']}_nk{args.nk}.pdf" for model in models),
    ],
}
(output_dir / f"slab_spectral_summary_nk{args.nk}.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
