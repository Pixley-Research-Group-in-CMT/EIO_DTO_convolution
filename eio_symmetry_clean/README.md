# Symmetry-clean EIO tight-binding model

This folder regenerates a bulk-symmetry-preserving EIO model from the archived
381-translation `../model/ham0.txt`. The published model and its outputs remain
unchanged; this model is intended for future EIO/TTO calculations.

## Workflow

`build_model.py` performs the following operations in order:

1. Read the archived 24-orbital hopping tensor.
2. Apply the historical `2*|a1| = 14.523973... A` distance cutoff using ideal
   crystallographic Ir centers. The strict cutoff excludes the complete shell
   lying at the boundary and leaves 79 translations.
3. Apply the symmetry projectors recovered from Shiang Fang's MATLAB code:
   inversion, the [111] threefold rotation, cubic twofold generators, and the
   AIAO-compatible antiunitary reflection.
4. Fourier-transform the symmetrized `13x13x13` grid back to the retained
   real-space translations and enforce Hermiticity and inversion exactly.
5. Write `model/ham0.txt`, `model/cellindices.txt`, and
   `model/wannier90_centres.xyz` in the same format as the archived model.

The orbital basis is four Ir sites times three `t2g` orbitals times two spins.
Ordinary time reversal remains broken, as required for the AIAO state.

## Run

From the repository root:

```bash
python eio_symmetry_clean/build_model.py
python eio_symmetry_clean/check_model.py
python eio_symmetry_clean/generate_surface_states.py --nk 24
```

The symmetry check covers real-space Hermiticity and inversion, all 48 cubic
operations at random momenta, ordinary time-reversal breaking, and the residual
threefold symmetry of the [111] slab.

The generated surface-state file uses the same state-selection and radial
velocity conventions as the archived workflow. It is written under `outputs/`
with a neighboring JSON validation report.

## Surface spectral functions

`generate_slab_spectra.py` makes a like-for-like comparison between the
archived cutoff model and the inversion-restored model. Both calculations use
the same 11-cell [111] slab, momentum grid, filling, and 2 meV Gaussian
broadening. The physical lower triangular termination and upper kagome
termination are projected separately.

```bash
conda run -n pyqmc_dev python \
    eio_symmetry_clean/generate_slab_spectra.py --nk 100 --overwrite
```

For each model, the HDF5 output stores all 264 slab eigenvalues and the
probability of every eigenstate on each terminating plane. It also stores the
full normalized 264-component eigenvectors within `+/-6 meV` of the Fermi
energy, together with their momentum and band indices. This is the complete
window relevant to the plotted 2 meV Gaussian broadening down to a relative
weight of `exp(-9)`. Saving every eigenvector at every momentum would require
roughly 11 GB per model and is unnecessary for these spectral functions.

Within each termination, the two model panels use the same absolute intensity
scale. This makes the before/after comparison direct without forcing the
six-component triangular plane and the 18-component kagome plane onto an
artificial common scale.

The script writes vector PDFs only:

- `figures/slab_spectral_comparison_nk100.pdf`
- `figures/slab_spectral_without_inversion_nk100.pdf`
- `figures/slab_spectral_with_inversion_nk100.pdf`

## Validated result

The regenerated model has 79 translations and agrees with the recovered MATLAB
`ham0_wrcut.txt` to `1.0e-12 eV` per stored real or imaginary component. The
current checks give zero real-space Hermiticity and inversion residuals, bulk
cubic and slab-C3 spectral residuals below `2e-11 eV`, and a `5.2%` ordinary
time-reversal mismatch.

On the `nk=24` slab grid, the clean model selects the same 15 momenta and the
same band indices as the archived model. The largest selected-state energy
change is `1.40 meV`; this is why the archived and symmetry-clean models remain
separate rather than silently replacing one with the other.
