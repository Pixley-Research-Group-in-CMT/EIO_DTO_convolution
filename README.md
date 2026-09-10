# EIO/DTO interface convolution

This directory contains a clean, linear reproduction of the archived U=1.3, 8x8/24x24 EIO/DTO interface-transport calculation.

The workflow is

```text
EIO real-space tight binding
    -> EIO surface eigenvalues and projected spinors
DTO classical-spin snapshots
    -> sample-resolved S_j(q)
EIO spinors + S_j(q) + interface geometry
    -> |T_kk'|^2
|T_kk'|^2 + elastic Boltzmann transport
    -> tau(k), sigma_int, rho_int
    -> CSV files and vector-PDF plots
```

This is a **static and elastic** calculation. It uses DTO snapshots and does not yet use a dynamical structure factor `S(q, omega)`.

## Environment

The scripts were tested with the `pyqmc_dev` environment:

```bash
conda activate pyqmc_dev
```

The required packages are listed in `requirements.txt`.

## Input data

### EIO

The numerical U=1.3 model is included in `model/`:

```text
ham0.txt
cellindices.txt
wannier90_centres.xyz
```

`ham0.txt` is the exact input used by the archived zero-field notebooks. This workflow does not apply an additional symmetrization. The supplied file may have been processed upstream, but that history is not encoded in this directory.

### DTO

The expected DTO input is:

```text
inputs/dto/gathered_snapshots_hex.hdf5
```

On the original workstation this can be linked to Zhengtao's surviving 8x8 snapshot file. When moving the repository, replace that link with the corresponding HDF5 file or pass its path explicitly to `02_compute_dto_fourier.py --snapshots`.

The required HDF5 hierarchy is:

```text
coords
H{field}/phi{angle}/kT0.5/seed{seed}/tstep{snapshot}/configs
```

with fields `0.1, 0.5, 1.0, 2.0, 4.0`, 60 angles, four seeds, and 20 snapshots.

## Run the workflow

Run the scripts in numerical order from this directory:

```bash
python 01_generate_eio.py
python 02_compute_dto_fourier.py
python 03_compute_scattering.py
python 04_compute_transport.py
python 05_plot_transport.py
```

Each computational stage accepts `--fields` where applicable. The first three stages refuse to overwrite their principal outputs unless `--overwrite` is supplied.

For a quick one-field run:

```bash
python 01_generate_eio.py
python 02_compute_dto_fourier.py --fields 0.1
python 03_compute_scattering.py --fields 0.1
python 04_compute_transport.py --fields 0.1
```

The plotting script expects all five default fields.

## Stage 1: EIO surface states

`01_generate_eio.py`:

1. reads the 24x24x381 real-space hopping tensor;
2. retains hopping records within the historical 14.524-A cutoff;
3. constructs an 11-cell [111] slab with a 264x264 Hamiltonian;
4. diagonalizes a 24x24 hexagonal momentum grid;
5. selects states with top-kagome weight greater than 0.5 and energy within 0.01 eV of the fixed Fermi energy;
6. saves 15 selected momenta, energies, band indices, radial velocities, and complex `(3 orbital, 2 spin)` amplitudes for three interface sublattices.

Output:

```text
data/surface_states_summary_nk24.hdf5
data/eio_validation_nk24.json
```

The generated state summary is automatically compared with the small archived reference in `reference/eio/`. Eigenvector comparisons allow one physically irrelevant global phase per state.

## Stage 2: DTO Fourier amplitudes

`02_compute_dto_fourier.py` rotates and rescales the DTO coordinates exactly as in the archived 8x8 notebook, selects the 192 interfacial kagome spins, and separates them into three 64-site sublattices.

For every field angle and every `q = k' - k`, it saves

```text
S_q[angle, sublattice, ik, ikprime, seed, snapshot, spin_component]
```

using the historical normalization `1/sqrt(64)`. The archived production files were generated in two notebook states: field 0.1 used snapshots 4 through 19, while fields 0.5, 1.0, 2.0, and 4.0 used snapshots 10 through 19. This field-dependent inconsistency is preserved explicitly so that the old transport curves can be reproduced.

Outputs for each field:

```text
data/spin_fourier_h{field}.hdf5
data/structure_factor_h{field}.csv
```

The CSV contains the thermally averaged scalar structure factor. The sample-resolved complex amplitudes remain in HDF5 because they are needed for the coherent interface contraction.

## Stage 3: Interface scattering

`03_compute_scattering.py` evaluates the historical interface matrix element by combining:

- DTO Fourier spin amplitudes;
- Pauli matrices;
- EIO orbital-spin amplitudes;
- three EIO and DTO kagome sublattices;
- two Ir-Dy bonds per DTO sublattice;
- bond phases `exp[i (k' - k) . delta_ij]`.

The bond and orbital amplitudes are summed coherently before squaring and averaging over seeds and snapshots.

Outputs for each field:

```text
data/scattering_h{field}.hdf5
data/scattering_h{field}.csv
```

Both the raw matrix elements and the historical `gaussian_filter1d(sigma=1)` angular smoothing are saved. Smoothing is never substituted for the raw result.

## Stage 4: Boltzmann transport

`04_compute_transport.py` evaluates

```text
1/tau(k) = sum_k' |T_kk'|^2 delta(e_k' - e_k) [1 - cos(theta_kk')]
```

and

```text
sigma_int = sum_k tau(k) delta(E_F - e_k) |v_x(k)|^2 P_surface(k)
rho_int = 1 / sigma_int
```

It intentionally preserves the old Gaussian delta, momentum-angle transport factor, surface weighting, and radial velocity. The raw resistivity is compared with archived CSVs in `reference/transport/`.

Outputs include:

```text
data/rho_h{field}_raw.csv
data/rho_h{field}_smoothed.csv
data/transport_summary.csv
data/relaxation_times.csv
```

The resistivity is in arbitrary units because the absolute interface Kondo coupling was not fixed in the archived calculation.

## Stage 5: Plotting

`05_plot_transport.py` produces:

```text
figures/interface_resistivity.pdf
```

Panel a shows the unsmoothed resistivity. Panel b shows the historically smoothed result normalized to the zero-angle value. No noise replacement or additional symmetry mirroring is applied. Final figures are saved only as vector PDFs.

## Faithful historical conventions

The following choices are retained to reproduce the archived result:

- fixed `E_F = 5.142059246119027 eV` from the old nk100 filling calculation;
- top-kagome weight cutoff 0.5;
- energy selection window 0.01 eV;
- 11 primitive-cell slab and 24 basis functions per cell;
- radial finite-difference `v_kf`, without an hbar conversion;
- four DTO seeds; field 0.1 uses snapshots 4 through 19, while the other fields use snapshots 10 through 19;
- `S_q` normalization by `sqrt(64)`;
- additional matrix-element normalization by `sqrt(64*2)`;
- equal hopping amplitudes for all three t2g orbitals;
- historical nonstandard Gaussian energy delta with width 0.01 eV;
- momentum-angle factor `1 - cos(theta_kk')`;
- optional nonperiodic Gaussian smoothing along field angle.

## Issues to revisit, but not silently change

For future physics work, especially EIO/TTO with `S(q, omega)`, test the following separately from this reference implementation:

1. Replace the archived radial velocity with the full Cartesian group velocity.
2. Track bands by wavefunction overlap near crossings.
3. Verify the transposed distance indexing in the hopping cutoff.
4. Derive the `sqrt(64*2)` normalization explicitly.
5. Compare momentum-angle and velocity-angle transport vertices.
6. Determine whether the final surface factor double-counts localization already present in the projected matrix element.
7. Reconcile the coherent bond/sublattice cross terms with the compact analytical expression.
8. Replace the static elastic kernel by the dynamical inelastic kernel only after this reference calculation is locked down.

## Provenance

`reference/` contains only small numerical fingerprints needed to verify the clean implementation. Large DTO snapshots are not duplicated. Generated HDF5 intermediates and the local DTO input link are ignored by Git; the compact validation metadata, CSV results, plotting source, and final vector PDF are retained.
