# EIO/TTO spin-wave calculation package

A standalone source package for exactly three workflows:

- `swt_transport`: EIO interface resistivity versus in-plane field angle, saved as vector PDF.
- `dssf_path_movie`: field-dependent dynamical spin structure factor (DSSF) along the FCC path Γ–X–W–K–Γ–L–U.
- `swt_kernel`: a field movie of the on-shell transport kernel in the interface momentum plane.

All required numerical sources and both EIO hopping models are included. No sibling repositories, precomputed data, or gMFT tables are needed. Shared helper filenames inherited from the original repository are internal modules, not additional workflows. Static SSF/SSSF maps, other DSSF movies, and gMFT calculations are omitted.

## Install

Use Python 3.10 or newer on Linux or macOS (the transport driver uses POSIX file locking). Extract the ZIP, enter its `eio_tto_swt` directory, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GIF output works with the supplied dependencies. For MP4, install FFmpeg on your system or run `python -m pip install imageio-ffmpeg`, then select `VIDEO_FORMAT="mp4"`. Use `python run.py --help` for the launcher and each plotting script's `--help` for all numerical options. This is a portable source distribution; installation with `pip install .` is not required or supported.

## 1. Resistivity plots

Edit `swt_transport_job.json`, then run from the package directory:

```bash
python run.py swt_transport
```

The launcher computes EIO surface states, the spin-wave DSSF at the electronic energy transfers, scattering matrix elements, relaxation times, conductivity, and resistivity. The example uses fields `[0.25, 0.45, 1.5]` and a row layout. Change `run_label` for each independent calculation.

Main controls:

| Setting | Meaning |
| --- | --- |
| `fields` | Bare Zeeman energies μB\|B\|/Jzz, before field couplings |
| `phi_start_degrees`, `phi_stop_degrees`, `n_phi` | Inclusive nominal angle range; `n_phi` counts intervals, giving `n_phi + 1` angles |
| `phi_offset_degrees` | Added only to the angle evaluated by the solver; labels retain nominal angles |
| `reflect_negative_phi` | Optionally copy positive-angle transport results to their negative partners |
| `transport_component` | `total`, `one_magnon`, `one_magnon_1` … `one_magnon_4`, or `two_magnon` |
| `resistivity_plot_mode` | `rows` or `overlay` for vector PDF |
| `nk` | Internal spin-wave integration grid, not the EIO surface grid |
| `q_perp_mode`, `n_q_perp` | `slice` at zero normal momentum or a normalized `integrated` average |
| `eio_model` | `original` or `symmetrized` hopping model |

With the example paths, outputs are:

```text
results/plots/transport_example.pdf
results/transport/transport_example/manifest.json
results/transport/transport_example/source/
results/transport/transport_example/data/surface_states_summary_nk24.hdf5
results/transport/transport_example/data/tto_spin_wave_structure_factor.hdf5
results/transport/transport_example/data/tto_spin_wave_scattering.hdf5
results/transport/transport_example/data/tto_spin_wave_transport_summary.csv
results/transport/transport_example/data/tto_spin_wave_relaxation_times.csv
```

The directory also contains conditions, scattering CSVs, and stage logs. A repeated identical command resumes completed stages; changed configurations or source hashes require a new run label. To regenerate a plot from an existing run:

```bash
python plot_swt_results.py results/transport/transport_example
```

Absolute resistivity uses interface exchange magnitude squared set to one and is in arbitrary units. The normalized response is `[rho(phi)-rho(0)]/rho(0)`. With an angle offset, the nominal zero reference is evaluated at that offset. Four one-magnon bands are sorted by energy independently at each momentum, rather than tracked continuously through crossings.

### Optional kernels underneath resistivity

Set `"show_kernel_plots": true` and `"kernel_phi_degrees": [0.0, 30.0]` in the transport JSON. The option defaults to `false`. Each field's resistivity row gains kernel maps directly below it; overlay mode puts kernel rows below the overlaid curves. Resistivity movies show kernels synchronized to each field frame, with a scale shared by that frame's selected kernel angles.

The maps are reconstructed from the saved scattering HDF5 and use the same component selected for resistivity, including numbered one-magnon bands. They multiply the saved pair scattering by `1-cos(theta_kk')`, then sum all incoming states and group momentum transfers using the standalone kernel movie's convention. No spin-wave recalculation or extra EIO file is needed. These embedded maps therefore use the transport component, whereas the separate `swt_kernel` workflow always uses the total one- plus two-magnon response.

Angles must be explicitly sampled nominal angles in the scattering data. Reflected-only angles cannot be selected. Panel titles report actual evaluated angles, including any offset, along with nominal angles when they differ. Replot an existing run without changing its calculation or manifest:

```bash
python plot_swt_results.py results/transport/transport_example --show-kernel-plots --kernel-phis 0 30
python plot_swt_results.py results/transport/transport_example --no-kernel-plots
```

The lower-level `05_plot_transport.py` also accepts `--show-kernel-plots --kernel-phis 0 30`. Existing job restart rules still apply if the JSON changes; use the replot command to change only the display. Static combined figures remain vector PDF; movies remain GIF.

## 2. DSSF path field movie

Edit `swt_path_movie_job.conf`, then:

```bash
python run.py dssf_path_movie
```

`MU_B_B_START`, `MU_B_B_STOP`, and `N_FIELD_FRAMES` specify a uniform field scan including both endpoints. `FIELD_PHIS_DEGREES` chooses the two in-plane angles. `COMPONENT` has the same choices as transport. `POINTS_PER_SEGMENT`, `N_OMEGA`, `DSSF_NK`, and `ETA_MEV` control path, energy, internal momentum resolution, and Lorentzian broadening. `FPS`, `DPI`, and `VIDEO_FORMAT` control the movie. The supplied configuration preserves the repository's fine low-field scan and its field-coupling choices; it is not a fit prescribed by the note.

Outputs under `results/dssf_path_movie/RUN_LABEL/`:

```text
data/swt_dssf_high_symmetry_field_movie.hdf5
videos/swt_dssf_high_symmetry_sublattices_phi_0p1.gif
videos/swt_dssf_high_symmetry_mu_sum_phi_0p1.gif
videos/swt_dssf_high_symmetry_sublattices_phi_30p1.gif
videos/swt_dssf_high_symmetry_mu_sum_phi_30p1.gif
```

Names follow the selected angles and encoding. Sublattice movies show μ=1,2,3; sum movies show their sum. Each movie uses a fixed color scale across field frames. The energy range covers the selected magnon bands; `total` and `two_magnon` extend to twice the full one-magnon maximum. These are bulk path spectra, without the interface perpendicular-momentum average. Unstable conditions stop this workflow with a diagnostic.

## 3. SWT kernel field movie

Edit `swt_kernel_job.conf`, retain `PLOT_MODE="movie"`, then:

```bash
python run.py swt_kernel
```

The field-axis and movie controls have the same meaning as above. Both configured angles appear together. This workflow always includes the one- and two-magnon inelastic response and all incoming surface states. `Q_PERP_MODE` and `N_Q_PERP` control the interface projection. Each frame has its own color scale shared by the two angle panels; compare colors within a frame, and read its scale before comparing frames.

Outputs under `results/swt_kernel/RUN_LABEL/`:

```text
data/surface_states_summary_nk24.hdf5
data/swt_kernel_movie.hdf5
movies/swt_kernel_movie.gif
source/
source_sha256.json
```

The HDF5 file retains pair kernels with axes `(phi, field, incoming_k, outgoing_k)`, grouped display values, momenta, signed energy transfers, ordered states, and stability diagnostics. Invalid spin-wave conditions appear as gray “Unstable” panels and NaN numerical entries; other errors stop the job. The DSSF uses the original momentum transfer; folding into the interface Brillouin zone and summing coincident points are display operations without interpolation. Existing movie/data files require a new output directory (or explicit `--overwrite` when using the plotting script directly).

For either movie, choose a custom destination with `--output-root /path/to/new/run` or another configuration with `--config /path/to/settings.conf`. `--dry-run` prints the command without computing.

## Scientific conventions and reference

The only scientific reference bundled and used here is [the EIO/TTO note](papers/EIO_TTO.pdf), *Theory for a phenomenological description of the magnetoresistance of a EIO/TTO heterostructure*, dated October 12, 2026.

- Section 1, Eqs. (1)–(5): bond-dependent exchange and linear/quadratic magnetic-field couplings. Fields are μB\|B\|/Jzz; exchange ratios are relative to Jzz. `j_zz_ev`/`J_ZZ_EV` are in eV; `ETA_MEV` is in meV, whereas transport `eta_ev` is in eV. Quadratic coefficients are in inverse meV.
- Eqs. (9)–(21): classical-state relaxation, the quadratic bosonic problem, and one-/two-magnon response. Elastic Bragg scattering is excluded throughout the packaged calculations.
- Section 3, Eqs. (23)–(25): electronic energy conservation eliminates the omega integral, giving `Kij = (1-cos(theta_ij)) sum_mu W_mu,ij S_mu(ki-kj, epsilon_j-epsilon_i)`. The kernel movie displays this signed-energy-transfer kernel, rather than an unweighted integral of the DSSF.

The note gives `g_z` approximately 10, `g_perp < 1`, and a quadratic-coupling magnitude scale near `(1.4 meV)^-1`; it does not fix their exact values and signs. The examples preserve the checkout's `g_perp=0` and `delta1=delta2=0`, which are configurable model choices. The note's Section 2 is unfinished: EIO hopping files, surface-state selection, quadrature, broadening, minimization tolerances, and perpendicular averaging are numerical choices inherited from this repository, not prescriptions supplied by that section. Its Figure 2 also has differing Jzz values between panel text and caption, so this package does not claim exact reproduction of the note's figures.

## Cost, checks, and cluster use

First try a small field/angle grid and few movie frames. Lower `DSSF_NK`/`nk`, `POINTS_PER_SEGMENT`, and `N_OMEGA` for workflow checks; then check convergence before using quantitative results. `total`/`two_magnon` and integrated perpendicular sampling are substantially more expensive. EIO surface generation diagonalizes an 11-layer slab on the configured nk24 grid and can dominate a first run; compatible states are reused. Production runs may take many hours. GIF encoding can require substantial memory because frames are retained in memory.

For SLURM, the three queue/submit pairs are included: `queue_swt.sh`, `queue_swt_path_movie.sh`, and `queue_swt_kernel.sh`. Adapt partition, resources, Python module, environment activation, and output locations to your cluster before submission; the inherited scripts use home-directory result roots and cluster-specific defaults. Local `run.py` needs no scheduler. Config files for the local movie launcher accept literal assignments and arrays, without shell expansion or commands; the cluster scripts source them as shell files.

Static figures are vector PDF only; movie frames are encoded directly into GIF/MP4 without PNG or SVG sidecars. All plotting sources remain included. `BUILD_INFO.json` records source-file hashes from the packaging checkout, and `SHA256SUMS.json` records packaged-file hashes. The parent repository retains `tools/build_swt_package.py` to rebuild the distribution.

To run the packaged checks:

```bash
python -m pip install -r requirements-dev.txt
MPLBACKEND=Agg python -m pytest -q tests
```

The checks cover configuration validation, component contraction, PDF layouts, GIF encoding, model-cache selection, and unstable-frame handling using synthetic electronic states. They do not establish convergence of a production scan.
