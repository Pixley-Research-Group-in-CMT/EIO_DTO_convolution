# EIO/TTO spin-wave Python package

Functions for calculating EIO/TTO interface resistivity, animating the dynamical spin structure factor along an FCC symmetry path, and animating the on-shell transport kernel. The optional kernel panels beneath resistivity plots are included.

The package contains Python functions, the two EIO hopping models needed by those functions, installation metadata, and this documentation. It has no scheduler scripts, shell configuration files, job drivers, or command-line entry points. gMFT, static SSF/SSSF maps, and unrelated workflows are excluded. Importing the package does not start a calculation or create result files. All outputs go to directories you pass explicitly.

## Installation

Requires Python 3.10 or newer. From the extracted folder containing `pyproject.toml`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Python must already be installed. pip installs NumPy, SciPy, pandas, h5py, Matplotlib, and Pillow. GIF output needs no external encoder. For MP4, install `python -m pip install '.[mp4]'` or provide a system FFmpeg installation. The functions can be called from a notebook or your own Python script.

## Shared model and numerical parameters

```python
from eio_tto_swt import SpinWaveParameters

parameters = SpinWaveParameters(
    j_pm=0.2,
    j_pmpm=0.0,
    j_zz_ev=0.00017,
    g_z=10.0,
    g_perp=0.0,
    delta1_per_mev=0.0,
    delta2_per_mev=0.0,
    eta_mev=0.02,
    nk=8,
    q_perp_mode="integrated",
    n_q_perp=8,
)
```

`j_pm` and `j_pmpm` are exchange ratios relative to Jzz. Field values are bare μB|B|/Jzz, before the field couplings are applied. `j_zz_ev` is in eV, `eta_mev` is in meV, and the quadratic coefficients are in inverse meV. `nk` controls the internal magnon momentum grid; the electronic surface grid is nk24. Other parameter defaults are `lattice_constant=10.154` Å, `spin_length=0.5`, `regularization=1e-8`, `stability_tilt=0.01`, and `random_starts=8`. `random_starts` controls the movie solvers; transport retains the repository minimizer's default multistart count.

`q_perp_mode="slice"` evaluates zero perpendicular momentum. `"integrated"` takes a normalized perpendicular-momentum average with `n_q_perp` samples. This projection applies to transport and the kernel movie. The path movie displays bulk spectra without perpendicular averaging. Its angle is θ=90°; transport also exposes `theta`.

## Resistivity calculation and plots

```python
from eio_tto_swt import compute_transport, plot_resistivity

result = compute_transport(
    "results/transport",
    fields=(0.25, 0.45, 1.5),
    phi_start=0.0,
    phi_stop=180.0,
    n_phi=90,
    phi_offset=0.1,
    component="total",
    eio_model="symmetrized",
    parameters=parameters,
)

pdf = plot_resistivity(
    result,
    layout="rows",
    show_kernel_plots=True,
    kernel_phis=(0.0, 30.0),
)
print(pdf)
```

`n_phi` counts intervals: this example computes 91 endpoint-inclusive nominal angles. `phi_offset` is added during the calculation; nominal angles remain the plot coordinates. `reflect_negative_phi=True` optionally mirrors positive-angle transport results onto negative angles.

`component` accepts `total`, `one_magnon`, `one_magnon_1` through `one_magnon_4`, or `two_magnon`. Numbered bands are sorted by energy independently at each momentum. Elastic Bragg scattering is always excluded.

The result provides `summary_file`, `scattering_file`, and `surface_states_file` paths. Numerical outputs, conditions, and relaxation times are saved under `results/transport/data/`. The default figure is `results/transport/figures/resistivity.pdf`. Absolute resistivity uses interface exchange magnitude squared set to one and is in arbitrary units. The normalized response is `[rho(phi)-rho(0)]/rho(0)`, with the nominal zero reference evaluated at `phi_offset` when an offset is used.

`layout` can be `rows`, `overlay`, or `movie`. Static figures are vector PDFs. For the resistivity field movie:

```python
movie = plot_resistivity(
    result,
    layout="movie",
    fps=2,
    show_kernel_plots=True,
    kernel_phis=(0.0, 30.0),
)
```

This saves `figures/resistivity.gif`. Optional kernel maps appear below each field's resistivity row, below the overlaid figure, or below the matching field's movie frame. They use the **same selected transport component** and the saved pair scattering, multiplied by `1-cos(theta_kk')`. They sum all incoming states and group equivalent momentum transfers. A field's angle panels share a scale; different fields can have different scales.

Kernel angles must be nominal angles explicitly sampled in the scattering file. Reflected-only angles cannot be selected. Titles show actual evaluated angles, including offsets. No DSSF recalculation or extra EIO file is needed to add these panels.

To replot an existing run, pass its directory or `data/` subdirectory instead of `result`:

```python
plot_resistivity(
    "results/transport",
    layout="overlay",
    show_kernel_plots=True,
    output="results/transport/figures/with_kernels.pdf",
)
```

`output` chooses the complete figure filename. Static output must end in `.pdf`; resistivity movie output must end in `.gif`. Plotting replaces the selected figure file. Calculations protect existing numerical outputs unless `overwrite=True` is explicitly passed. EIO surface states are reused only for the matching model; choose a new directory when changing models.

## DSSF path movie

```python
from eio_tto_swt import make_dssf_path_movie

movies = make_dssf_path_movie(
    "results/path",
    field_start=0.25,
    field_stop=1.5,
    n_field_frames=31,
    phi_degrees=(0.1, 30.1),
    component="total",
    points_per_segment=40,
    n_omega=200,
    parameters=parameters,
    fps=10,
    video_format="gif",
)
print(movies.data_file)
print(movies.movie_files)
```

The inclusive field scan follows Γ–X–W–K–Γ–L–U. At each of the two angles it writes a movie with μ=1,2,3 spectra and another with their sum. `component` has the transport choices listed above. A fixed color scale is used throughout each movie. `vmax_percentile=99` is the default clipping percentile; select 100 to avoid clipping. The energy range includes the selected one-magnon bands or the full two-magnon range. Unstable conditions stop this workflow with a diagnostic.

Outputs are `data/swt_dssf_high_symmetry_field_movie.hdf5` and four files under `videos/`. Other arguments are `omega_chunk_size=8`, `dpi=100`, and `overwrite=False`. The HDF5 intensity axes are `(phi, field, sublattice, omega, q_path)`.

## SWT kernel movie

```python
from eio_tto_swt import make_kernel_movie

movie = make_kernel_movie(
    "results/kernel",
    field_start=1.0,
    field_stop=1.1,
    n_field_frames=11,
    phi_degrees=(0.0, 30.0),
    eio_model="symmetrized",
    parameters=parameters,
    fps=10,
    video_format="gif",
)
```

The two angle panels animate together. This standalone kernel always uses the total one- plus two-magnon response and all incoming surface states, excluding elastic Bragg scattering. Each frame has its own scale, shared by its two angles. Read the color scale before comparing fields.

Outputs are `data/swt_kernel_movie.hdf5` and `movies/swt_kernel_movie.gif`. HDF5 pair-kernel axes are `(phi, field, incoming_k, outgoing_k)`; grouped display values and stability diagnostics are also retained. Invalid spin-wave spectra are saved as NaNs and displayed as gray “Unstable” panels; other errors stop the calculation. Other arguments are `vmax_percentile=100`, `marker_size=20`, `dpi=120`, and `overwrite=False`.

## Electronic surface states

Transport and kernel movies generate the EIO states they need. Both `eio_model="original"` and `"symmetrized"` are supported. To generate states separately:

```python
from eio_tto_swt import generate_surface_states

states = generate_surface_states("results/electronic", eio_model="symmetrized")
```

Pass `eio_states=states` to `compute_transport` or `make_kernel_movie` to reuse them in a new output directory. Reusing a model-mismatched file is rejected. `generate_surface_states` accepts `nk=24` or `nk=36`; the two coupled workflows require nk24 states.

## Cost and a small first calculation

The first electronic calculation diagonalizes an 11-layer slab at 576 momenta. Two-magnon calculations and integrated perpendicular sampling are more expensive than one-magnon and slice calculations. Production scans can take hours; GIF encoding also retains frames in memory.

For a quick execution check, use `SpinWaveParameters(nk=2, q_perp_mode="slice")`, transport `fields=(1.0, 1.1)`, `phi_stop=30`, `n_phi=2`, and two movie field frames. For the path movie also use `points_per_segment=2`, `n_omega=8`, and `dpi=30`. These coarse grids check execution; establish convergence before interpreting quantitative results.

Functions do not resume partial jobs. Completed numerical outputs are protected by default; to recompute an interrupted run, supply `overwrite=True` or a new output directory. Source modules and model files inside the installed package are read-only inputs; outputs are always written to the selected directories. Movie frames are encoded directly, without PNG or SVG sidecars.

## Scientific reference and limits

The only scientific reference used here is the bundled [EIO/TTO note](docs/EIO_TTO.pdf), *Theory for a phenomenological description of the magnetoresistance of a EIO/TTO heterostructure*, October 12, 2026.

Section 1, Eqs. (1)–(21), supplies the exchange, field couplings, classical-state expansion, and spin-wave response. Section 3, Eqs. (23)–(25), supplies the signed-energy-transfer kernel:

```text
Kij = (1-cos(theta_ij)) sum_mu W_mu,ij S_mu(ki-kj, epsilon_j-epsilon_i)
```

Energy conservation eliminates the omega integral. The displayed kernel is not an unweighted energy integral of the DSSF. Folding transfers into the interface Brillouin zone and summing coincident transfers are display operations without interpolation.

The note gives gz approximately 10, gperp less than 1, and a quadratic-coupling magnitude scale near `(1.4 meV)^-1`; it does not fix exact values or signs. The defaults `g_perp=0` and `delta1=delta2=0` are configurable model choices inherited from the repository. Section 2 is unfinished: the EIO models, surface-state selection, broadening, quadrature, tolerances, and perpendicular averaging are repository implementation choices. The package does not claim exact reproduction of the note's figures.

Numerical helpers are available in `eio_tto_swt.spin_wave`, `eio_tto_swt.projection`, and `eio_tto_swt.eio`. Modules beginning with `_` support the public functions and are internal implementation details.

## Validation

The package was built and installed in a separate directory, then tested away from the research repository. Seven API integration checks passed, including generation of symmetrized nk24 electronic states, transport calculation, both movie calculations, PDF layouts, combined resistivity/kernel GIF encoding, and output protection. Reduced numerical grids were used for execution checks; these checks do not establish production convergence.
