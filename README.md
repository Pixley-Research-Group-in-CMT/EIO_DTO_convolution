# Dynamical EIO/TTO interface convolution

This repository computes the EIO surface resistivity produced by dynamic TTO
moments. It implements the gMFT spinon structure factor in Sec. 4.1, an
independent linear-spin-wave structure factor in Sec. 4.2, and the inelastic
interface relaxation rate in Sec. 5 of
`papers/gMFT.pdf`.

The active workflow is

```text
EIO real-space tight binding
    -> surface energies, velocities, and projected spinors
Sec. 4.1: gMFT tables -> spinon continuum and condensate terms
Sec. 4.2: requested field grid -> Zeeman seed -> classical minimum -> magnons
EIO projections + S_mu(q, epsilon_k' - epsilon_k)
    -> |T_kk'(omega)|^2
    -> tau(k), sigma_int, rho_int
    -> resistivity versus in-plane phi
```

Two DSSF branches are available:

- `02_compute_tto_structure_factor.py` implements the Sec. 4.1 spinon DSSF;
- `02_compute_tto_spin_wave_structure_factor.py` implements the Sec. 4.2
  one- and two-magnon DSSF.

The older static DTO snapshot script, `02_compute_dto_fourier.py`, remains as
a historical utility but is not part of this workflow.

## Inputs and defaults

The EIO Wannier model is included in `model/`.  The Sec. 4.1 spinon branch reads the
sibling folder

```text
../gMFT_in_plane_min/phi_*/field_diagnostics_min_energy.csv
```

Each angle is deliberately read only from its matching `phi_N` directory.  In
particular, the stray phi=1 rows also present in `phi_0` are not mixed into the
scan.  Rows must have `status=ok`, and every requested field must have all 31
angles from 0 through 30 degrees.

The default slice is `J_pm/J_zz=0.2` at the four fields with complete converged
angle scans: `h/J_zz = 0.01, 0.15, 0.20, 0.25`.  The supplied 0.03 and 0.05
scans contain failed saddle-point rows and are rejected if requested.  All
ratios can be selected on the Sec. 4.1 stage-2 command line.

The Sec. 4.2 spin-wave branch does not read these tables. It generates the
requested field directions internally and minimizes the classical spin energy
from a Zeeman-minimizing starting configuration.

The gMFT tables use `J_zz` as their energy unit whereas the EIO model uses eV.
The default conversion is

```text
J_zz = 0.17 meV
```

Set `--j-zz-ev` to use another material parameter.  The DSSF Lorentzian width
defaults to 0.02 meV (`--eta-ev 2e-5`) and the internal Brillouin-zone grid to
8 cubed points, matching the grid used to generate the supplied gMFT tables.

## Run

Install the packages in `requirements.txt`, then run:

```bash
python 01_generate_eio.py
python 02_compute_tto_structure_factor.py
python 03_compute_scattering.py
python 04_compute_transport.py
python 05_plot_transport.py
```

Run the high-field spin-wave branch with:

```bash
python 02_compute_tto_spin_wave_structure_factor.py
python 03_compute_scattering.py --theory spin-wave
python 04_compute_transport.py --theory spin-wave
python 05_plot_transport.py --theory spin-wave
```

### SLURM submission with selected fields

Edit `spin_wave_job.conf` to choose the external-field ratios, model constants,
output location, Python environment, and a label for the run.  Submit
with the helper so the configuration is snapshotted immediately:

```bash
./queue_swt.sh
```

The main editable entries are:

```bash
FIELDS=(0.15 0.20 0.25)
RUN_LABEL="spin_wave"
J_PM_RATIO=0.2
J_PMPM_RATIO=0.1
J_ZZ_EV=0.00017
ETA_EV=0.00002
DSSF_NK=8
Q_PERP_MODE="slice"
N_Q_PERP=8
STABILITY_TILT=0.01
FIELD_THETA_DEGREES=90
PHI_OFFSET_DEGREES=0.1
PHI_START_DEGREES=0
PHI_STOP_DEGREES=180
N_PHI=91
REFLECT_NEGATIVE_PHI=true
```

The color map shows only the evaluated `(phi, h)` grid points. Each cell uses
the fractional change `[rho(phi) - rho(0)] / rho(0)` for that field, with no
interpolation or plot-only smoothing. A fourth panel shows the same normalized
curves in polar coordinates and leaves every unevaluated angular sector empty.

`PHI_OFFSET_DEGREES=0.1` evaluates the SWT structure factor at `phi + 0.1`
degrees so symmetry-axis directions are avoided. The saved `phi_degrees`
coordinate and all plot ticks retain the nominal values (0, 2, ..., 180), while
the HDF5 file and conditions CSV also record the actual evaluated angle. Thus
the nominal `rho(0)` reference is the result evaluated at 0.1 degrees.

With `REFLECT_NEGATIVE_PHI=true`, the DSSF and scattering stages evaluate only
the configured nonnegative angles. The transport stage then copies every
positive-angle result to its negative-angle partner, enforcing
`rho(-phi) = rho(phi)`. The default two-degree grid from 0 through 180 uses
91 DSSF directions and writes 181 transport angles from -180 through 180;
phi=0 is not duplicated. The spin-wave conditions CSV and structure-factor
HDF5 retain only the explicitly evaluated nonnegative angles, while the
transport and relaxation-time CSV files contain the reflected full range.

Field values are dimensionless `h/J_zz` ratios.  Positional values remain
available as an optional one-off override, e.g.
`./queue_swt.sh 0.15 0.25`, without modifying the saved
configuration.

Set `RUN_LABEL` to the exact output-directory name for each submission. For
example, `RUN_LABEL="fine_grid"` writes to `$HOME/SWT_transport/fine_grid/`.
Use a new label for every submission: the job refuses to write into an existing
result directory so earlier data cannot be overwritten or mixed with a new run.

The helper saves a private configuration snapshot for every submission.  This
means `spin_wave_job.conf` can be edited immediately for the next job, even
while earlier jobs remain queued.  The exact configuration used is also copied
into the result directory as `submitted_spin_wave_job.conf`.

The cluster job writes only numerical data and run metadata under
`$HOME/SWT_transport/RUN_LABEL/`. It does not create a figure
or a `figures/` directory on the cluster.

`Q_PERP_MODE="slice"` preserves the original calculation and evaluates the
bulk TTO DSSF at `q_perp=0`. Set `Q_PERP_MODE="integrated"` to construct the
kagome-layer response used by the interface theory by averaging the unchanged
bulk solver over `N_Q_PERP` uniform momenta along one complete `[111]`
reciprocal period. The default `N_Q_PERP=8` includes `q_perp=0` and matches the
internal DSSF grid. Integrated mode is approximately `N_Q_PERP` times more
expensive in the external momentum evaluation. Every HDF5 output records the
selected mode, the number of samples, their reduced offsets, and the reduced
normal direction `(1,1,1)`.

After downloading the complete run directory, edit `DOWNLOADED_RESULTS` near
the top of `plot_swt_results.py`:

```python
DOWNLOADED_RESULTS = Path("~/Downloads/RUN_LABEL")
```

Then generate the figure locally with:

```bash
python plot_swt_results.py
```

A positional path still works as a one-off override. Both forms accept either
the downloaded run directory or its `data/` directory:

```bash
python plot_swt_results.py /another/downloaded/RUN_LABEL
```

The script creates the same figure location used by earlier runs:

```text
/path/to/downloaded/RUN_LABEL/figures/
    tto_spin_wave_interface_resistivity.pdf
```

For an output containing exactly three field values, the companion plotter
creates three separate heatmap rows with a shared color scale and three polar
panels with a shared radial scale:

```bash
python plot_swt_three_field_results.py
```

Edit `DOWNLOADED_RESULTS` near the top of that file, or pass a run path as its
optional argument. The middle-field signal is multiplied by 2 in both figures
and is labeled accordingly; the CSV data is never modified. The outputs are:

```text
/path/to/downloaded/RUN_LABEL/figures/
    tto_spin_wave_three_field_heatmap.pdf
    tto_spin_wave_three_field_polar.pdf
```

The provided configuration uses the cluster's `python/3.8.2` module and
`$HOME/eio_tto_env/bin/activate`.  A separate
configuration can also be selected without modifying the default document:

```bash
sbatch --export=ALL,JOB_CONFIG=/path/to/my_spin_wave_job.conf \
    submit_swt.slurm
```

Any finite nonnegative `h/J_zz` values can be submitted. The spin-wave driver
generates `N_PHI` directions from `PHI_START_DEGREES` through
`PHI_STOP_DEGREES` at the configured polar angle, then applies
`PHI_OFFSET_DEGREES` only while evaluating the field direction. Physical
stability is checked from the relaxed classical state and magnon spectrum
rather than from a gMFT convergence table.

To recompute existing HDF5 outputs, pass `--overwrite` to stages 1 through 3.
For example:

```bash
python 02_compute_tto_structure_factor.py \
  --j-pm 0.2 \
  --fields 0.01 0.15 0.20 0.25 \
  --j-zz-ev 0.00017 \
  --eta-ev 0.00002 \
  --q-perp-mode integrated \
  --n-q-perp 8 \
  --overwrite
```

### Three-sublattice first-BZ structure-factor maps

Two independent cluster workflows plot the Sec. 4.2 response for the three
interfacial kagome sublattices, `mu=1,2,3`, over the hexagonal two-dimensional
Brillouin zone perpendicular to cubic `[111]`. The `mu=0` component is neither
calculated nor shown. The plotted Cartesian axes are `x || [-1,0,1]` and
`y || [1,-2,1]` in inverse Angstrom. Each retained in-plane point is converted
to the FCC reduced coordinates expected by the SWT Hamiltonian.

The two configuration files expose the run label, exchange constants, plotted
component, and color-scale percentile:

```bash
RUN_LABEL="ssf_run_01"
J_PM_RATIO=0.20
J_PMPM_RATIO=0.0
J_ZZ_EV=0.0002
COMPONENT="total"
VMAX_PERCENTILE=100
Q_PERP_MODE="slice"
N_Q_PERP=8
```

Every job evaluates the fixed scan
`phi = 0.1, 30.1 degrees` and `h/J_zz = 0.25, 0.45, 1.5`. The fixed-energy
DSSF uses `omega_0 = 0.05 meV` and Lorentzian width `eta = 0.02 meV`. Both
workflows use a `101 x 101` bounding grid and an internal `8 x 8 x 8`
two-magnon grid. `COMPONENT` may be `total`, `bragg`, `one_magnon`, or
`two_magnon`. `VMAX_PERCENTILE` sets the common upper color limit for all nine
panels in each PDF.

Both map workflows support the same perpendicular-momentum projection used by
the transport solver. `Q_PERP_MODE="slice"` preserves the original
`q_perp=0` maps. Set `Q_PERP_MODE="integrated"` to average the bulk response
over `N_Q_PERP` uniform points spanning one complete `[111]` reciprocal
period. The HDF5 file records the selected mode and actual reduced offsets,
and every PDF title states whether it shows the slice or integrated response.
Integrated mode is approximately `N_Q_PERP` times more expensive in the
external momentum evaluation.

For the fixed-energy DSSF, edit `swt_ssf_job.conf` and submit it with:

```bash
./queue_swt_ssf.sh
```

The job writes to the configured run label:

```text
$HOME/SWT_SSF/RUN_LABEL/
    submitted_swt_ssf_job.conf
    run_parameters.txt
    data/swt_ssf.hdf5
    figures/swt_ssf_phi_0.pdf
    figures/swt_ssf_phi_30.pdf
    figures/swt_ssf_mu_sum_phi_0.pdf
    figures/swt_ssf_mu_sum_phi_30.pdf
```

For the energy-integrated static structure factor, edit
`swt_sssf_job.conf` and submit:

```bash
./queue_swt_sssf.sh
```

Its output is:

```text
$HOME/SWT_SSSF/RUN_LABEL/
    submitted_swt_sssf_job.conf
    run_parameters.txt
    data/swt_sssf.hdf5
    figures/swt_sssf_phi_0.pdf
    figures/swt_sssf_phi_30.pdf
    figures/swt_sssf_mu_sum_phi_0.pdf
    figures/swt_sssf_mu_sum_phi_30.pdf
```

Only points inside the hexagonal 2D BZ are evaluated; cells outside it are left
empty. Each per-sublattice PDF contains three field rows and the three
`mu=1,2,3` columns, with one shared color scale and the black-to-yellow `magma`
palette. The HDF5
file stores `one_magnon`, `two_magnon`, `bragg`, and `total` arrays with shape
`(2, 3, 3, 101, 101)`. Their axes are ordered as `(phi, h/J_zz, sublattice,
q_y, q_x)`. The file also stores the BZ mask, inverse-Angstrom axes, FCC
reduced momenta, all six relaxed ordered states and local Zeeman fields, and
the corresponding stability diagnostics.

For each phi, the additional `*_mu_sum_phi_*.pdf` contains one panel per field
for `sum_(mu=1)^3 S_mu`, with a color scale shared across the three fields. The
HDF5 file stores the same sums explicitly as `one_magnon_mu_sum`,
`two_magnon_mu_sum`, `bragg_mu_sum`, and `total_mu_sum`, with axes
`(phi, h/J_zz, q_y, q_x)`.

The static routine integrates Eqs. (99)-(100) analytically over energy rather
than sampling an omega grid. Consequently it has no `OMEGA_MEV` or `ETA_MEV`
setting and is independent of energy resolution. Both jobs start from the
Zeeman-minimizing configuration and perform the same multistart classical
relaxation and dynamical-stability check used by the transport SWT workflow.

Every submission snapshots its configuration and refuses to overwrite an
existing `RUN_LABEL` directory. Direct `sbatch` submission is also supported:

```bash
sbatch submit_swt_ssf.slurm
sbatch submit_swt_sssf.slurm
```

### Energy-resolved DSSF animations

`plot_swt_dssf_video.py` evaluates `S_mu(q, omega)` on the same `(111)` BZ,
field rows, angles, and `mu=1,2,3` columns used by the fixed-energy SSF maps.
The common animation axis runs from the smallest to the largest one-magnon
energy found across all six displayed field/angle conditions. Magnon modes are
reused across frames rather than recalculated independently at every omega.

Edit `swt_dssf_job.conf`:

```bash
RUN_LABEL="dssf_run_01"
J_PM_RATIO=0.20
J_PMPM_RATIO=0.0
J_ZZ_EV=0.0002
COMPONENT="one_magnon"
VMAX_PERCENTILE=95
N_OMEGA_FRAMES=60
FPS=10
VIDEO_FORMAT="gif"
```

`COMPONENT` may be `total`, `bragg`, `one_magnon`, or `two_magnon`. GIF is
the portable default and uses Pillow, which is already required by Matplotlib.
Set `VIDEO_FORMAT="mp4"` only when `ffmpeg` or `imageio-ffmpeg` is installed in
the cluster Python environment. Submit with:

```bash
./queue_swt_dssf.sh
```

The job writes:

```text
$HOME/SWT_DSSF/RUN_LABEL/
    submitted_swt_dssf_job.conf
    run_parameters.txt
    data/swt_dssf.hdf5
    videos/swt_dssf_phi_0.gif
    videos/swt_dssf_phi_30.gif
```

The HDF5 `intensity` axes are `(phi, h/J_zz, sublattice, omega, q_y, q_x)`.
It also stores the sampled omega axis, condition-resolved dispersion bounds,
ordered states, local Zeeman fields, and stability diagnostics. The same color
scale is used for every panel and frame within each video. Animating
`two_magnon` or `total` is substantially more expensive than `one_magnon`.

### Energy-resolved DSSF along the FCC high-symmetry path

`plot_swt_high_symmetry_dssf.py` evaluates the bulk Sec. 4.2 response along
the standard FCC path `Gamma-X-W-K-Gamma-L-U`. The horizontal coordinate is
the cumulative physical wave-vector distance in inverse Angstrom, while the
axis labels identify the symmetry points. The energy axis starts at zero and
extends through the largest one-magnon energy among the three fields. For
`two_magnon` and `total`, it extends to twice that value so the pair continuum
is not truncated.

Edit `swt_path_job.conf`:

```bash
RUN_LABEL="path_run_01"
FIELDS=(0.25 0.45 1.5)
FIELD_PHIS_DEGREES=(0.1 30.1)
J_PM_RATIO=0.20
J_PMPM_RATIO=0.0
J_ZZ_EV=0.0002
COMPONENT="one_magnon"
VMAX_PERCENTILE=99
POINTS_PER_SEGMENT=40
N_OMEGA=200
ETA_MEV=0.02
DSSF_NK=8
```

Submit a snapshotted configuration with:

```bash
./queue_swt_path.sh
```

The job writes:

```text
$HOME/SWT_DSSF_path/RUN_LABEL/
    submitted_swt_path_job.conf
    run_parameters.txt
    data/swt_dssf_high_symmetry.hdf5
    figures/swt_dssf_high_symmetry_sublattices_phi_0p1.pdf
    figures/swt_dssf_high_symmetry_mu_sum_phi_0p1.pdf
    figures/swt_dssf_high_symmetry_sublattices_phi_30p1.pdf
    figures/swt_dssf_high_symmetry_mu_sum_phi_30p1.pdf
```

For each of `phi=0.1` and `30.1` degrees, the first PDF has the three configured
fields as rows and `mu=1,2,3` as columns. The second has one field per row and
displays `sum_(mu=1)^3 S_mu(q,omega)`. Each file uses one shared black-to-yellow
color scale. The HDF5 intensity axes are `(field_phi_degrees, h/J_zz,
sublattice, omega, q_path)` and the file also records the summed response,
FCC-reduced and Cartesian momenta,
symmetry-point positions, ordered states, dispersion bounds, and stability
diagnostics. `COMPONENT` may be `total`, `bragg`, `one_magnon`, or
`two_magnon`.

### Four-band magnon dispersion maps

`plot_swt_magnon_dispersion.py` evaluates all four positive magnon energies
over the hexagonal 2D BZ perpendicular to `(111)`. Edit
`swt_magnon_job.conf`:

```bash
RUN_LABEL="magnon_run_01"
J_PM_RATIO=0.20
J_PMPM_RATIO=0.0
J_ZZ_EV=0.0002
```

Then submit with:

```bash
./queue_swt_magnon.sh
```

The job writes:

```text
$HOME/SWT_magnon/RUN_LABEL/
    submitted_swt_magnon_job.conf
    run_parameters.txt
    data/swt_magnon_dispersion.hdf5
    figures/swt_magnon_dispersion_phi_0.pdf
    figures/swt_magnon_dispersion_phi_30.pdf
```

Each PDF contains the three fixed field values `h/J_zz = 0.25, 0.45, 1.5`
as rows and the four magnon bands as columns. Both angles use one common magnon
energy scale in meV and the same black-to-yellow `magma` palette as the
structure-factor maps. The HDF5 `band_energy_mev` axes are
`(phi, h/J_zz, band, q_y, q_x)`. Bands are ordered by energy independently at
each momentum; an exact band crossing is not relabeled by eigenvector
continuity. The file also stores the BZ mask, momenta, relaxed ordered states,
local Zeeman fields, and stability diagnostics.

## Physics implemented

For each gMFT row, stage 2 reconstructs
`s_minus_mu = s_x_mu - i s_y_mu` and reads `lambda`, `rho`, `condensed`, and
`k_min` directly.  It evaluates Eqs. (85)-(87):

- the two-spinon continuum;
- both one-condensate channels when `rho > 0`;
- the two-condensate Bragg contribution at reciprocal `q=0`.

The soft mode is removed from the continuum with the same dimensionless
`1e-10` tolerance used by the source solver.  The DSSF is evaluated directly
at `q=k-k'` and `omega=epsilon_k'-epsilon_k`, which analytically performs the
omega integral and energy delta in Eq. (101).

Stage 3 implements the local, incoherent bond sum in Eq. (102), including the
three TTO interfacial kagome sublattices (gMFT mu=1,2,3) and their two EIO
neighbors.  Stage 4 therefore does not apply the old elastic
`delta(epsilon_k'-epsilon_k)` a second time.

### Sec. 4.2 ordered-state preparation

For every requested field magnitude and direction, the code projects the global
field onto the four local Ising axes. It constructs a fixed-length spin state
with `S_z_mu = S sign(h_z_mu)`, which exactly minimizes the Zeeman term; a zero
local projection uses `+z` as a deterministic representative.
If the Zeeman state has negative tangent-space curvature, two additional starts
are tilted slightly in opposite directions along its most unstable mode. These
and 12 deterministic random starts seed a multistart minimization of the full
four-sublattice classical energy at the configured `J_pm/J_zz` and
`J_pmpm/J_zz`. The relaxed spins, field projections, seed spins, initial and
final energies, seed and final tangent curvatures, whether the tilt was used,
residual torque, and maximum imaginary frequency are saved in
`data/tto_spin_wave_conditions.csv`.

The spin-wave branch accepts the full bond-dependent `J_pmpm/J_zz` term from
Eqs. (1)-(2).  Its complex `gamma_mu_nu` phases are converted to a real local
Cartesian exchange tensor and included consistently in both the classical
minimization and the quadratic Bogoliubov Hamiltonian.  Set `J_PMPM_RATIO` in
`spin_wave_job.conf`, or use `--j-pmpm` when running stage 2 directly.  The
direct stage-2 default is zero, which reproduces the Sec. 4.2 specialization
in the paper; cluster submissions use the value in `spin_wave_job.conf`.

The quadratic Hamiltonian includes the Zeeman diagonal term associated with
Eq. (9), in addition to Eqs. (93)-(96).  The one-magnon transverse response and
two-magnon longitudinal response follow Eqs. (99)-(100).  Midpoint BZ
quadrature is used for the two-magnon integral so the isolated Goldstone point,
which has zero measure in the thermodynamic integral, is not assigned a finite
grid weight.  A `1e-8 J_zz` diagonal regularization is retained only for stable
paraunitary normalization and is recorded in the HDF5 metadata.

## Outputs

The main products are:

```text
data/tto_dynamic_structure_factor.hdf5
data/tto_gmft_conditions.csv
data/tto_dynamic_scattering.hdf5
data/tto_dynamic_scattering.csv
data/tto_transport_summary.csv
data/tto_relaxation_times.csv
figures/tto_interface_resistivity.pdf
data/tto_spin_wave_structure_factor.hdf5
data/tto_spin_wave_conditions.csv
data/tto_spin_wave_scattering.hdf5
data/tto_spin_wave_scattering.csv
data/tto_spin_wave_transport_summary.csv
data/tto_spin_wave_relaxation_times.csv
figures/tto_spin_wave_interface_resistivity.pdf
```

The HDF5 files retain the continuum and both condensate components separately.
The interface exchange magnitude is set to one, so absolute resistivity remains
in arbitrary units. Cluster submissions stop after the spin-wave CSV and HDF5
outputs; `plot_swt_results.py` creates the final vector PDF locally.

Run the focused tests with:

```bash
pytest -q
```
