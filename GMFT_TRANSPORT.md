# General gMFT transport in one cluster job

Edit `gmft_transport_job.json`, then submit from this checkout:

```bash
bash queue_gmft_transport.sh
```

The helper snapshots the JSON into the configured log run directory immediately,
so later edits do not change queued jobs. It also places the Slurm standard
output and error files there. The SLURM file defaults to one node, 16 CPUs,
32 GB and 24 hours. Adjust
these resource requests for the cluster. It uses `python/3.8.2` and
`$HOME/eio_tto_env/bin/activate`, as the existing transport jobs do; override
`PYTHON_MODULE`, `PYTHON_ENV_ACTIVATE` or `PYTHON_BIN` as needed. The environment
needs `requirements.txt` and the updated sibling `gMFT_for_QSI` source. No
cluster submission is performed by writing these files.

`gmft_source` is relative to the EIO_TTO checkout, or an absolute path. Copy both
repositories to the cluster, including the updated `gmft_field_general.py`.
Change only `run_label` for each independent submission. The fixed `log_root`
and `plot_root` settings are combined with that label automatically:

```json
"run_label": "run_35",
"log_root": "~/gMFT_transport_log",
"plot_root": "~/gMFT_transport_plots",
"resistivity_plot_mode": "overlay",
"resistivity_movie_fps": 2.0
```

This writes data and logs under `~/gMFT_transport_log/run_35`. Overlay mode
writes `~/gMFT_transport_plots/run_35.pdf`; movie mode writes
`~/gMFT_transport_plots/run_35.gif`. Root paths may be
absolute or expanded from `~`. Older saved manifests containing the former
`run_directory` and `plot_directory` keys remain readable.

The main configuration values are:

- `fields`: bare Zeeman energies `mu_B |B| / Jzz`, before any g factors.
- `J_pm_ratio`: `J_pm/Jzz`. `J_pmpm_ratio` must be exactly zero. The Ansatz
  also fixes `J_zpm=0` and uniform zero flux.
- `g_z`, `g_perp`, `delta1_per_mev`, `delta2_per_mev`: general local field
  couplings. The deltas are in inverse meV and are converted to solver units
  using `1000*j_zz_ev`. The default deltas are nonzero.
- `j_zz_ev`, `eta_ev`: physical exchange scale and Lorentzian width in eV.
- `resistivity_plot_mode`: `overlay` puts every configured field in the same
  vector PDF; `movie` writes a GIF with one frame per configured field.
  `resistivity_movie_fps` controls the GIF playback speed.
- `theta_degrees`: 90 for an in-plane field. Directions use the gMFT source's
  `magnetic_field_direction` convention.
- `phi_start_degrees`, `phi_stop_degrees`, and `n_phi` set the nominal angular
  scan. `n_phi` is the number of equal intervals, so both endpoints are kept
  and the job evaluates `n_phi+1` nominal angles. For example,
  `phi_start_degrees=0`, `phi_stop_degrees=60`, and `n_phi=30` produce
  `0,2,4,...,60` degrees.
- `phi_offset_degrees`: 0.05 by default. It is added to each nominal
  nonnegative angle during the gMFT calculation, so the example above is
  evaluated at `0.05,2.05,4.05,...,60.05` degrees to avoid symmetry axes.
  Saved trial records, saddle NPZ files, HDF5 files and transport tables record
  both `phi_degrees` (nominal) and `evaluated_phi_degrees`. Plot axes remain
  nominal, and plot titles show the scan and applied offset.
- `nk`: shared MF and DSSF quadrature grid. `niter`, `tol`, `lambda_tol`, `mix`
  control the existing general solver.
- `convergence_criterion`: `"full"` checks the complex transverse spins and
  signed s_z. `"amplitudes"` checks only signed s_z and the transverse modulus:
  `max(max|s_z_target-s_z|, max||s_minus_target|-|s_minus||) < tol`.
  The editable example configuration now selects `"amplitudes"`; the solver
  and configurations omitting this setting still default to `"full"`.
  Complex spin updates and the backtracking rule are retained. This option
  does not freeze phases, replace s_minus by its modulus in the spectrum, or
  certify phase/DSSF convergence. For g_z=0, a polarized_x seed may satisfy the
  amplitude condition immediately; its full residual is still reported.
  Trial JSON, all_trials.csv, winning_saddles.csv and saddle NPZ files record
  `convergence_criterion`, `residual_full`, `residual_amplitudes` and
  `fully_converged`. Plot titles identify amplitude-only convergence.
  The spectrum and matrix weights depend on complex spins in the current
  finite-grid calculation, even though the local prefactor is |s_minus|^2.
  Check momentum-grid/DSSF sensitivity when using this relaxed criterion.
- `random_starts`: 50 by default, **plus** one `polarized_x` start at every
  field and every configured nominal angle. Each start is independent; no
  previous field or angle is used as a seed.
- `seed`: reproducible random conditions independent of worker scheduling.
- `q_perp_mode`: `slice` or `integrated`, with `n_q_perp` samples for the latter.
- `transfer_chunk_size`: 128 by default; reduce it if memory is tight.

With `F` fields, the scan runs
`(n_phi+1)*(random_starts+1)*F` solves. The editable default therefore runs
`31*51*F` solves. Trial processes share the
allocated CPUs, with one numerical-library thread per process to avoid nested
threading. Only this multistart phase uses the process pool. Subsequent stages
run serially; they reuse EIO states, chunk the DSSF momentum transfers, write
each DSSF condition directly to disk, and read the DSSF by condition during
scattering contraction. Runtime depends on convergence and grid size; the
24-hour allocation is not a runtime estimate.

## Physics and checks

`02_compute_tto_structure_factor.py --general-saddles RUN_DIRECTORY` reads
lossless winning saddle files. The new `tto_general_gmft.py` evaluates the
two-band response in Sec. 4.1, Eqs. (92), (97)-(99) of the updated
`gMFT_for_QSI/docs/gMFT.pdf`. Its local transport response is
`Sxx+Syy=(S+-+S-+)/2`; the phase factors cancel for these same-link traces.
Both bands and their spectral-projector diagonal weights contribute. This is
the contraction required by the existing spin-independent EIO transport
weights, not an export of the full sublattice/spin tensor.

Continuum, one-condensate, and elastic two-condensate terms are saved
separately. Condensate weights come directly from the solver's density
matrices; all occupied soft momenta are included. Their sum has trace `2*rho`,
which reproduces `rho*I` in the twofold-degenerate scalar limit. This extends
the single-minimum expressions to the solver's equally occupied soft modes.
The total retains the elastic term, matching the existing spinon workflow.

Every accepted trial has a finite energy and the solver's `converged=True`
under the selected convergence criterion. `fully_converged` separately records
whether the full complex fixed-point residual is below tol.
The winner is the lowest energy among those converged trials, with start index
as a deterministic tie breaker. This multistart search does not prove a global
minimum. Nonconverged trials and exceptions are retained in the diagnostics.
If a field/angle has no converged candidate, `winning_saddles.csv` retains a
`status=not_converged` row. The DSSF and transport keep this point as NaN, not
zero, and all plots leave a gap or white cell there. The other points continue.
Even a field with no converged angles is retained as an entirely missing row.
If nominal phi=0 has no converged state for a field, its normalized resistivity
curve stays undefined; the code does not substitute another reference angle.
Its converged absolute-resistivity values at other angles can still be shown.
If a converged saddle is unstable at DSSF momenta, the whole field/angle point
is also skipped. Its DSSF, resistivity and spin-plot cell stay undefined; the
saved winning MF state is retained for diagnosis. No negative spectrum is
silently clipped. Other conditions continue, including if the unstable point
occurs partway through transfer chunks or perpendicular-momentum samples.
Only this identified spectrum-instability exception is suppressed: missing
files, incompatible grids and unexpected programming errors still stop the job.

The run folder's `warnings.txt` records every skipped point, with the stage,
bare field, nominal/evaluated angles, and error message. Spectrum warnings
include the minimum `(lambda+v)/Jzz`. The file is refreshed on each DSSF run.
`data/tto_gmft_conditions.csv` records `mf_status`, final `status` (including
`unstable_dssf`) and `message`. HDF5 files and transport tables distinguish
`mf_converged` from `dssf_valid`; their compatibility `converged` mask means
the point is usable for transport. A missing or unstable zero reference leaves
the normalized curve undefined while valid absolute values remain available.

The general path rejects a DSSF grid different from the saved saddle's grid.
Historical scalar CSV input remains supported without `--general-saddles`.
General saddle tables cannot silently fall back to that scalar path.

## Results and restarting

The derived `log_root/run_label` directory contains all numerical data, diagnostics,
submission records, logs, and non-resistivity outputs:

```text
manifest.json                 exact resolved configuration and source hashes
submitted_config_*.json       immutable configuration snapshots from submission
gmft_transport_JOBID.out/.err Slurm standard output and error
source/                       archived numerical and plotting source
all_trials.csv                convergence/error diagnostics for all starts
trials/field_*/phi_*/          atomic per-trial JSON and converged saddle NPZ
winning_saddles.csv            selected start, seed, energy and saddle path
mean_fields/phi_*/             stage-2 input tables
data/tto_dynamic_structure_factor.hdf5
data/tto_dynamic_scattering.hdf5
data/tto_transport_summary.csv
data/tto_relaxation_times.csv
gmft_spin_components_vs_phi.pdf
warnings.txt                  skipped MF/DSSF points and their warning messages
```

The matching plot root contains one output named after the run. Its format is
selected by `resistivity_plot_mode`:

```text
RUN_LABEL.pdf                  overlay mode: all fields in one vector PDF
RUN_LABEL.gif                  movie mode: one field per GIF frame
```

Transport contains exactly the configured nonnegative nominal-angle scan; it
does not copy results onto negative angles. The nominal zero reference is
evaluated at +0.05 degrees.
If the configured nominal scan does not include zero, absolute resistivity is
still generated, but normalized resistivity remains undefined
because there is no `rho(0)` reference.

Resubmit with the same configuration and unchanged source to restart. Completed
trials, including failures, are reused. Completed transport stages are skipped;
an interrupted DSSF/scattering stage is recomputed from its beginning. A file
lock prevents two jobs from writing the same run concurrently. A manifest
mismatch refuses to combine different configurations or code. To change solver
settings after failures, select a new run directory. Adding the angular offset
changes the actual physical directions, so old unshifted trials must not be
reused as shifted results. Trial NPZ files are
written before their completion JSON, so interrupted writes are not mistaken
for completed trials.

For a scan without transport, or a local run with a selected worker count:

```bash
python gmft_transport_job.py --config gmft_transport_job.json --workers 16 --scan-only
```

After downloading a run, regenerate its configured resistivity output using:

```bash
python plot_gmft_transport.py /path/to/downloaded/run \
    --output-directory /path/to/gMFT_transport_plots
```

Without `--output-directory`, the plotter derives the destination from the
saved `plot_root` and `run_label`. Old manifests using explicit output
directories remain supported.

To recover an existing interrupted run from its saved winning states using
the current postprocessing code, without repeating any gMFT trials:

```bash
python postprocess_gmft_transport.py /path/to/run
```

This recomputes DSSF, scattering, transport and both plots. It keeps the original
manifest and source archive, records the current code in `postprocessing_source/`
and `postprocessing_manifest.json`, and saves the prior DSSF file as
`data/tto_dynamic_structure_factor.before_reprocess.hdf5`. It uses a file lock
and should not run concurrently with the numerical job for the same folder.

The plotting source is also archived under `source/EIO_TTO/`. Overlay mode is
vector PDF; movie mode is an animated GIF. No PNG or SVG frame files are produced.

### Spin components versus phi

New jobs run `plot_gmft_spin_components.py` and write the single-page vector
`gmft_spin_components_vs_phi.pdf`. Phi follows the configured nominal scan on
the x-axis.
The 2x4 heatmap layout shows signed `s_z` above `|s_perp|=|s_minus|`, with
sublattices 0, 1, 2, 3 in the columns. The y-axis is the bare reduced field;
colors show spin values on a fixed [-0.5, 0.5] scale. Cell boundaries are
midpoints between the sampled coordinates, without interpolation.
The title records `J_pm/Jzz` and the other couplings. It reads only saved
winning states and does not rerun MF or transport calculations.

For a downloaded run, use:

```bash
python plot_gmft_spin_components.py /path/to/downloaded/run
```

The standalone command does not change the run manifest. Upload both the new
plotter and the updated driver for future cluster jobs; retain
`plot_gmft_spin_movie.py`, which provides the shared data loader.

### Optional spin-component movie

The older `plot_gmft_spin_movie.py` remains available, writing `gmft_spin_components.gif`
and a vector `gmft_spin_components.pdf` with one frame per configured nominal
angle, using only the saved winning states. The 2x4
layout shows signed `s_z` in the top row and `|s_perp|=|s_minus|` in the bottom
row for sublattices 0, 1, 2, 3. The axes are `J_pm/Jzz` and the bare reduced
field `mu_B |B|/Jzz`; the color scale is fixed at [-0.5, 0.5]. A single run
has one J_pm column. No interpolation or new MF calculations are performed.
GIF frames are encoded directly from memory; PNG/SVG files are never created.

For an existing downloaded run, generate the movie directly without resuming
the numerical job or editing its manifest:

```bash
python plot_gmft_spin_movie.py /path/to/downloaded/run --fps 3
```

To combine several J_pm columns, pass multiple run directories and optionally
`--output-directory /path/to/output`. All other couplings, field values, grids,
and projection settings must agree. Runs 03 (integrated, J_pm=0.2) and 04
(slice, J_pm=0.15) therefore have separate movies. Updating the job driver adds
the new plotter to future source manifests; use the standalone command above
for existing runs so their original manifests remain intact.

## In-plane convergence failure with vanishing effective fields

The earlier general solver could exhaust `niter` on a link whose effective
field vanished: dividing the field by the numerical floor produced a
zero-length target, incompatible with the fixed spin length 1/2. The corrected
`gmft_for_qsi/gmft_field.py` and `gmft_for_qsi/gmft_field_general.py` keep the
current spin direction when the effective-field magnitude is at most `1e-12`
in solver energy units. Nonzero fields above that numerical threshold still
use the usual normalized target. The convergence tolerance is unchanged.
The general field projection also removes longitudinal dot-product roundoff
below `8*machine_epsilon*|h_ext|`, preventing a symmetry-forbidden field from
being amplified when normalizing a nearly free spin. Explicit `b_z` overrides
are preserved.

If this affected a cluster run, copy **both** corrected solver files into the
cluster's `gMFT_for_QSI/gmft_for_qsi/` directory. Set `run_label` in the JSON
to a new value, such as `"overnight_02"`, and submit again with
`bash queue_gmft_transport.sh`. The old run is kept for diagnostics: its failed
trial records would otherwise be reused, and its source manifest intentionally
rejects mixing results from the old and corrected solvers.
