#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --job-name=swt_kernel
#SBATCH --output=swt_kernel_%j.out
#SBATCH --error=swt_kernel_%j.err
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
JOB_CONFIG="${JOB_CONFIG:-$PROJECT_ROOT/swt_kernel_job.conf}"
if [[ ! -f "$JOB_CONFIG" ]]; then
    echo "Job configuration does not exist: $JOB_CONFIG" >&2
    exit 1
fi
source "$JOB_CONFIG"
PLOT_MODE="${PLOT_MODE:-movie}"
plot_options=(--plot-mode "$PLOT_MODE")
if [[ "$PLOT_MODE" == "side_by_side" ]]; then
    plot_options+=(--fields "${FIELDS[@]}")
elif [[ "$PLOT_MODE" != "movie" ]]; then
    echo "PLOT_MODE must be side_by_side or movie." >&2
    exit 2
fi
if ! [[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid RUN_LABEL: $RUN_LABEL" >&2
    exit 2
fi
PYTHON_MODULE="${PYTHON_MODULE:-python/3.8.2}"
PYTHON_ENV_ACTIVATE="${PYTHON_ENV_ACTIVATE:-$HOME/eio_tto_env/bin/activate}"
if command -v module >/dev/null 2>&1 && [[ -n "$PYTHON_MODULE" ]]; then
    module load "$PYTHON_MODULE"
fi
if [[ -f "$PYTHON_ENV_ACTIVATE" ]]; then
    source "$PYTHON_ENV_ACTIVATE"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/swt_kernel_mpl_${SLURM_JOB_ID:-manual}"
export PYTHONUNBUFFERED=1
RESULT_ROOT="$HOME/SWT/kernel"
RUN_DIR="$RESULT_ROOT/$RUN_LABEL"
mkdir -p "$RESULT_ROOT" "$MPLCONFIGDIR"
if ! mkdir "$RUN_DIR"; then
    echo "Result directory already exists: $RUN_DIR; choose a different RUN_LABEL." >&2
    exit 2
fi
cp "$JOB_CONFIG" "$RUN_DIR/submitted_swt_kernel_job.conf"
echo "SWT transport-kernel output: $RUN_DIR"
srun -n 1 -c "${SLURM_CPUS_PER_TASK:-1}" "$PYTHON_BIN" -u \
    "$PROJECT_ROOT/plot_swt_kernel.py" \
    "${plot_options[@]}" --output-root "$RUN_DIR" --eio-model "$EIO_MODEL" \
    --mu-b-b-start "$MU_B_B_START" --mu-b-b-stop "$MU_B_B_STOP" --n-field-frames "$N_FIELD_FRAMES" \
    --phi-degrees "${FIELD_PHIS_DEGREES[@]}" --j-pm "$J_PM_RATIO" --j-pmpm "$J_PMPM_RATIO" \
    --g-z "$G_Z" --g-perp "$G_PERP" \
    --delta1-per-mev "$DELTA1_PER_MEV" --delta2-per-mev "$DELTA2_PER_MEV" \
    --j-zz-ev "$J_ZZ_EV" \
    --eta-mev "$ETA_MEV" \
    --dssf-nk "$DSSF_NK" --q-perp-mode "$Q_PERP_MODE" --n-q-perp "$N_Q_PERP" \
    --lattice-constant "$LATTICE_CONSTANT" \
    --spin-length "$SPIN_LENGTH" --regularization "$REGULARIZATION" \
    --stability-tilt "$STABILITY_TILT" --random-starts "$RANDOM_STARTS" \
    --vmax-percentile "$VMAX_PERCENTILE" \
    --marker-size "$MARKER_SIZE" --fps "$FPS" --video-format "$VIDEO_FORMAT" --dpi "$DPI"
