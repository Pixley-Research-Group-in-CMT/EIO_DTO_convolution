#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --job-name=swt_path_movie
#SBATCH --output=swt_path_movie_%j.out
#SBATCH --error=swt_path_movie_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
JOB_CONFIG="${JOB_CONFIG:-$PROJECT_ROOT/swt_path_movie_job.conf}"
if [[ ! -f "$JOB_CONFIG" ]]; then
    echo "Job configuration does not exist: $JOB_CONFIG" >&2
    exit 1
fi
source "$JOB_CONFIG"

RUN_LABEL="${RUN_LABEL:-path_movie}"
MU_B_B_START="${MU_B_B_START:-0.25}"
MU_B_B_STOP="${MU_B_B_STOP:-1.5}"
N_FIELD_FRAMES="${N_FIELD_FRAMES:-31}"
if ! declare -p FIELD_PHIS_DEGREES >/dev/null 2>&1; then
    FIELD_PHIS_DEGREES=(0.1 30.1)
fi
J_PM_RATIO="${J_PM_RATIO:-0.205}"
J_PMPM_RATIO="${J_PMPM_RATIO:-0.0}"
G_Z="${G_Z:-10.0}"
G_PERP="${G_PERP:-0.0}"
DELTA1_PER_MEV="${DELTA1_PER_MEV:-0.7142857142857143}"
DELTA2_PER_MEV="${DELTA2_PER_MEV:-0.7142857142857143}"
J_ZZ_EV="${J_ZZ_EV:-0.00019}"
COMPONENT="${COMPONENT:-one_magnon}"
VMAX_PERCENTILE="${VMAX_PERCENTILE:-99}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-40}"
N_OMEGA="${N_OMEGA:-200}"
ETA_MEV="${ETA_MEV:-0.02}"
DSSF_NK="${DSSF_NK:-8}"
OMEGA_CHUNK_SIZE="${OMEGA_CHUNK_SIZE:-8}"
FPS="${FPS:-10}"
VIDEO_FORMAT="${VIDEO_FORMAT:-gif}"
DPI="${DPI:-100}"
RESULT_ROOT="$HOME/SWT_DSSF_path_movie"

if ! [[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid RUN_LABEL: $RUN_LABEL" >&2
    exit 2
fi
if [[ "${#FIELD_PHIS_DEGREES[@]}" -ne 2 ]]; then
    echo "FIELD_PHIS_DEGREES must contain exactly two angles." >&2
    exit 2
fi
if ! [[ "$N_FIELD_FRAMES" =~ ^[0-9]+$ ]] || (( N_FIELD_FRAMES < 2 )); then
    echo "N_FIELD_FRAMES must be an integer of at least 2." >&2
    exit 2
fi
case "$COMPONENT" in
    total|one_magnon|one_magnon_[1-4]|two_magnon) ;;
    *)
        echo "Invalid COMPONENT: $COMPONENT" >&2
        exit 2
        ;;
esac
if [[ "$VIDEO_FORMAT" != "gif" && "$VIDEO_FORMAT" != "mp4" ]]; then
    echo "VIDEO_FORMAT must be gif or mp4: $VIDEO_FORMAT" >&2
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
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/swt_path_movie_mpl_${SLURM_JOB_ID:-manual}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$RESULT_ROOT"

RUN_DIR="$RESULT_ROOT/$RUN_LABEL"
if ! mkdir "$RUN_DIR"; then
    echo "Result directory already exists: $RUN_DIR" >&2
    echo "Choose a different RUN_LABEL or move the existing directory." >&2
    exit 2
fi
cp "$JOB_CONFIG" "$RUN_DIR/submitted_swt_path_movie_job.conf"

{
    echo "job_id=${SLURM_JOB_ID:-manual_$$}"
    echo "run_label=$RUN_LABEL"
    echo "job_config=$JOB_CONFIG"
    echo "output=$RUN_DIR"
    echo "mu_B_B_start_over_Jzz=$MU_B_B_START"
    echo "mu_B_B_stop_over_Jzz=$MU_B_B_STOP"
    echo "n_field_frames=$N_FIELD_FRAMES"
    echo "field_theta_degrees=90"
    echo "field_phi_degrees=${FIELD_PHIS_DEGREES[*]}"
    echo "J_pm_over_Jzz=$J_PM_RATIO"
    echo "J_pmpm_over_Jzz=$J_PMPM_RATIO"
    echo "g_z=$G_Z"
    echo "g_perp=$G_PERP"
    echo "delta1_per_mev=$DELTA1_PER_MEV"
    echo "delta2_per_mev=$DELTA2_PER_MEV"
    echo "J_zz_ev=$J_ZZ_EV"
    echo "component=$COMPONENT"
    echo "vmax_percentile=$VMAX_PERCENTILE"
    echo "points_per_segment=$POINTS_PER_SEGMENT"
    echo "n_omega=$N_OMEGA"
    echo "eta_mev=$ETA_MEV"
    echo "dssf_nk=$DSSF_NK"
    echo "omega_chunk_size=$OMEGA_CHUNK_SIZE"
    echo "fps=$FPS"
    echo "video_format=$VIDEO_FORMAT"
    echo "dpi=$DPI"
    echo "tto_cubic_lattice_angstrom=10.154"
    echo "python=$($PYTHON_BIN --version 2>&1)"
} > "$RUN_DIR/run_parameters.txt"

echo "SWT high-symmetry-path field movie output: $RUN_DIR"
srun -n 1 -c "${SLURM_CPUS_PER_TASK:-1}" "$PYTHON_BIN" -u \
    "$PROJECT_ROOT/plot_swt_high_symmetry_dssf_movie.py" \
    --output-root "$RUN_DIR" \
    --mu-b-b-start "$MU_B_B_START" \
    --mu-b-b-stop "$MU_B_B_STOP" \
    --n-field-frames "$N_FIELD_FRAMES" \
    --field-phis "${FIELD_PHIS_DEGREES[@]}" \
    --j-pm "$J_PM_RATIO" \
    --j-pmpm "$J_PMPM_RATIO" \
    --g-z "$G_Z" \
    --g-perp "$G_PERP" \
    --delta1-per-mev "$DELTA1_PER_MEV" \
    --delta2-per-mev "$DELTA2_PER_MEV" \
    --j-zz-ev "$J_ZZ_EV" \
    --component "$COMPONENT" \
    --vmax-percentile "$VMAX_PERCENTILE" \
    --points-per-segment "$POINTS_PER_SEGMENT" \
    --n-omega "$N_OMEGA" \
    --eta-mev "$ETA_MEV" \
    --dssf-nk "$DSSF_NK" \
    --omega-chunk-size "$OMEGA_CHUNK_SIZE" \
    --fps "$FPS" \
    --video-format "$VIDEO_FORMAT" \
    --dpi "$DPI"

echo "Data: $RUN_DIR/data/swt_dssf_high_symmetry_field_movie.hdf5"
echo "Videos: $RUN_DIR/videos"
