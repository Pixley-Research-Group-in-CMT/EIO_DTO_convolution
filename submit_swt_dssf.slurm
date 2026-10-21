#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --job-name=swt_dssf
#SBATCH --output=swt_dssf_%j.out
#SBATCH --error=swt_dssf_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
JOB_CONFIG="${JOB_CONFIG:-$PROJECT_ROOT/swt_dssf_job.conf}"
if [[ ! -f "$JOB_CONFIG" ]]; then
    echo "Job configuration does not exist: $JOB_CONFIG" >&2
    exit 1
fi
source "$JOB_CONFIG"

RUN_LABEL="${RUN_LABEL:-dssf}"
J_PM_RATIO="${J_PM_RATIO:-0.205}"
J_PMPM_RATIO="${J_PMPM_RATIO:-0.0}"
G_Z="${G_Z:-10.0}"
G_PERP="${G_PERP:-0.0}"
DELTA1_PER_MEV="${DELTA1_PER_MEV:-0.7142857142857143}"
DELTA2_PER_MEV="${DELTA2_PER_MEV:-0.7142857142857143}"
J_ZZ_EV="${J_ZZ_EV:-0.00019}"
COMPONENT="${COMPONENT:-one_magnon}"
VMAX_PERCENTILE="${VMAX_PERCENTILE:-95}"
N_OMEGA_FRAMES="${N_OMEGA_FRAMES:-60}"
FPS="${FPS:-10}"
VIDEO_FORMAT="${VIDEO_FORMAT:-gif}"
RESULT_ROOT="$HOME/SWT_DSSF"

if ! [[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid RUN_LABEL: $RUN_LABEL" >&2
    exit 2
fi
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
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/swt_dssf_mpl_${SLURM_JOB_ID:-manual}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$RESULT_ROOT"

RUN_DIR="$RESULT_ROOT/$RUN_LABEL"
if ! mkdir "$RUN_DIR"; then
    echo "Result directory already exists: $RUN_DIR" >&2
    echo "Choose a different RUN_LABEL or move the existing directory." >&2
    exit 2
fi
cp "$JOB_CONFIG" "$RUN_DIR/submitted_swt_dssf_job.conf"

{
    echo "job_id=${SLURM_JOB_ID:-manual_$$}"
    echo "run_label=$RUN_LABEL"
    echo "job_config=$JOB_CONFIG"
    echo "output=$RUN_DIR"
    echo "mu_B_B_over_Jzz=0.25 0.45 1.5"
    echo "field_theta_degrees=90"
    echo "field_phi_degrees=0.1 30.1"
    echo "J_pm_over_Jzz=$J_PM_RATIO"
    echo "J_pmpm_over_Jzz=$J_PMPM_RATIO"
    echo "g_z=$G_Z"
    echo "g_perp=$G_PERP"
    echo "delta1_per_mev=$DELTA1_PER_MEV"
    echo "delta2_per_mev=$DELTA2_PER_MEV"
    echo "J_zz_ev=$J_ZZ_EV"
    echo "component=$COMPONENT"
    echo "vmax_percentile=$VMAX_PERCENTILE"
    echo "n_omega_frames=$N_OMEGA_FRAMES"
    echo "fps=$FPS"
    echo "video_format=$VIDEO_FORMAT"
    echo "eta_mev=0.02"
    echo "tto_cubic_lattice_angstrom=10.154"
    echo "n_q=101"
    echo "dssf_nk=8"
    echo "python=$($PYTHON_BIN --version 2>&1)"
} > "$RUN_DIR/run_parameters.txt"

echo "SWT DSSF animation output: $RUN_DIR"
srun -n 1 -c "${SLURM_CPUS_PER_TASK:-1}" "$PYTHON_BIN" -u \
    "$PROJECT_ROOT/plot_swt_dssf_video.py" \
    --output-root "$RUN_DIR" \
    --j-pm "$J_PM_RATIO" \
    --j-pmpm "$J_PMPM_RATIO" \
    --g-z "$G_Z" \
    --g-perp "$G_PERP" \
    --delta1-per-mev "$DELTA1_PER_MEV" \
    --delta2-per-mev "$DELTA2_PER_MEV" \
    --j-zz-ev "$J_ZZ_EV" \
    --component "$COMPONENT" \
    --vmax-percentile "$VMAX_PERCENTILE" \
    --n-omega-frames "$N_OMEGA_FRAMES" \
    --fps "$FPS" \
    --video-format "$VIDEO_FORMAT"

echo "Data: $RUN_DIR/data/swt_dssf.hdf5"
echo "Video: $RUN_DIR/videos/swt_dssf_phi_0.$VIDEO_FORMAT"
echo "Video: $RUN_DIR/videos/swt_dssf_phi_30.$VIDEO_FORMAT"
