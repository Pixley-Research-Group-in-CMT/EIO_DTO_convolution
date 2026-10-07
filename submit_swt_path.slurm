#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --job-name=swt_path
#SBATCH --output=swt_path_%j.out
#SBATCH --error=swt_path_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
JOB_CONFIG="${JOB_CONFIG:-$PROJECT_ROOT/swt_path_job.conf}"
if [[ ! -f "$JOB_CONFIG" ]]; then
    echo "Job configuration does not exist: $JOB_CONFIG" >&2
    exit 1
fi
source "$JOB_CONFIG"

RUN_LABEL="${RUN_LABEL:-path}"
if ! declare -p FIELDS >/dev/null 2>&1; then
    FIELDS=(0.25 0.45 1.5)
fi
if ! declare -p FIELD_PHIS_DEGREES >/dev/null 2>&1; then
    FIELD_PHIS_DEGREES=(0.1 30.1)
fi
J_PM_RATIO="${J_PM_RATIO:-0.205}"
J_PMPM_RATIO="${J_PMPM_RATIO:-0.0}"
J_ZZ_EV="${J_ZZ_EV:-0.00019}"
COMPONENT="${COMPONENT:-one_magnon}"
VMAX_PERCENTILE="${VMAX_PERCENTILE:-99}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-40}"
N_OMEGA="${N_OMEGA:-200}"
ETA_MEV="${ETA_MEV:-0.02}"
DSSF_NK="${DSSF_NK:-8}"
OMEGA_CHUNK_SIZE="${OMEGA_CHUNK_SIZE:-8}"
RESULT_ROOT="$HOME/SWT_DSSF_path"

if ! [[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid RUN_LABEL: $RUN_LABEL" >&2
    exit 2
fi
if [[ "${#FIELDS[@]}" -ne 3 ]]; then
    echo "FIELDS must contain exactly three h/J_zz values." >&2
    exit 2
fi
if [[ "${#FIELD_PHIS_DEGREES[@]}" -ne 2 ]]; then
    echo "FIELD_PHIS_DEGREES must contain exactly two angles." >&2
    exit 2
fi
case "$COMPONENT" in
    total|bragg|one_magnon|two_magnon) ;;
    *)
        echo "Invalid COMPONENT: $COMPONENT" >&2
        exit 2
        ;;
esac

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
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/swt_path_mpl_${SLURM_JOB_ID:-manual}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$RESULT_ROOT"

RUN_DIR="$RESULT_ROOT/$RUN_LABEL"
if ! mkdir "$RUN_DIR"; then
    echo "Result directory already exists: $RUN_DIR" >&2
    echo "Choose a different RUN_LABEL or move the existing directory." >&2
    exit 2
fi
cp "$JOB_CONFIG" "$RUN_DIR/submitted_swt_path_job.conf"

{
    echo "job_id=${SLURM_JOB_ID:-manual_$$}"
    echo "run_label=$RUN_LABEL"
    echo "job_config=$JOB_CONFIG"
    echo "output=$RUN_DIR"
    echo "h_over_Jzz=${FIELDS[*]}"
    echo "field_theta_degrees=90"
    echo "field_phi_degrees=${FIELD_PHIS_DEGREES[*]}"
    echo "J_pm_over_Jzz=$J_PM_RATIO"
    echo "J_pmpm_over_Jzz=$J_PMPM_RATIO"
    echo "J_zz_ev=$J_ZZ_EV"
    echo "component=$COMPONENT"
    echo "vmax_percentile=$VMAX_PERCENTILE"
    echo "points_per_segment=$POINTS_PER_SEGMENT"
    echo "n_omega=$N_OMEGA"
    echo "eta_mev=$ETA_MEV"
    echo "dssf_nk=$DSSF_NK"
    echo "omega_chunk_size=$OMEGA_CHUNK_SIZE"
    echo "tto_cubic_lattice_angstrom=10.154"
    echo "python=$($PYTHON_BIN --version 2>&1)"
} > "$RUN_DIR/run_parameters.txt"

echo "SWT high-symmetry-path DSSF output: $RUN_DIR"
srun -n 1 -c "${SLURM_CPUS_PER_TASK:-1}" "$PYTHON_BIN" -u \
    "$PROJECT_ROOT/plot_swt_high_symmetry_dssf.py" \
    --output-root "$RUN_DIR" \
    --fields "${FIELDS[@]}" \
    --field-phis "${FIELD_PHIS_DEGREES[@]}" \
    --j-pm "$J_PM_RATIO" \
    --j-pmpm "$J_PMPM_RATIO" \
    --j-zz-ev "$J_ZZ_EV" \
    --component "$COMPONENT" \
    --vmax-percentile "$VMAX_PERCENTILE" \
    --points-per-segment "$POINTS_PER_SEGMENT" \
    --n-omega "$N_OMEGA" \
    --eta-mev "$ETA_MEV" \
    --dssf-nk "$DSSF_NK" \
    --omega-chunk-size "$OMEGA_CHUNK_SIZE"

echo "Data: $RUN_DIR/data/swt_dssf_high_symmetry.hdf5"
echo "Figure: $RUN_DIR/figures/swt_dssf_high_symmetry_sublattices_phi_0p1.pdf"
echo "Figure: $RUN_DIR/figures/swt_dssf_high_symmetry_mu_sum_phi_0p1.pdf"
echo "Figure: $RUN_DIR/figures/swt_dssf_high_symmetry_sublattices_phi_30p1.pdf"
echo "Figure: $RUN_DIR/figures/swt_dssf_high_symmetry_mu_sum_phi_30p1.pdf"
