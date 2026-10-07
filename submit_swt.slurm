#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=eio_tto_spin_wave
#SBATCH --output=eio_tto_spin_wave_%j.out
#SBATCH --error=eio_tto_spin_wave_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

# Keep the commonly changed values in a small, editable document.  Set
# JOB_CONFIG at submission time only when a different configuration file is
# desired.  Positional fields, when supplied, override FIELDS from this file.
JOB_CONFIG="${JOB_CONFIG:-$PROJECT_ROOT/spin_wave_job.conf}"
if [[ ! -f "$JOB_CONFIG" ]]; then
    echo "Job configuration does not exist: $JOB_CONFIG" >&2
    exit 1
fi
source "$JOB_CONFIG"

if (( $# > 0 )); then
    FIELDS=("$@")
fi
if ! declare -p FIELDS >/dev/null 2>&1; then
    echo "FIELDS is not defined. Set FIELDS=(...) in $JOB_CONFIG." >&2
    exit 2
fi
if (( ${#FIELDS[@]} == 0 )); then
    echo "No field values configured. Set FIELDS=(...) in $JOB_CONFIG." >&2
    exit 2
fi

# These defaults match the existing Python setup on this cluster.  A different
# environment can be selected at submission time with, for example,
#   sbatch --export=ALL,PYTHON_MODULE=python/3.11,PYTHON_ENV_ACTIVATE=/path/to/bin/activate ...
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
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/eio_tto_mpl_${SLURM_JOB_ID:-manual}"
export PYTHONUNBUFFERED=1

run_command() {
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
        srun -n 1 -c "${SLURM_CPUS_PER_TASK:-1}" "$@"
    else
        "$@"
    fi
}

RESULT_ROOT="${RESULT_ROOT:-$HOME/SWT_transport}"
J_PM_RATIO="${J_PM_RATIO:-0.2}"
J_PMPM_RATIO="${J_PMPM_RATIO:-0.0}"
J_ZZ_EV="${J_ZZ_EV:-0.00017}"
ETA_EV="${ETA_EV:-0.00002}"
DSSF_NK="${DSSF_NK:-8}"
Q_PERP_MODE="${Q_PERP_MODE:-slice}"
N_Q_PERP="${N_Q_PERP:-$DSSF_NK}"
STABILITY_TILT="${STABILITY_TILT:-0.01}"
FIELD_THETA_DEGREES="${FIELD_THETA_DEGREES:-90}"
PHI_OFFSET_DEGREES="${PHI_OFFSET_DEGREES:-0.1}"
PHI_START_DEGREES="${PHI_START_DEGREES:-0}"
PHI_STOP_DEGREES="${PHI_STOP_DEGREES:-180}"
N_PHI="${N_PHI:-91}"
REFLECT_NEGATIVE_PHI="${REFLECT_NEGATIVE_PHI:-true}"
RUN_LABEL="${RUN_LABEL:-spin_wave}"

case "$Q_PERP_MODE" in
    slice|integrated) ;;
    *)
        echo "Invalid Q_PERP_MODE: $Q_PERP_MODE" >&2
        echo "Use slice or integrated in $JOB_CONFIG." >&2
        exit 2
        ;;
esac
if ! [[ "$N_Q_PERP" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_Q_PERP must be a positive integer: $N_Q_PERP" >&2
    exit 2
fi
if [[ "$Q_PERP_MODE" == "integrated" && "$N_Q_PERP" -lt 2 ]]; then
    echo "Integrated Q_PERP_MODE requires N_Q_PERP >= 2." >&2
    exit 2
fi

case "$REFLECT_NEGATIVE_PHI" in
    true|TRUE|1|yes|YES)
        REFLECT_NEGATIVE_PHI=true
        REFLECTION_ARGS=(--reflect-negative-phi)
        ;;
    false|FALSE|0|no|NO)
        REFLECT_NEGATIVE_PHI=false
        REFLECTION_ARGS=(--no-reflect-negative-phi)
        ;;
    *)
        echo "Invalid REFLECT_NEGATIVE_PHI: $REFLECT_NEGATIVE_PHI" >&2
        echo "Use true or false in $JOB_CONFIG." >&2
        exit 2
        ;;
esac

if ! [[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid RUN_LABEL: $RUN_LABEL" >&2
    echo "Use only letters, numbers, periods, underscores, and hyphens." >&2
    exit 2
fi

for field in "${FIELDS[@]}"; do
    if ! [[ "$field" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        echo "Invalid h/J_zz value: $field" >&2
        exit 2
    fi
done

JOB_TAG="${SLURM_JOB_ID:-manual_$$}"
RUN_DIR="$RESULT_ROOT/$RUN_LABEL"
export EIO_TTO_OUTPUT_ROOT="$RUN_DIR"
mkdir -p "$RESULT_ROOT" "$MPLCONFIGDIR"
if ! mkdir "$RUN_DIR"; then
    echo "Result directory already exists: $RUN_DIR" >&2
    echo "Choose a different RUN_LABEL or move the existing directory." >&2
    exit 2
fi
mkdir "$RUN_DIR/data"
cp "$JOB_CONFIG" "$RUN_DIR/submitted_spin_wave_job.conf"

BASE_EIO="$PROJECT_ROOT/data/surface_states_summary_nk24.hdf5"
RUN_EIO="$RUN_DIR/data/surface_states_summary_nk24.hdf5"
if [[ -f "$BASE_EIO" ]]; then
    ln -s "$BASE_EIO" "$RUN_EIO"
else
    echo "The cached EIO surface states were not found; generating them now."
    run_command "$PYTHON_BIN" -u "$PROJECT_ROOT/01_generate_eio.py"
fi

{
    echo "job_id=$JOB_TAG"
    echo "run_label=$RUN_LABEL"
    echo "job_config=$JOB_CONFIG"
    echo "fields_h_over_Jzz=${FIELDS[*]}"
    echo "J_pm_over_Jzz=$J_PM_RATIO"
    echo "J_pmpm_over_Jzz=$J_PMPM_RATIO"
    echo "J_zz_ev=$J_ZZ_EV"
    echo "eta_ev=$ETA_EV"
    echo "dssf_nk=$DSSF_NK"
    echo "q_perp_mode=$Q_PERP_MODE"
    echo "n_q_perp=$N_Q_PERP"
    echo "stability_tilt=$STABILITY_TILT"
    echo "field_theta_degrees=$FIELD_THETA_DEGREES"
    echo "phi_offset_degrees=$PHI_OFFSET_DEGREES"
    echo "phi_start_degrees=$PHI_START_DEGREES"
    echo "phi_stop_degrees=$PHI_STOP_DEGREES"
    echo "n_phi=$N_PHI"
    echo "reflect_negative_phi=$REFLECT_NEGATIVE_PHI"
    echo "project_root=$PROJECT_ROOT"
    echo "python=$($PYTHON_BIN --version 2>&1)"
} > "$RUN_DIR/run_parameters.txt"

echo "Project: $PROJECT_ROOT"
echo "Results: $RUN_DIR"
echo "h/J_zz values: ${FIELDS[*]}"

run_command "$PYTHON_BIN" -u \
    "$PROJECT_ROOT/02_compute_tto_spin_wave_structure_factor.py" \
    --j-pm "$J_PM_RATIO" \
    --j-pmpm "$J_PMPM_RATIO" \
    --fields "${FIELDS[@]}" \
    --theta "$FIELD_THETA_DEGREES" \
    --phi-offset "$PHI_OFFSET_DEGREES" \
    --phi-start "$PHI_START_DEGREES" \
    --phi-stop "$PHI_STOP_DEGREES" \
    --n-phi "$N_PHI" \
    --j-zz-ev "$J_ZZ_EV" \
    --eta-ev "$ETA_EV" \
    --dssf-nk "$DSSF_NK" \
    --q-perp-mode "$Q_PERP_MODE" \
    --n-q-perp "$N_Q_PERP" \
    "${REFLECTION_ARGS[@]}" \
    --stability-tilt "$STABILITY_TILT"

run_command "$PYTHON_BIN" -u \
    "$PROJECT_ROOT/03_compute_scattering.py" --theory spin-wave

run_command "$PYTHON_BIN" -u \
    "$PROJECT_ROOT/04_compute_transport.py" --theory spin-wave

echo "Completed successfully."
echo "Data directory: $RUN_DIR/data"
echo "Transport table: $RUN_DIR/data/tto_spin_wave_transport_summary.csv"
echo "Relaxation diagnostics: $RUN_DIR/data/tto_spin_wave_conditions.csv"
echo "Download this run directory and generate its figure locally with plot_swt_results.py."
