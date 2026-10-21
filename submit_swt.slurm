#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=swt_transport
#SBATCH --output=swt_transport_%j.out
#SBATCH --error=swt_transport_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
module load "${PYTHON_MODULE:-python/3.8.2}"
source "${PYTHON_ENV_ACTIVATE:-$HOME/eio_tto_env/bin/activate}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/swt_mpl_${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"
job_config="${JOB_CONFIG:-$PWD/swt_transport_job.json}"
if [[ -n "${SWT_CONFIG_BASE64:-}" ]]; then
    job_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/swt_config_${SLURM_JOB_ID:-manual}_XXXXXXXX.json")
    trap 'rm -f "$job_config"' EXIT
    printf '%s' "$SWT_CONFIG_BASE64" | base64 --decode > "$job_config"
fi
if [[ ! -r "$job_config" ]]; then
    echo "Configuration file missing or unreadable: $job_config. Resubmit using queue_swt.sh." >&2
    exit 1
fi
srun -n 1 -c "$SLURM_CPUS_PER_TASK" "${PYTHON_BIN:-python3}" -u \
    swt_transport_job.py --config "$job_config"
