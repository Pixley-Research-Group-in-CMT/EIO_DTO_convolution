#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_CONFIG="${JOB_CONFIG:-$PROJECT_ROOT/swt_dssf_job.conf}"
SNAPSHOT_DIR="$PROJECT_ROOT/slurm_config_snapshots"

if [[ ! -f "$SOURCE_CONFIG" ]]; then
    echo "Job configuration does not exist: $SOURCE_CONFIG" >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available in this environment." >&2
    exit 1
fi

mkdir -p "$SNAPSHOT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_SNAPSHOT=$(mktemp "$SNAPSHOT_DIR/swt_dssf_${TIMESTAMP}_XXXXXX.conf")
cp "$SOURCE_CONFIG" "$CONFIG_SNAPSHOT"
cd "$PROJECT_ROOT"

if ! sbatch --export=ALL,JOB_CONFIG="$CONFIG_SNAPSHOT" \
    "$PROJECT_ROOT/submit_swt_dssf.slurm"; then
    rm -f "$CONFIG_SNAPSHOT"
    exit 1
fi

echo "Configuration snapshot: $CONFIG_SNAPSHOT"
