#!/bin/bash
# Snapshot the editable SWT configuration at submission, including while queued.
set -euo pipefail
cd "$(dirname "$0")"
config="${1:-swt_transport_job.json}"
if [[ ! -r "$config" ]]; then
    echo "Configuration file missing or unreadable: $config" >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available in this environment." >&2
    exit 1
fi
run_directory=$(python3 -c \
    'import json, sys; from swt_transport_job import configured_output_directories; print(configured_output_directories(json.load(open(sys.argv[1])))[0])' \
    "$config")
mkdir -p "$run_directory"
snapshot=$(mktemp "$run_directory/submitted_config_XXXXXXXX.json")
cp "$config" "$snapshot"
config_payload=$(base64 < "$snapshot" | tr -d '\n')
sbatch \
    --output="$run_directory/swt_transport_%j.out" \
    --error="$run_directory/swt_transport_%j.err" \
    --export="ALL,JOB_CONFIG=$snapshot,SWT_CONFIG_BASE64=$config_payload" \
    submit_swt.slurm
