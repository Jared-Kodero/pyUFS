#!/usr/bin/env bash
# Update the fix directory: bash sync_noaa_fix.sh --dest /path/to/fix [--skip-carto]
# Needs the AWS CLI (no credentials) and a Python with cartopy for carto/.
set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module load awscli || true
fi
command -v aws >/dev/null || { echo "ERROR: aws CLI not found" >&2; exit 1; }

python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/update_fix.py" "$@"

