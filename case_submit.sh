#!/bin/bash -l

# Deactivate all active Conda environments for this script process.

conda deactivate >/dev/null 2>&1

UFS_UTILS_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"$UFS_UTILS_DIR/launchers/case_submit.py"
