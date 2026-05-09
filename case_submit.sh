#!/bin/bash
set -e

export CASE_PWD="$(pwd)"
export CASE_DIR="$(basename "$CASE_PWD")"
export CASE_PARENT_DIR="$(basename "$(dirname "$CASE_PWD")")"
export UFS_UTILS_DIR="$(cd "$(dirname "$0")" && pwd)"

parse_result=$("$UFS_UTILS_DIR/tools/parse_config.py" 2>&1)
if [[ "$parse_result" == *ERROR:* ]]; then
    printf '%s\n' "$parse_result"
    exit 1
fi

source "$parse_result" && rm -f "$parse_result"

CASE_NAME_X="${CASE_NAME}"
SLURM_OPEN_MODE="truncate"

for ((i=0; i<CASE_ENSEMBLES; i++)); do
    CASE_ENSEMBLE_ID=$((i + 1))

    if (( CASE_ENSEMBLES == 1 )); then
        CASE_ENSEMBLE_ID=0
        SLURM_JOB_NAME="${CASE_PARENT_DIR}.${CASE_DIR}"
        CASE_NAME="$CASE_NAME_X"
        CASE_DATA_SYMLINK="$CASE_PWD/run"
        CASE_LOG_FILE="$SBATCH_OUTPUT"
    else
        rm -f "$CASE_PWD/run"
        MEM_ID=$(printf "%02d" "$CASE_ENSEMBLE_ID")
        SLURM_JOB_NAME="${CASE_PARENT_DIR}.${CASE_DIR}.ENS${MEM_ID}"
        CASE_NAME="${CASE_NAME_X}/ENS${MEM_ID}"
        CASE_DATA_SYMLINK="$CASE_PWD/run${MEM_ID}"
        CASE_LOG_FILE="${SBATCH_OUTPUT%.log}_${MEM_ID}.log"
    fi

    source "$UFS_UTILS_DIR/tools/sbatch.sh"
done

if (( EXIT_CODE == 0 )); then
    echo "SUCCESS: Case Submitted"
else
    echo "ERROR: Job submission failed"
    exit 1
fi