#!/bin/bash

# case_submit.sh - script to submit SHiELD case to SLURM with specified parameters

set -e




export RUN_DIR="$(pwd)"
export CASE_DIR="$(basename "$RUN_DIR")"
export CASE_PARENT_DIR="$(basename "$(dirname "$RUN_DIR")")"
export UFS_UTILS="$(cd "$(dirname "$0")" && pwd)"



parse_result=$("$UFS_UTILS/tools/parse_config.py" 2>&1)
if [[ $? -ne 0 ]]; then printf '%s\n' "$parse_result"; exit 1; fi
source "$parse_result" && rm -f "$parse_result"


if (( "$SBATCH_NNODES" > 1 )); then
    SBATCH_MULTI_NODE=1
    SBATCH_MEMORY_FLAG="--mem-per-cpu=${SBATCH_MEM_PER_CPU}g"
else
    SBATCH_MEMORY_FLAG="--mem=${SBATCH_MEM}g"
fi

X_CASE_NAME="${CASE_NAME}"
SBATCH_NODE_EXCLUSIVE_FLAG=""


# MAP SBATCH_NTASKS_PER_NODE -> SBATCH_NODE_CONSTRAINT 
if (( SBATCH_NODE_CONSTRAINT == 1 )); then

    if (( SBATCH_NTASKS_PER_NODE <= 24 )); then
        CONSTRAINT="24core"

    elif (( SBATCH_NTASKS_PER_NODE <= 32 )); then
        CONSTRAINT="32core"

    elif (( SBATCH_NTASKS_PER_NODE <= 48 )); then
        CONSTRAINT="48core"

    elif (( SBATCH_NTASKS_PER_NODE <= 64 )); then
        CONSTRAINT="64core"

    else
        CONSTRAINT="192core"
    fi

    if (( SBATCH_NTASKS_PER_NODE <= 64 )); then
        SBATCH_EXCLUSIVE_NODE=1
    fi

    SBATCH_NODE_CONSTRAINT_FLAG="--constraint=${CONSTRAINT}"

else
    SBATCH_NODE_CONSTRAINT_FLAG="--constraint="
fi


if (( "$SBATCH_EXCLUSIVE_NODE" == 1 )); then
     SBATCH_NODE_EXCLUSIVE_FLAG="--exclusive"
fi

SLURM_OPEN_MODE="truncate"

for ((i=0; i<N_ENSEMBLES; i++)); do

    ENSEMBLE_ID=$((i + 1))

    # Set naming logic based on whether it's an ensemble or single run
    if (( "$N_ENSEMBLES" == 1 )); then
        ENSEMBLE_ID=0  # Not used for single runs, but set to 0 for consistency
        SLURM_JOB_NAME="${CASE_PARENT_DIR}.${CASE_DIR}"
        CASE_NAME="$X_CASE_NAME"
        DATA_SYMLINK="$RUN_DIR/run"
        LOG_FILE="$SBATCH_OUTPUT"
     
    else
        rm -f "$RUN_DIR/run"
        MEM_ID=$(printf "%02d" "$ENSEMBLE_ID")
        SLURM_JOB_NAME="${CASE_PARENT_DIR}.${CASE_DIR}.ENS${MEM_ID}"
        CASE_NAME="${X_CASE_NAME}/ENS${MEM_ID}"
        DATA_SYMLINK="$RUN_DIR/run${MEM_ID}"
        LOG_FILE="${SBATCH_OUTPUT%.log}_${MEM_ID}.log"
 
    fi

    source "$UFS_UTILS/tools/sbatch.sh"


done


if (( EXIT_CODE == 0 )); then
    echo "SUCCESS: Case Submitted"
else
    echo "ERROR: Job submission failed"
    exit 1
fi



