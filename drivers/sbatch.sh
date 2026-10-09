# shellcheck shell=bash

# sbatch.sh - Script to submit a job to SLURM with the appropriate environment variables

# Every CASE_* and path variable is in the environment: set by case_submit.py
# for the first segment and inherited, with updated values, by case_run.sh for
# later ones. Exporting ALL forwards them without listing values in --export,
# which splits on commas.
EXPORT_VARS="ALL"


EXIT_CODE=0
# "|| EXIT_CODE=$?" keeps a failed sbatch from aborting a caller that runs
# under set -e (case_run.sh) before the error below is reported.
JOB_ID=$(sbatch --parsable \
$CASE_NODE_EXCLUSIVE_FLAG \
$CASE_NODE_CONSTRAINT_FLAG \
$CASE_MEMORY_FLAG \
$CASE_SBATCH_OPTIONS \
--time=$CASE_TIME_LIMIT \
--job-name=$SLURM_JOB_NAME \
--nodes=$CASE_NNODES \
--ntasks=$CASE_NTASKS \
--cpus-per-task=$CASE_CPUS_PER_TASK \
--ntasks-per-node=$CASE_NTASKS_PER_NODE \
--partition=$CASE_PARTITION \
--output=/dev/null \
--open-mode=$SLURM_OPEN_MODE \
--export="$EXPORT_VARS" \
"$UFS_UTILS_DIR/drivers/case_run.sh") || EXIT_CODE=$?
export EXIT_CODE
export JOB_ID



if (( EXIT_CODE != 0 )) || [[ -z "$JOB_ID" ]]; then
    echo "Case.Submit - ERROR - sbatch failed for job $SLURM_JOB_NAME" >&2
    (( EXIT_CODE == 0 )) && EXIT_CODE=1
    # works whether this file is sourced (case_run.sh) or executed (case_submit.py)
    if (return 0 2>/dev/null); then
        return "$EXIT_CODE"
    fi
    exit "$EXIT_CODE"
fi

