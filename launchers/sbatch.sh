# shellcheck shell=bash

# sbatch.sh - submit launchers/case_run.sh to SLURM.
#
# Called by case_submit.py (first segment, executed) and by case_run.sh (later
# segments, sourced). It builds no values itself: every CASE_* variable below
# is already in the environment (documented in launchers/case_submit.py).
#
#   CASE_NODE_EXCLUSIVE_FLAG   --exclusive, or empty
#   CASE_NODE_CONSTRAINT_FLAG  --constraint=<value>, or empty
#   CASE_MEMORY_FLAG           --mem=<n>g (one node), --mem-per-cpu=<n>g (several), or empty
#   CASE_SBATCH_OPTIONS        raw extra sbatch arguments from run_config.yaml (sbatch_options)
#   CASE_TIME_LIMIT            H:00:00 from walltime
#   CASE_NNODES / CASE_NTASKS / CASE_NTASKS_PER_NODE / CASE_CPUS_PER_TASK / CASE_PARTITION
#   SLURM_JOB_NAME             <parent>.<case>[.MEMnn]
#   SLURM_OPEN_MODE            truncate (first segment) or append (resubmission)
#
# The flag variables are deliberately unquoted so that an empty value adds no
# argument and CASE_SBATCH_OPTIONS can hold several words.

# Every CASE_* and path variable is in the environment: set by case_submit.py
# for the first segment and inherited, with updated values, by case_run.sh for
# later ones. Exporting ALL forwards them without listing values in --export,
# which splits on commas.
EXPORT_VARS="ALL"


EXIT_CODE=0
# "|| EXIT_CODE=$?" keeps a failed sbatch from aborting a caller that runs
# under set -e (case_run.sh) before the error below is reported.
# --parsable prints only the job id. --output=/dev/null: case_run.sh writes its
# own log (CASE_LOG_FILE), so SLURM does not create a second one.
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
"$UFS_UTILS_DIR/launchers/case_run.sh") || EXIT_CODE=$?
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

