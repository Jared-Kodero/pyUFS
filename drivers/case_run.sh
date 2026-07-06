#!/bin/bash -l

# case_run.sh - main driver script for running SHiELD case on SLURM

# -------------------------------------------------------------------------#
# --- DO NOT MODIFY BELOW THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING! --- #
# -------------------------------------------------------------------------#

set -e

module purge



export WORK_DIR="$JOBTMP_DIR/$CASE_PARENT_DIR/$CASE_NAME"
export CASE_DIR="$CASE_ROOT_DIR/$CASE_PARENT_DIR/$CASE_NAME"
export ARCHIVE_DIR="$ARCHIVE_ROOT_DIR/$CASE_PARENT_DIR/$CASE_NAME"
export TMP_DIR="$JOBTMP_DIR/tmp"


if [ -z "$CASE_RUN_START_TIME" ]; then
    export CASE_RUN_START_TIME=$(date +%s)
fi

SESSION_START_TIME=$(date +%s)


cd "$CASE_PWD"


if  [ ! -d "$JOBTMP_DIR" ]; then 
    WORK_DIR="$CASE_DIR"
    SYNC=0
else
    SYNC=1
    rm -rf "$WORK_DIR"
fi


# PREPARE DIRECTORIES
mkdir -p "$WORK_DIR"
mkdir -p "$CASE_DIR"
mkdir -p "$TMP_DIR"


if [ ! -d "$CONTAINERS_DIR" ] || [ -z "$(ls -A "$CONTAINERS_DIR")" ]; then
    source "$UFS_UTILS_DIR/configs/install_images.sh" > $WORK_DIR/image_build.log 2>&1
fi


# CREATE SYMLINK TO WORK_DIR
rm -f "$CASE_DATA_SYMLINK"
ln -s "$WORK_DIR" "$CASE_DATA_SYMLINK"

# SYNC CASE_DIR TO WORK_DIR
if (( SYNC == 1 )); then
    rsync -a --delete "$CASE_DIR/" "$WORK_DIR/"
fi


# RUNTIME FILES
EXIT_CODE_FILE="$WORK_DIR/exit_code"
SHIELD_NATIVE="$WORK_DIR/shield.native"

touch "$EXIT_CODE_FILE"

# CHECK FOR PREVIOUS RUN
if (( CASE_RESUBMIT_INDEX  > 0 )); then
    exec >>"$CASE_LOG_FILE" 2>&1
else
     exec >"$CASE_LOG_FILE" 2>&1
fi


export TMPDIR="$TMP_DIR"
export APPTAINER_CACHEDIR=$TMP_DIR
export APPTAINER_HOME=$HOME
export APPTAINER_BINDPATH=$(printf "%s" "$CONTAINER_BINDPATH" | base64 -d)


FREGRID="apptainer exec $FREGRID_SIF $UFS_UTILS_DIR/fregrid"
PREPROCESS="apptainer exec $PREPROCESS_SIF $UFS_UTILS_DIR/preprocess"
SHIELD_PREFIX="apptainer exec $SHIELD_SIF"

ON_SUCCESS="rsync -a --delete "$WORK_DIR/" "$CASE_DIR/""
ON_FAILURE="rsync -a "$WORK_DIR/" "$CASE_DIR/""

$PREPROCESS # Run preprocess to stage grid and IC files (if needed)


if (( $(<"$EXIT_CODE_FILE") == 0 && CASE_PREPROCESS_ONLY == 1 )); then
    $ON_SUCCESS
    rm -f "$CASE_DATA_SYMLINK"
    ln -s "$CASE_DIR" "$CASE_DATA_SYMLINK"
    echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - IC and Grid generation complete."
    exit 0
fi

if (( CASE_MULTI_NODE_FLAG == 1 )) || [[ -f "$SHIELD_NATIVE" ]]; then
    SHIELD="$WORK_DIR/shield"
else
    SHIELD="$SHIELD_PREFIX $WORK_DIR/shield"
fi

RUN_START_TIME=$(date +%s)

(( $(<"$EXIT_CODE_FILE") == 0 )) && $SHIELD
(( $(<"$EXIT_CODE_FILE") == 0 )) && $FREGRID


RUN_END_TIME=$(date +%s)


EXIT_CODE=$(<"$EXIT_CODE_FILE")

if (( SYNC == 1 )); then
    (( EXIT_CODE == 0 )) && $ON_SUCCESS
    (( EXIT_CODE != 0 )) && $ON_FAILURE
fi


elapsed_hours() {
    awk -v start="$1" -v end="$2" \
        'BEGIN { printf "%.2f", (end - start) / 3600 }'
}

add_hours() {
    awk -v total="$1" -v increment="$2" \
        'BEGIN { printf "%.2f", total + increment }'
}

if (( EXIT_CODE == 0 )); then
    now=$(date +%s)
    elapsed_session=$(elapsed_hours "$SESSION_START_TIME" "$now")
    CASE_TOTAL_WALLTIME=$(add_hours "${CASE_TOTAL_WALLTIME}" "$elapsed_session")

    if (( CASE_RESUBMIT_INDEX == CASE_RESUBMIT_MAX )); then
        msg="Case $SLURM_JOB_NAME completed"
        elapsed_total=$(elapsed_hours "$CASE_RUN_START_TIME" "$now")

        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - $msg"
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Total Walltime: ${CASE_TOTAL_WALLTIME} hours."
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Total Time Taken: ${elapsed_total} hours."
    fi
fi


if (( EXIT_CODE == 0 )) && (( CASE_RESUBMIT_INDEX == CASE_RESUBMIT_MAX )); then
    CASE_OUT="$CASE_DIR/OUTPUT"
    rm -rf "$CASE_DIR"/IC/R*_INPUT

    if (( CASE_ENSEMBLES == 1 )); then
        CASE_ARCHIVE_DIR="$ARCHIVE_DIR/ensembles"
    else
        CASE_ARCHIVE_DIR="$ARCHIVE_DIR/case"
    fi

    mkdir -p "$CASE_ARCHIVE_DIR"

    if (( CASE_ARCHIVE == 1 )); then
        cp -rf "$CASE_OUT"/*.nc "$CASE_ARCHIVE_DIR/"
        rm -rf "$CASE_ARCHIVE_DIR"/atmos_static*
        rm -rf "$CASE_ARCHIVE_DIR"/grid_spec*
        rm -rf "$CASE_OUT"
        rm -rf "$CASE_DIR"/HIST
        cp -f "$CASE_DIR"/state.yaml "$CASE_ARCHIVE_DIR/state.yaml"
        cp -rf "$CASE_DIR"/LOGS/shield "$CASE_ARCHIVE_DIR/shield_log"

        TARFILE="$ARCHIVE_DIR/case.tar.gz"

        if tar --use-compress-program='pigz -p 32' -cf "$TARFILE" -C "$CASE_DIR" . \
            && tar -tzf "$TARFILE" > /dev/null; then

            rm -rf "$CASE_DIR"
            rm -f "$CASE_DATA_SYMLINK"
            ln -s "$CASE_ARCHIVE_DIR" "$CASE_DATA_SYMLINK"

            echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Archived files to: $CASE_ARCHIVE_DIR"

            exit 0
        else
            echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - ERROR - Failed to archive case directory: $CASE_DIR"
            rm -f "$TARFILE"
            exit 1
        fi
    fi

fi


if (( EXIT_CODE == 0 && CASE_RESUBMIT_INDEX < CASE_RESUBMIT_MAX )); then
    SLURM_OPEN_MODE="append"
    CASE_TIME_LIMIT=$(squeue -j "$SLURM_JOB_ID" -h -o "%l")
    CASE_RESUBMIT_INDEX=$((CASE_RESUBMIT_INDEX + 1))
    source "$UFS_UTILS_DIR/drivers/sbatch.sh"
    scontrol top "$JOB_ID"
    exit 0
fi


exit $EXIT_CODE







