#!/bin/bash -l

# case_run.sh - main driver script for running SHiELD case on SLURM

# -------------------------------------------------------------------------#
# --- DO NOT MODIFY BELOW THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING! --- #
# -------------------------------------------------------------------------#

set -e

module purge

echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Starting Case"

export WORK_DIR="$JOB_TMP/$CASE_PARENT_DIR/$CASE_NAME"
export CASE_DIR="$CASE_ROOT/$CASE_PARENT_DIR/$CASE_NAME"
export ARCHIVE_DIR="$ARCHIVE_ROOT/$CASE_PARENT_DIR/$CASE_NAME"
export TMP_DIR="$JOB_TMP/tmp"


if [ -z "$CASE_RUN_START_TIME" ]; then
    export CASE_RUN_START_TIME=$(date +%s)
fi


cd "$RUN_DIR"


if  [ ! -d "$JOB_TMP" ]; then 
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
    source "$UFS_UTILS/configs/install_images.sh" > $WORK_DIR/image_build.log 2>&1
fi


# CREATE SYMLINK TO WORK_DIR
rm -f "$DATA_SYMLINK"
ln -s "$WORK_DIR" "$DATA_SYMLINK"

# SYNC CASE_DIR TO WORK_DIR
if (( SYNC == 1 )); then
    rsync -a --delete "$CASE_DIR/" "$WORK_DIR/"

fi


# RUNTIME FILES
ID_FILE="$WORK_DIR/run.id"
EXIT_CODE_FILE="$WORK_DIR/exit_code"
SHIELD_NATIVE="$WORK_DIR/shield.native"

touch "$EXIT_CODE_FILE"

# CHECK FOR PREVIOUS RUN
if [ -f "$ID_FILE" ]; then
    PREV_RUN_ID=$(cat "$ID_FILE")
    exec >>"$LOG_FILE" 2>&1
else
    PREV_RUN_ID=0
    exec >"$LOG_FILE" 2>&1
fi

CURR_RUN_ID=$((PREV_RUN_ID + 1))

if [ "$CURR_RUN_ID" -gt 1 ]; then
    if [ "$(cat "$EXIT_CODE_FILE")" -ne 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - ERROR - Previous run $PREV_RUN_ID failed. Aborting!."
        exit 1
    fi
fi

echo "$CURR_RUN_ID" > "$ID_FILE"


export CURR_RUN_ID
export TMPDIR="$TMP_DIR"
export APPTAINER_CACHEDIR=$TMP_DIR
export APPTAINER_HOME=$HOME
export APPTAINER_BINDPATH=$(printf "%s" "$CONTAINER_BINDPATH" | base64 -d)


FREGRID="apptainer exec $FREGRID_SIF $UFS_UTILS/fregrid"
PREPROCESS="apptainer exec $PREPROCESS_SIF $UFS_UTILS/preprocess"
ON_SUCCESS="rsync -a --delete "$WORK_DIR/" "$CASE_DIR/""
ON_FAILURE="rsync -a --delete "$WORK_DIR/LOGS/" "$CASE_DIR/LOGS/""

$PREPROCESS # Run preprocess to stage grid and IC files (if needed)

if [ "$(cat "$EXIT_CODE_FILE")" -eq 0 ] && [ -f "$WORK_DIR/ic.only" ]; then
    $ON_SUCCESS && rm -f "$WORK_DIR/ic.only"
    echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - IC and Grid generation complete."
    exit 0
fi


if [[ "$SBATCH_MULTI_NODE" ==  1 ]] || [[ -f "$SHIELD_NATIVE" ]]; then
    SHIELD="$WORK_DIR/shield"
else
    SHIELD="apptainer exec $SHIELD_SIF $WORK_DIR/shield"
fi


RUN_START_TIME=$(date +%s)


[ "$(cat "$EXIT_CODE_FILE")" -eq 0 ] && $SHIELD
[ "$(cat "$EXIT_CODE_FILE")" -eq 0 ] && $FREGRID


RUN_END_TIME=$(date +%s)

if (( SYNC == 1 )); then
    [ "$(cat "$EXIT_CODE_FILE")" -eq 0 ] && $ON_SUCCESS
    [ "$(cat "$EXIT_CODE_FILE")" -ne 0 ] && $ON_FAILURE
fi


rm -f "$DATA_SYMLINK"
ln -s "$CASE_DIR" "$DATA_SYMLINK"


if (( RESUBMIT_COUNT == 0 )); then
    rm -f "$ID_FILE"
fi



EXIT_CODE=$(cat "$EXIT_CODE_FILE")
if (( RESUBMIT_COUNT == 0 )) && (( EXIT_CODE == 0 )); then

    CASE_OUT="$CASE_DIR/OUTPUT"
    rm -rf "$CASE_DIR"/INIT_DATA/R*_INPUT
    mkdir -p "$ARCHIVE_DIR"

    if (( ARCHIVE_DATA == 1 )); then
        cp -rf "$CASE_OUT"/*.nc "$ARCHIVE_DIR/"
        rm -rf "$ARCHIVE_DIR"/atmos_static*
        rm -rf "$ARCHIVE_DIR"/grid_spec*
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Archived files to: $ARCHIVE_DIR"
        rm -rf "$CASE_OUT"
    fi


fi

if (( EXIT_CODE == 0 && RESUBMIT_COUNT > 0 )); then
    SLURM_OPEN_MODE="append"
    RESUBMIT=$((RESUBMIT_COUNT - 1))
    source "$UFS_UTILS/tools/sbatch.sh"
    scontrol top "$JOB_ID"
fi


elapsed_hours () {
    awk -v start="$1" -v end="$2" 'BEGIN {printf "%.2f", (end - start)/3600}'
}

if (( EXIT_CODE == 0 )); then
    if (( RESUBMIT_COUNT > 0 )); then
        msg="Restart $((CURR_RUN_ID - 1)) completed"
        elapsed=$(elapsed_hours "$RUN_START_TIME" "$RUN_END_TIME")
    else
        msg="Case $SLURM_JOB_NAME completed"
        elapsed=$(elapsed_hours "$CASE_RUN_START_TIME" "$(date +%s)")
    fi


    echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - $msg in ${elapsed} hours."
fi

exit "$EXIT_CODE"






