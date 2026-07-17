#!/bin/bash -l

# case_run.sh - main driver script for running SHiELD case on SLURM

# -------------------------------------------------------------------------#
# --- DO NOT MODIFY BELOW THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING! --- #
# -------------------------------------------------------------------------#

set -e

module purge


cd "$CASE_PWD"

export WORK_DIR="$JOBTMP_DIR/$CASE_PARENT_DIR/$CASE_NAME"
export CASE_DIR="$CASE_ROOT_DIR/$CASE_PARENT_DIR/$CASE_NAME"
export ARCHIVE_DIR="$ARCHIVE_ROOT_DIR/$CASE_PARENT_DIR/$CASE_NAME"


if  [ ! -d "$JOBTMP_DIR" ]; then 
    SYNC=0
    WORK_DIR="$CASE_DIR"
    TMP_DIR="$CASE_ROOT_DIR/tmp"
    mkdir -p "$JOBTMP_DIR"
else
    SYNC=1
    TMP_DIR="$JOBTMP_DIR/tmp"
    rm -rf "$WORK_DIR"
fi





if [ -z "$CASE_RUN_START_TIME" ]; then
    export CASE_RUN_START_TIME=$(date +%s)
fi

export TMP_DIR
export SESSION_START_TIME=$(date +%s)



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

SYNC_DIRS="rsync -a "$WORK_DIR/" "$CASE_DIR/""


$PREPROCESS # Run preprocess to stage grid and IC files (if needed)


if (( $(<"$EXIT_CODE_FILE") == 0)); then
    $SYNC_DIRS
    if (( $CASE_PREPROCESS_ONLY == 1 )); then
        rm -f "$CASE_DATA_SYMLINK"
        ln -s "$CASE_DIR" "$CASE_DATA_SYMLINK"
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - IC and Grid generation complete."
        exit 0
    fi
fi

if (( CASE_MULTI_NODE_FLAG == 1 )) || [[ -f "$SHIELD_NATIVE" ]]; then
    SHIELD="$WORK_DIR/shield"
else
    SHIELD="$SHIELD_PREFIX $WORK_DIR/shield"
fi



(( $(<"$EXIT_CODE_FILE") == 0 )) && $SHIELD
(( $(<"$EXIT_CODE_FILE") == 0 )) && $FREGRID



EXIT_CODE=$(<"$EXIT_CODE_FILE")

if  (( EXIT_CODE == 0 && SYNC == 1 )); then
    $SYNC_DIRS
fi


if (( EXIT_CODE == 0 )); then
    CURRENT_TIME=$(date +%s)
    SESSION_ELAPSED_TIME=$(( CURRENT_TIME - SESSION_START_TIME ))
    TOTAL_WALLTIME_TIME=$(( TOTAL_WALLTIME_TIME + SESSION_ELAPSED_TIME ))
    

    if (( CASE_RESUBMIT_INDEX == CASE_RESUBMIT_MAX )); then

        if (( CASE_ARCHIVE == 1 )); then

            CASE_OUT="$CASE_DIR/OUTPUT"
            rm -rf "$CASE_DIR"/IC/R*_INPUT

            if (( CASE_ENSEMBLES == 1 )); then
                CASE_ARCHIVE_DIR="$ARCHIVE_DIR/ensembles"
            else
                CASE_ARCHIVE_DIR="$ARCHIVE_DIR/case"
            fi

            mkdir -p "$CASE_ARCHIVE_DIR"

            cp -rf "$CASE_OUT"/*.nc "$CASE_ARCHIVE_DIR/"
            rm -rf "$CASE_OUT" "$CASE_DIR"/HIST

            cp -f "$CASE_DIR"/state.yaml "$CASE_ARCHIVE_DIR/state.yaml"
            cp -rf "$CASE_DIR"/LOGS/shield "$CASE_ARCHIVE_DIR/shield_log"

            TARFILE="$ARCHIVE_DIR/case.tar.gz"

            tar --use-compress-program='pigz -p 32' -cf "$TARFILE" -C "$CASE_DIR" .
            rm -rf "$CASE_DIR" "$CASE_DATA_SYMLINK"
            ln -s "$CASE_ARCHIVE_DIR" "$CASE_DATA_SYMLINK"
        fi

        to_hours() { awk -v seconds="$1" 'BEGIN {printf "%.2f\n", seconds / 3600}'; }
        
        TOTAL_RUNTIME=$(( CURRENT_TIME - CASE_RUN_START_TIME ))

        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Total Walltime: $(to_hours "$TOTAL_WALLTIME_TIME") hours."
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Total Time Taken: $(to_hours "$TOTAL_RUNTIME") hours."
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Archived files to: $CASE_ARCHIVE_DIR"
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Case $SLURM_JOB_NAME completed"
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







