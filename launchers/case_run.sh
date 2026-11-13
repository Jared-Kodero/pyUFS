#!/bin/bash -l

# case_run.sh - main driver script for running SHiELD case on SLURM

# -------------------------------------------------------------------------#
# --- DO NOT MODIFY BELOW THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING! --- #
# -------------------------------------------------------------------------#

set -e

module purge


cd "$CASE_PWD"

export CASE_DIR="$CASE_ROOT_DIR/$CASE_PARENT_DIR/$CASE_NAME"
export ARCHIVE_DIR="$ARCHIVE_ROOT_DIR/$CASE_PARENT_DIR/$CASE_NAME"





if [ -z "$CASE_RUN_START_TIME" ]; then
    export CASE_RUN_START_TIME=$(date +%s)
fi

export SESSION_START_TIME=$(date +%s)



# PREPARE DIRECTORIES
mkdir -p "$CASE_DIR"


# BUILD MISSING IMAGES (run as a child process so its module/env changes do not leak)
if [ ! -f "$SHIELD_SIF" ] || [ ! -f "$PREPROCESS_SIF" ] || [ ! -f "$FREGRID_SIF" ]; then
    bash "$UFS_UTILS_DIR/tools/install_images.sh" > "$CASE_DIR"/image_build.log 2>&1
    for SIF in "$SHIELD_SIF" "$PREPROCESS_SIF" "$FREGRID_SIF"; do
        if [ ! -f "$SIF" ]; then
            echo "Case.Run - ERROR - image $SIF not found after build; images are written to $CONTAINERS_DIR" >&2
            exit 1
        fi
    done
fi


# CREATE SYMLINK TO CASE_DIR
rm -f "$CASE_DATA_SYMLINK"
ln -s "$CASE_DIR" "$CASE_DATA_SYMLINK"


# RUNTIME FILES
EXIT_CODE_FILE="$CASE_DIR/exit_code"
SHIELD_NATIVE="$CASE_DIR/shield.native"

echo 0 > "$EXIT_CODE_FILE"

# CHECK FOR PREVIOUS RUN
if (( CASE_RESUBMIT_INDEX  > 0 )); then
    exec >>"$CASE_LOG_FILE" 2>&1
else
     exec >"$CASE_LOG_FILE" 2>&1
fi


# The one temporary directory of the case (state.tmp), shared by the workflow and the
# processes it starts: Python multiprocessing (3.14 forkserver) keeps its pymp-*
# directory here, so cases that run at the same time never share one. It is
# removed after a successful segment and kept after a failure for debugging. A
# kept directory is cleared when the next job of the case starts, so a rerun
# never picks up files from the failed one.
export TMPDIR="$CASE_DIR/TMP"
[ -n "$CASE_DIR" ] && rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"
export APPTAINER_CACHEDIR="$TMPDIR"
export APPTAINER_HOME=$HOME
export APPTAINER_BINDPATH=$(printf "%s" "$CONTAINER_BINDPATH" | base64 -d)


FREGRID="apptainer exec $FREGRID_SIF $UFS_UTILS_DIR/launchers/fregrid"
PREPROCESS="apptainer exec $PREPROCESS_SIF $UFS_UTILS_DIR/launchers/preprocess"
SHIELD_PREFIX="apptainer exec $SHIELD_SIF"


# Nodes of this job for state.yaml (comma separated; read by get_allocated_nodes)
export CASE_NODE_LIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | paste -sd, -)


$PREPROCESS # Run preprocess to stage grid and IC files (if needed)


if (( $(<"$EXIT_CODE_FILE") == 0 && CASE_PREPROCESS_ONLY == 1 )); then
    echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - IC and Grid generation complete."
    exit 0
fi

if (( CASE_MULTI_NODE_FLAG == 1 )) || [[ -f "$SHIELD_NATIVE" ]]; then
    SHIELD="$CASE_DIR/shield"
else
    SHIELD="$SHIELD_PREFIX $CASE_DIR/shield"
fi




# Stage exit codes are carried by EXIT_CODE_FILE, so a failed stage still
# reaches the log copy below instead of aborting under set -e.
if (( $(<"$EXIT_CODE_FILE") == 0 )); then $SHIELD || true; fi
if (( $(<"$EXIT_CODE_FILE") == 0 )); then $FREGRID || true; fi



EXIT_CODE=$(<"$EXIT_CODE_FILE")

if (( EXIT_CODE == 0 )); then
    rm -rf "$TMPDIR"
    unset TMPDIR APPTAINER_CACHEDIR
fi

if (( EXIT_CODE == 0 )); then
    CURRENT_TIME=$(date +%s)
    SESSION_ELAPSED_TIME=$(( CURRENT_TIME - SESSION_START_TIME ))
    TOTAL_WALLTIME_TIME=$(( TOTAL_WALLTIME_TIME + SESSION_ELAPSED_TIME ))
    

    if (( CASE_RESUBMIT_INDEX == CASE_RESUBMIT_MAX )); then

        if (( CASE_ARCHIVE == 1 )); then

            mkdir -p "$ARCHIVE_DIR"

            cp -rf "$CASE_DIR"/OUTPUT/*.nc "$CASE_DIR"/state.yaml "$ARCHIVE_DIR/"
            cp -rf "$CASE_DIR"/LOGS/shield "$ARCHIVE_DIR/shield_log"

            # Only the final restart, state.yaml and run_config.yaml are kept; the grid,
            # orography and ICs are rebuilt from them (case_continue.sh --restart-dir).
            cp "$CASE_PWD/run_config.yaml" "$CASE_DIR/"
            tar --use-compress-program='pigz -p 32' -cf "$ARCHIVE_DIR/restart.tar.gz" \
                -C "$CASE_DIR" RESTART state.yaml run_config.yaml
                
            ln -sfn "$ARCHIVE_DIR" "$CASE_DATA_SYMLINK"
        fi

        to_hours() { awk -v seconds="$1" 'BEGIN {printf "%.2f\n", seconds / 3600}'; }
        
        TOTAL_RUNTIME=$(( CURRENT_TIME - CASE_RUN_START_TIME ))

        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Total Walltime: $(to_hours "$TOTAL_WALLTIME_TIME") hours."
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Total Time Taken: $(to_hours "$TOTAL_RUNTIME") hours."
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Archived files to: $ARCHIVE_DIR"
        echo "$(date '+%Y-%m-%d %H:%M') - UFS_UTILS - INFO - Case $SLURM_JOB_NAME completed"

        if (( CASE_ARCHIVE == 1 )); then
            cp -f "$CASE_LOG_FILE" "$ARCHIVE_DIR/shield_log/"
        fi
    fi
fi



if (( EXIT_CODE == 0 && CASE_RESUBMIT_INDEX < CASE_RESUBMIT_MAX )); then
    SLURM_OPEN_MODE="append"
    CASE_TIME_LIMIT=$(squeue -j "$SLURM_JOB_ID" -h -o "%l")
    CASE_RESUBMIT_INDEX=$((CASE_RESUBMIT_INDEX + 1))
    export CASE_CONTINUE_RUN=0  # only the segment started by case_continue.sh resumes
    unset CASE_RESTART_DIR CASE_RESTART_STATE  # only the first job stages a restart
    source "$UFS_UTILS_DIR/launchers/sbatch.sh"
    scontrol top "$JOB_ID"
    exit 0
fi

exit "$EXIT_CODE"






