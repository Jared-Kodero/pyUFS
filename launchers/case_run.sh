#!/bin/bash -l

# case_run.sh - main driver script for running SHiELD case on SLURM

# -------------------------------------------------------------------------#
# --- DO NOT MODIFY BELOW THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING! --- #
# -------------------------------------------------------------------------#

set -e

module purge


cd "$SUBMIT_DIR"

export CASE_DIR="$CASE_ROOT_DIR/$CASE_PARENT_DIR/$CASE_NAME"
export ARCHIVE_DIR="$ARCHIVE_ROOT_DIR/$CASE_ARCHIVE_SUBDIR/$CASE_PARENT_DIR/$CASE_NAME"
export CASE_NODE_LIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")




if [ -z "$CASE_RUN_START_TIME" ]; then
    export CASE_RUN_START_TIME=$(date +%s)
fi

export SESSION_START_TIME=$(date +%s)



# PREPARE DIRECTORIES
mkdir -p "$CASE_DIR"


# BUILD MISSING IMAGES. install_images.sh skips images that exist and fails if one
# cannot be built; it runs as a child process so its module/env changes do not leak.
IMAGES=()
if (( ${CASE_SHIELD_USE_CONTAINER:-0} == 1 )); then IMAGES+=(shield); fi
if (( ${CASE_PREPROCESS_USE_CONTAINER:-0} == 1 )); then IMAGES+=(preprocess); fi
if (( ${#IMAGES[@]} )); then
    bash "$PYUFS_DIR/tools/install_images.sh" "${IMAGES[@]}" > "$CASE_DIR/image_build.log" 2>&1
fi

# CREATE SYMLINK TO CASE_DIR
rm -f "$CASE_DATA_SYMLINK"
ln -s "$CASE_DIR" "$CASE_DATA_SYMLINK"


# RUNTIME FILES
EXIT_CODE_FILE="$CASE_DIR/exit_code"

echo 0 > "$EXIT_CODE_FILE"

# CHECK FOR PREVIOUS RUN
if (( CASE_RESUBMIT_INDEX  > 0 )); then
    exec >>"$CASE_LOG_FILE" 2>&1
else
     exec >"$CASE_LOG_FILE" 2>&1
fi


export TMPDIR="$CASE_DIR/TMP"
[ -n "$CASE_DIR" ] && rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

export APPTAINER_CACHEDIR="$TMPDIR"
export APPTAINER_HOME=$HOME
export APPTAINER_BINDPATH=$(printf "%s" "$CONTAINER_BINDPATH" | base64 -d)



# Python steps: natively through env.sh (conda env and modules, in its own process so
# they never reach SHiELD), or inside the preprocess image (PYUFS_ENV empty).
if (( ${CASE_PREPROCESS_USE_CONTAINER:-0} == 0 )); then
    PYUFS_ENV="$PYUFS_DIR/launchers/env.sh"
    PREPROCESS="$PYUFS_DIR/launchers/preprocess"
    REGRID="$PYUFS_DIR/launchers/regrid"
    FINALIZE="$PYUFS_DIR/launchers/finalize"
else
    PYUFS_ENV=""
    PREPROCESS="apptainer exec $PREPROCESS_SIF $PYUFS_DIR/launchers/preprocess"
    REGRID="apptainer exec $PREPROCESS_SIF $PYUFS_DIR/launchers/regrid"
    FINALIZE="apptainer exec $PREPROCESS_SIF $PYUFS_DIR/launchers/finalize"
fi


$PYUFS_ENV $PREPROCESS  # stage grid and IC files (if needed)

# Preprocess-only cases stop here (the driver has logged the completion).
if (( CASE_PREPROCESS_ONLY == 1 )); then exit "$(<"$EXIT_CODE_FILE")"; fi

if (( ${CASE_SHIELD_USE_CONTAINER:-0} == 1 )); then
    SHIELD_PREFIX="apptainer exec $SHIELD_SIF"
    SHIELD="$SHIELD_PREFIX $CASE_DIR/shield"
else
    SHIELD="$CASE_DIR/shield"
fi




# Stage exit codes are carried by EXIT_CODE_FILE, so a failed stage still
# reaches the log copy below instead of aborting under set -e.
if (( $(<"$EXIT_CODE_FILE") == 0 )); then $SHIELD || true; fi
if (( $(<"$EXIT_CODE_FILE") == 0 )); then $PYUFS_ENV $REGRID || true; fi
if (( $(<"$EXIT_CODE_FILE") == 0 )); then $PYUFS_ENV $FINALIZE || true; fi

EXIT_CODE=$(<"$EXIT_CODE_FILE")

if (( EXIT_CODE == 0 )); then
    rm -rf "$TMPDIR"
    unset TMPDIR APPTAINER_CACHEDIR
fi


if (( EXIT_CODE == 0 && CASE_RESUBMIT_INDEX < CASE_RESUBMIT_MAX )); then
    SLURM_OPEN_MODE="append"
    CASE_TIME_LIMIT=$(squeue -j "$SLURM_JOB_ID" -h -o "%l")
    CASE_RESUBMIT_INDEX=$((CASE_RESUBMIT_INDEX + 1))
    export CASE_CONTINUE_RUN=0  # only the segment started by case_continue.sh resumes
    unset CASE_RESTART_DIR CASE_RESTART_STATE  # only the first job stages a restart
    source "$PYUFS_DIR/launchers/sbatch.sh"
    scontrol top "$JOB_ID"
    exit 0
fi

exit "$EXIT_CODE"