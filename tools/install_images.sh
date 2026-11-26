#!/bin/bash
# Build the preprocess and shield Apptainer images used by the workflow (container mode only;
# native mode uses setup_all.sh instead).
#
#   CONTAINERS_DIR=/path/to/containers TMP_DIR=/scratch bash install_images.sh [preprocess] [shield]
#
# Names select the images to build (default: both). An image that already exists is
# skipped, so the script is cheap to call before every job; FORCE=1 rebuilds it.
# Images are written to $PREPROCESS_SIF and $SHIELD_SIF (case_submit.py sets them from
# preprocess_image and shield_image); when unset, to $CONTAINERS_DIR/{preprocess,shield}.sif.
# Image tags default to "latest"; pin them with PREPROCESS_TAG and SHIELD_TAG.
# gfdlfv3/fre-nctools (tag FREGRID_TAG) is pulled only to take its Open MPI help
# files for the other two images; it is deleted before the conda step and is not
# installed.
# A manifest of source, checksum and conda packages of each built image is written to
# $CONTAINERS_DIR/manifest/ (<name>.txt and <name>.conda.txt). The preprocess image provides the UFS_UTILS
# executables at /UFS_UTILS/exec and also runs src/fv3_regrid.py.
set -euo pipefail

: "${CONTAINERS_DIR:?CONTAINERS_DIR must be set}"
TMP_DIR="${TMP_DIR:-${TMPDIR:-/tmp}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.yaml"
MANIFEST_DIR="$CONTAINERS_DIR/manifest"

HELP_SOURCE="docker://gfdlfv3/fre-nctools:${FREGRID_TAG:-latest}"

declare -A SOURCES=(
    [preprocess]="docker://gfdlfv3/preprocessing:${PREPROCESS_TAG:-latest}"
    [shield]="docker://gfdlfv3/shield:${SHIELD_TAG:-latest}"
)
CONDA_IMAGES=(preprocess)

declare -A TARGET=(
    [preprocess]="${PREPROCESS_SIF:-$CONTAINERS_DIR/preprocess.sif}"
    [shield]="${SHIELD_SIF:-$CONTAINERS_DIR/shield.sif}"
)

log() { echo "$(date '+%Y-%m-%d %H:%M') - INSTALL_IMAGES - $*"; }
fail() {
    log "ERROR - $*" >&2
    exit 1
}

# --- 0. Which images to build ---
NAMES=("$@")
(( ${#NAMES[@]} )) || NAMES=(preprocess shield)
TODO=()
for NAME in "${NAMES[@]}"; do
    [[ -v SOURCES[$NAME] ]] || fail "unknown image '$NAME' (preprocess or shield)"
    if [[ -f "${TARGET[$NAME]}" && "${FORCE:-0}" != 1 ]]; then
        log "INFO - ${TARGET[$NAME]} exists; skipped"
    else
        TODO+=("$NAME")
    fi
done
(( ${#TODO[@]} )) || exit 0

# --- 1. Checks and scratch space ---
command -v apptainer >/dev/null || fail "apptainer not found"
command -v wget >/dev/null || fail "wget not found"
[ -f "$ENV_FILE" ] || fail "env.yaml not found in $SCRIPT_DIR"

mkdir -p "$CONTAINERS_DIR" "$MANIFEST_DIR" "$TMP_DIR"
WORK="$(mktemp -d "$TMP_DIR/apptainer.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/sandboxes" "$WORK/cache"

if command -v module >/dev/null 2>&1; then
    module purge || true
fi
unset LD_LIBRARY_PATH
export APPTAINER_CACHEDIR="$WORK/cache"
export APPTAINER_BINDPATH="$WORK:/workdir"

MINICONDA_SH="$WORK/miniconda.sh"
log "INFO - downloading the Miniconda installer"
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_SH"
chmod +x "$MINICONDA_SH"

# --- 2. Pull and unpack (fre-nctools only supplies the Open MPI help files) ---
HELP_SANDBOX="$WORK/sandboxes/help"
log "INFO - pulling $HELP_SOURCE (Open MPI help files only)"
apptainer pull "$WORK/help.sif" "$HELP_SOURCE"
apptainer build --sandbox "$HELP_SANDBOX" "$WORK/help.sif"

for NAME in "${TODO[@]}"; do
    log "INFO - pulling ${SOURCES[$NAME]}"
    apptainer pull "$WORK/$NAME.sif" "${SOURCES[$NAME]}"
    apptainer build --sandbox "$WORK/sandboxes/$NAME" "$WORK/$NAME.sif"

    mkdir -p "$WORK/sandboxes/$NAME/opt/openmpi"
    cp -rf "$HELP_SANDBOX/opt/openmpi/share" "$WORK/sandboxes/$NAME/opt/openmpi/"
done

# The fre-nctools image is not used again; free its space before the conda step.
rm -rf "$HELP_SANDBOX" "$WORK/help.sif"

# --- 3. Conda environments for the Python stages ---
for NAME in "${TODO[@]}"; do
    [[ " ${CONDA_IMAGES[*]} " == *" $NAME "* ]] || continue
    SANDBOX="$WORK/sandboxes/$NAME"
    cp "$MINICONDA_SH" "$SANDBOX/miniconda.sh"
    cp "$ENV_FILE" "$SANDBOX/env.yaml"

    log "INFO - installing the $NAME conda environment"
    apptainer exec --writable --no-home "$SANDBOX" bash -euo pipefail -c "
        mkdir -p /workdir/tmp
        export TMPDIR=/workdir/tmp
        ./miniconda.sh -b -p /opt/conda
        /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
        /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
        /opt/conda/bin/conda install --name base --channel conda-forge mamba --yes --quiet
        /opt/conda/bin/mamba env create --name $NAME --file /env.yaml --yes --quiet
        /opt/conda/bin/mamba clean --all --yes
        ln -sf /opt/conda/envs/$NAME/bin/python /$NAME
        ln -sf /opt/conda/envs/$NAME/bin/wget /wget
        ln -sf /opt/conda/envs/$NAME/bin/wgrib2 /wgrib2
        /opt/conda/bin/conda list --name $NAME --explicit > /workdir/$NAME.conda.txt
        rm -rf /workdir/tmp /miniconda.sh /env.yaml
    "
    [ -x "$SANDBOX/opt/conda/envs/$NAME/bin/python" ] || fail "$NAME python environment missing"
done

if [[ -d "$WORK/sandboxes/preprocess" ]]; then
    [ -x "$WORK/sandboxes/preprocess/UFS_UTILS/exec/chgres_cube" ] ||
        fail "preprocess image has no /UFS_UTILS/exec/chgres_cube"
fi

# --- 4. Final images, replaced only after a successful build ---
for NAME in "${TODO[@]}"; do
    mkdir -p "$(dirname "${TARGET[$NAME]}")"
    apptainer build "$WORK/$NAME.final.sif" "$WORK/sandboxes/$NAME"
    mv -f "$WORK/$NAME.final.sif" "${TARGET[$NAME]}"
    {
        echo "built: $(date -u '+%Y-%m-%dT%H:%MZ')"
        echo "source: ${SOURCES[$NAME]}"
        echo "help_files.source: $HELP_SOURCE"
        echo "sha256: $(sha256sum "${TARGET[$NAME]}" | cut -d' ' -f1)"
    } >"$MANIFEST_DIR/$NAME.txt"
    [ ! -f "$WORK/$NAME.conda.txt" ] || cp "$WORK/$NAME.conda.txt" "$MANIFEST_DIR/"
    log "INFO - wrote ${TARGET[$NAME]}"
done
log "INFO - manifest written to $MANIFEST_DIR"
