#!/bin/bash
# Build the fregrid, preprocess and shield Apptainer images used by the workflow.
#
#   CONTAINERS_DIR=/path/to/containers TMP_DIR=/scratch bash install_images.sh
#
# Images are written to $CONTAINERS_DIR/{fregrid,preprocess,shield}.sif, which
# must match fregrid_image, preprocess_image and shield_image in run_config.yaml.
# Image tags default to "latest"; pin them with FREGRID_TAG, PREPROCESS_TAG and
# SHIELD_TAG. A manifest of sources, checksums and conda packages is written to
# $CONTAINERS_DIR/manifest/. The preprocess image provides the UFS_UTILS
# executables at /UFS_UTILS/exec.
set -euo pipefail

: "${CONTAINERS_DIR:?CONTAINERS_DIR must be set}"
TMP_DIR="${TMP_DIR:-${TMPDIR:-/tmp}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.yaml"
MANIFEST_DIR="$CONTAINERS_DIR/manifest"

declare -A SOURCES=(
    [fregrid]="docker://gfdlfv3/fre-nctools:${FREGRID_TAG:-latest}"
    [preprocess]="docker://gfdlfv3/preprocessing:${PREPROCESS_TAG:-latest}"
    [shield]="docker://gfdlfv3/shield:${SHIELD_TAG:-latest}"
)
CONDA_IMAGES=(fregrid preprocess)

log() { echo "$(date '+%Y-%m-%d %H:%M') - INSTALL_IMAGES - $*"; }
fail() {
    log "ERROR - $*" >&2
    exit 1
}

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

# --- 2. Pull and unpack (fregrid first: its OpenMPI share files are reused) ---
for NAME in fregrid preprocess shield; do
    log "INFO - pulling ${SOURCES[$NAME]}"
    apptainer pull "$WORK/$NAME.sif" "${SOURCES[$NAME]}"
    apptainer build --sandbox "$WORK/sandboxes/$NAME" "$WORK/$NAME.sif"

    if [ "$NAME" != "fregrid" ]; then
        mkdir -p "$WORK/sandboxes/$NAME/opt/openmpi"
        cp -rf "$WORK/sandboxes/fregrid/opt/openmpi/share" "$WORK/sandboxes/$NAME/opt/openmpi/"
    fi
done

# --- 3. Conda environments for the Python stages ---
for NAME in "${CONDA_IMAGES[@]}"; do
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

[ -x "$WORK/sandboxes/preprocess/UFS_UTILS/exec/chgres_cube" ] ||
    fail "preprocess image has no /UFS_UTILS/exec/chgres_cube"

# --- 4. Final images, replaced only after a successful build ---
{
    echo "built: $(date -u '+%Y-%m-%dT%H:%MZ')"
    for NAME in fregrid preprocess shield; do
        echo "$NAME.source: ${SOURCES[$NAME]}"
    done
} >"$WORK/images.txt"

for NAME in fregrid preprocess shield; do
    apptainer build "$WORK/$NAME.final.sif" "$WORK/sandboxes/$NAME"
    mv -f "$WORK/$NAME.final.sif" "$CONTAINERS_DIR/$NAME.sif"
    echo "$NAME.sha256: $(sha256sum "$CONTAINERS_DIR/$NAME.sif" | cut -d' ' -f1)" >>"$WORK/images.txt"
    log "INFO - wrote $CONTAINERS_DIR/$NAME.sif"
done

cp "$WORK/images.txt" "$MANIFEST_DIR/images.txt"
cp "$WORK"/*.conda.txt "$MANIFEST_DIR/"
log "INFO - manifest written to $MANIFEST_DIR"
