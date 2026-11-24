#!/bin/bash -l

# setup_all.sh - one-time native setup of this pyUFS checkout, run from its root.
#
#   ./setup_all.sh --all | [--skip-env] [--skip-shield] [--skip-ufs] [--skip-fix]
#                          [--release RELEASE] [-j JOBS]
#
# At least one argument is required; with none the script aborts. --all runs every step,
# and the --skip-* options leave individual steps out.
#
# 1. Conda env: creates <parent of pyUFS>/conda_env/pyufs from tools/env.yaml, linked as
#    pyUFS/conda_env. Miniconda is installed under $HOME/miniconda3 when neither mamba
#    nor conda is found (same installer as tools/install_images.sh, for this user).
# 2. SHiELD: tools/build_shield.py, installed in <parent>/bin (pyUFS/bin).
# 3. UFS_UTILS: tools/build_ufs.py, installed in <parent>/bin/preprocessing.
# 4. Fix data: tools/update_fix.py, written to <parent>/fix (pyUFS/fix).
#
# Afterwards set shield_exe and ufs_bin_dir in run_config.yaml (the build scripts print
# the paths). A set ufs_bin_dir makes case_submit.sh run preprocess and regrid with the
# conda env instead of the container. Every step is safe to rerun.
set -euo pipefail

PYUFS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$PYUFS_DIR")/conda_env"
ENV_PREFIX="$ENV_DIR/pyufs"
MINICONDA_DIR="$HOME/miniconda3"
TMP_DIR="${TMP_DIR:-${TMPDIR:-/tmp}}"

usage() {
    echo "Usage: $0 --all | [--skip-env] [--skip-shield] [--skip-ufs] [--skip-fix] [--release RELEASE] [-j JOBS]" >&2
    echo "  --all   run every step (env, SHiELD, UFS_UTILS, fix data); combine with --skip-* to leave steps out" >&2
}

if (( $# == 0 )); then
    echo "SETUP_ALL - ERROR - no arguments given; pass --all to run every step" >&2
    usage
    exit 1
fi

fail_arg() { echo "SETUP_ALL - ERROR - $1 needs a value" >&2; usage; exit 1; }
DO_ENV=1 DO_SHIELD=1 DO_UFS=1 DO_FIX=1
SHIELD_ARGS=() UFS_ARGS=()
while (( $# )); do
    case "$1" in
        --all) ;;
        --skip-env) DO_ENV=0 ;;
        --skip-shield) DO_SHIELD=0 ;;
        --skip-ufs) DO_UFS=0 ;;
        --skip-fix) DO_FIX=0 ;;
        --release) (( $# >= 2 )) || fail_arg "$1"; SHIELD_ARGS+=(--release "$2"); shift ;;
        -j) (( $# >= 2 )) || fail_arg "$1"; UFS_ARGS+=(-j "$2"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "SETUP_ALL - ERROR - unknown argument: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

log() { echo "$(date '+%Y-%m-%d %H:%M') - SETUP_ALL - $*"; }
fail() { log "ERROR - $*" >&2; exit 1; }

if (( DO_ENV )); then
    log "INFO - conda env $ENV_PREFIX"
    CONDA_BIN=""
    for tool in mamba conda; do
        if command -v "$tool" >/dev/null 2>&1; then CONDA_BIN="$(command -v "$tool")"; break; fi
    done
    if [ -z "$CONDA_BIN" ] && [ -x "$MINICONDA_DIR/bin/conda" ]; then
        CONDA_BIN="$MINICONDA_DIR/bin/conda"
    fi
    if [ -z "$CONDA_BIN" ]; then
        command -v wget >/dev/null || fail "wget not found"
        MINICONDA_SH="$(mktemp "$TMP_DIR/miniconda.XXXXXX.sh")"
        trap 'rm -f "$MINICONDA_SH"' EXIT
        log "INFO - downloading the Miniconda installer"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_SH"
        bash "$MINICONDA_SH" -b -p "$MINICONDA_DIR"
        "$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
        "$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
        CONDA_BIN="$MINICONDA_DIR/bin/conda"
    fi

    mkdir -p "$ENV_DIR"
    if [ -x "$ENV_PREFIX/bin/python" ]; then
        "$CONDA_BIN" env update --prefix "$ENV_PREFIX" --file "$PYUFS_DIR/tools/env.yaml" --prune --yes
    else
        "$CONDA_BIN" env create --prefix "$ENV_PREFIX" --file "$PYUFS_DIR/tools/env.yaml" --yes
    fi
    [ -x "$ENV_PREFIX/bin/python" ] || fail "$ENV_PREFIX/bin/python missing after the env build"
    "$ENV_PREFIX/bin/python" -c "import dask.array, f90nml, xarray, xesmf, cfgrib, cartopy, yaml" \
        || fail "the conda env does not import its core packages; rebuild it: rm -rf $ENV_PREFIX and rerun"
    "$ENV_PREFIX/bin/python" "$PYUFS_DIR/tools/pyufs_links.py"
fi

[ -x "$ENV_PREFIX/bin/python" ] || fail "conda env not found: $ENV_PREFIX (run without --skip-env)"
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"

if (( DO_SHIELD )); then
    log "INFO - building SHiELD"
    python "$PYUFS_DIR/tools/build_shield.py" ${SHIELD_ARGS[@]+"${SHIELD_ARGS[@]}"}
fi
if (( DO_UFS )); then
    log "INFO - building UFS_UTILS"
    python "$PYUFS_DIR/tools/build_ufs.py" ${UFS_ARGS[@]+"${UFS_ARGS[@]}"}
fi
if (( DO_FIX )); then
    log "INFO - updating the fix data"
    python "$PYUFS_DIR/tools/update_fix.py"
fi

log "INFO - done; set shield_exe and ufs_bin_dir in run_config.yaml to the paths printed above"