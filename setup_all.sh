#!/bin/bash -l

# setup_all.sh - one-time native setup of this pyUFS checkout, run from its root.
#
#   ./setup_all.sh --all | [--skip-env] [--skip-shield] [--skip-ufs] [--skip-fix]
#                          [--release RELEASE [RELEASE ...]] [-j JOBS]
#
# At least one argument is required; with none the script aborts. --all runs every step,
# and the --skip-* options leave individual steps out.
#
# 1. Conda env: creates <parent of pyUFS>/conda_env/pyufs from tools/env.yaml, linked as
#    pyUFS/conda_env. Miniconda is installed under $HOME/miniconda3 when neither mamba
#    nor conda is found (same installer as tools/install_images.sh, for this user).
# 2. SHiELD: tools/build_shield.py, installed in <parent>/bin (pyUFS/bin) as
#    <release>.fv3.public_SHiELD_nh.repro.32bit.gnu.x with a .manifest.yaml beside it.
#    --release takes one or more releases (--release 202411 202604); SHiELD is built once
#    per release, in the order given. Without --release the build script default is used.
# 3. UFS_UTILS: tools/build_ufs.py, installed in <parent>/bin/preprocessing with manifest.yaml.
# 4. Fix data: tools/update_fix.py, written to <parent>/fix (pyUFS/fix).
#
# Afterwards set shield_exe (with the matching shield_release) and ufs_bin_dir in
# run_config.yaml; the build scripts print the paths. A set ufs_bin_dir makes
# case_submit.sh run preprocess and regrid with the conda env instead of the
# container. Every step is safe to rerun.
#
# The builds are heavy: run this script in an interactive session (interact) on a compute node.
set -euo pipefail


PYUFS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$PYUFS_DIR")/conda_env"
ENV_PREFIX="$ENV_DIR/pyufs"
MINICONDA_DIR="$HOME/miniconda3"
TMP_DIR="${TMP_DIR:-${TMPDIR:-/tmp}}"

usage() {
    echo "Usage: $0 --all | [--skip-env] [--skip-shield] [--skip-ufs] [--skip-fix] [--release RELEASE [RELEASE ...]] [-j JOBS]" >&2
    echo "  --all       run every step (env, SHiELD, UFS_UTILS, fix data); combine with --skip-* to leave steps out" >&2
    echo "  --release   one or more SHiELD releases to build, e.g. --release 202411 202604" >&2
}

if (( $# == 0 )); then
    echo "SETUP_ALL - ERROR - no arguments given; pass --all to run every step" >&2
    usage
    exit 1
fi

fail_arg() { echo "SETUP_ALL - ERROR - $1 needs a value" >&2; usage; exit 1; }
DO_ENV=1 DO_SHIELD=1 DO_UFS=1 DO_FIX=1
RELEASES=() UFS_ARGS=()
while (( $# )); do
    case "$1" in
        --all) ;;
        --skip-env) DO_ENV=0 ;;
        --skip-shield) DO_SHIELD=0 ;;
        --skip-ufs) DO_UFS=0 ;;
        --skip-fix) DO_FIX=0 ;;
        --release)
            # Stores every following value up to the next option: --release 202411 202604
            shift
            while (( $# )) && [[ "$1" != -* ]]; do RELEASES+=("$1"); shift; done
            continue ;;
        -j) (( $# >= 2 )) || fail_arg "$1"; UFS_ARGS+=(-j "$2"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "SETUP_ALL - ERROR - unknown argument: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

log() { echo "$(date '+%Y-%m-%d %H:%M') - SETUP_ALL - $*"; }
fail() { log "ERROR - $*" >&2; exit 1; }

# Step 1: the conda environment that runs preprocessing, regridding and the build scripts.
if (( DO_ENV )); then
    module purge
    log "INFO - conda env $ENV_PREFIX"
    # Use mamba or conda if installed, else the Miniconda of this user, else install it.
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

    # Create the environment, or bring an existing one in line with tools/env.yaml.
    mkdir -p "$ENV_DIR"
    if [ -x "$ENV_PREFIX/bin/python" ]; then
        "$CONDA_BIN" env update --prefix "$ENV_PREFIX" --file "$PYUFS_DIR/tools/env.yaml" --prune --yes
    else
        "$CONDA_BIN" env create --prefix "$ENV_PREFIX" --file "$PYUFS_DIR/tools/env.yaml" --yes
    fi
    [ -x "$ENV_PREFIX/bin/python" ] || fail "$ENV_PREFIX/bin/python missing after the env build"
    # Fail now, not in a later job, if the environment is broken.
    "$ENV_PREFIX/bin/python" -c "import dask.array, f90nml, xarray, xesmf, cfgrib, cartopy, yaml" \
        || fail "the conda env does not import its core packages; rebuild it: rm -rf $ENV_PREFIX and rerun"
    "$ENV_PREFIX/bin/python" "$PYUFS_DIR/tools/pyufs_links.py"
fi

# The build scripts below run with the Python of this environment.
[ -x "$ENV_PREFIX/bin/python" ] || fail "conda env not found: $ENV_PREFIX (run without --skip-env)"
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"

# Step 2: build SHiELD (the longest step; tools/build_shield.py describes each stage).
# One build per release, sequentially; a failing build aborts the script (set -e).
if (( DO_SHIELD )); then
    (( ${#RELEASES[@]} )) || RELEASES=("")   # no --release: one build with the script default
    for release in "${RELEASES[@]}"; do
        log "INFO - building SHiELD ${release:-(default release)}"
        module purge
        python "$PYUFS_DIR/tools/build_shield.py" ${release:+--release "$release"}
    done
fi
# Step 3: UFS_UTILS (chgres_cube, orog, make_hgrid, ...) for native preprocessing.
if (( DO_UFS )); then
    log "INFO - building UFS_UTILS"
    python "$PYUFS_DIR/tools/build_ufs.py" ${UFS_ARGS[@]+"${UFS_ARGS[@]}"}
fi
# Step 4: static fix data (orography, climatologies) used by the preprocessing.
if (( DO_FIX )); then
    module purge
    log "INFO - updating the fix data"
    python "$PYUFS_DIR/tools/update_fix.py"
fi

log "INFO - done; set shield_exe (with its shield_release) and ufs_bin_dir in run_config.yaml to the paths printed above"