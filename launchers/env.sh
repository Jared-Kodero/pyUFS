#!/bin/bash -l
# env.sh - run a command in the pyufs conda env with the host modules loaded.
# Usage: env.sh COMMAND [ARGS...]
# A separate process, so the conda PATH and the modules never reach the SHiELD step.
set -e

export CONDA_PREFIX="$(dirname "$(dirname "$PYUFS_PYTHON")")"
export PATH="$CONDA_PREFIX/bin:$PATH"
for f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do
    [ -f "$f" ] && source "$f"
done
for m in $CASE_MODULES; do module load "$m"; done

exec "$@"