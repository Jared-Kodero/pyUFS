#!/bin/bash -l

# case_continue.sh - root entrypoint for resuming a failed case. Run it from the
# case directory (the one holding run_config.yaml) after fixing the cause of the
# failure; it hands over to launchers/case_continue.py, which resumes from the
# last good restart on disk. With an ensemble, pass --ensembles (all members) or
# --mem N [N ...] (selected members); arguments go through to the launcher.
#
# Deactivate any active Conda environment for this script process, so the
# submission does not depend on it.

conda deactivate >/dev/null 2>&1

# The repository is located from this file's directory; the case itself is
# read from the current working directory.
UFS_UTILS_DIR="$(cd "$(dirname "$0")" && pwd)"

"$UFS_UTILS_DIR/launchers/case_continue.py" "$@"
