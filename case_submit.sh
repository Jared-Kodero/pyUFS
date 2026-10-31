#!/bin/bash -l

# case_submit.sh - root entrypoint. Run it from the case directory (the one
# holding run_config.yaml); it hands over to launchers/case_submit.py.
#
# Deactivate any active Conda environment for this script process, so the
# submission does not depend on it.

conda deactivate >/dev/null 2>&1

# The repository is located from this file's directory; the case itself is
# read from the current working directory.
UFS_UTILS_DIR="$(cd "$(dirname "$0")" && pwd)"

"$UFS_UTILS_DIR/launchers/case_submit.py"
