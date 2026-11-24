"""Helpers shared by tools/build_shield.py and tools/build_ufs.py."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD_ROOT = Path("/tmp") / (os.environ.get("USER") or Path.home().name)

# The build scripts read the release configuration through src/.
sys.path.insert(0, str(REPO / "src"))
from fv3_yaml import load_config
from release import CONFIG_DIR


class BuildError(RuntimeError):
    """A failed build step."""


def release_config(release: str) -> dict:
    """Default configuration of a release: config/<release>/run_config.yaml and its includes."""
    config = load_config(CONFIG_DIR / release / "run_config.yaml")
    modules = config.get("modules")
    if not modules or not isinstance(modules, list):
        raise BuildError(f"release {release}: 'modules' must be a non-empty list")
    exe = config.get("shield_exe")
    if not isinstance(exe, str) or not exe.strip():
        raise BuildError(f"release {release}: 'shield_exe' must be a non-empty path")
    return config


def module_script(modules: list[str]) -> str:
    """Shell lines that load `modules` in a fresh bash."""
    init = (
        'if [[ -z ${MODULESHOME:-} || ! -f $MODULESHOME/init/sh ]]; then\n'
        "  echo 'Missing $MODULESHOME/init/sh' >&2; exit 1\n"
        "fi\n"
        'source "$MODULESHOME/init/sh"\n'
    )
    loads = "".join(f"module load {shlex.quote(name)} || exit 1\n" for name in modules)
    return init + loads
