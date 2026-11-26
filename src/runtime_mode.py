"""Execution mode of a case: native executables or containers.

SHiELD and UFS_UTILS (preprocessing and regridding) each run either from a
native build or from a container image. The mode is chosen only by the
``shield_use_container`` and ``preprocess_use_container`` switches of
run_config.yaml. A path key (``shield_exe``, ``ufs_bin_dir``) is required and
checked only when its switch is false, and is ignored when it is true.

This module imports nothing from the workflow so that case_submit.py and the
tests can use it without a case directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def _switch(cfg: Mapping, key: str) -> bool:
    """Return a boolean switch, rejecting strings such as "false" that are truthy."""
    value = cfg.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false, got {value!r}")
    return value


def resolve_runtime(cfg: Mapping, n_nodes: int) -> dict:
    """Resolve and check the execution mode of a merged case configuration.

    Returns the two switches plus ``shield_exe`` and ``ufs_bin_dir`` with
    environment variables expanded. A path is returned empty in container mode.
    Raises ``ValueError`` with an actionable message when the combination
    cannot run: a missing or nonexistent native path, a non-boolean switch, or
    the SHiELD container on more than one node (the image is single node only).
    """
    shield_container = _switch(cfg, "shield_use_container")
    prepro_container = _switch(cfg, "preprocess_use_container")

    shield_exe = ""
    if shield_container:
        if n_nodes > 1:
            raise ValueError(
                "shield_use_container: true runs on a single node. Set n_nodes: 1, "
                + "or set shield_use_container: false and a native shield_exe."
            )
    else:
        shield_exe = os.path.expandvars(str(cfg.get("shield_exe") or ""))
        if not shield_exe:
            raise ValueError(
                "shield_use_container is false: set shield_exe to the native "
                + "SHiELD executable, or set shield_use_container: true."
            )
        if not Path(shield_exe).is_file():
            raise ValueError(f"shield_exe not found: {shield_exe}")

    ufs_bin_dir = ""
    if not prepro_container:
        ufs_bin_dir = os.path.expandvars(str(cfg.get("ufs_bin_dir") or ""))
        if not ufs_bin_dir:
            raise ValueError(
                "preprocess_use_container is false: set ufs_bin_dir to the native "
                + "UFS_UTILS directory, or set preprocess_use_container: true."
            )
        if not Path(ufs_bin_dir).is_dir():
            raise ValueError(f"ufs_bin_dir not found: {ufs_bin_dir}")

    return {
        "shield_use_container": shield_container,
        "preprocess_use_container": prepro_container,
        "shield_exe": shield_exe,
        "ufs_bin_dir": ufs_bin_dir,
    }
