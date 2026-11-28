"""SHiELD release selection and the release-dependent settings.

``shield_release`` in a case ``run_config.yaml`` (one of VALID_RELEASES) picks the namelist templates
(``templates/<release>.<name>``) and the ``coupler_nml`` time-step key. The defaults
for all other keys are in ``config/run_config.yaml``, which is the same for every
release. This module is the only place that reads and validates the value.

``resolve_runtime`` checks the execution mode: SHiELD and UFS_UTILS each run
from a native build or a container image, chosen only by the
``shield_use_container`` and ``preprocess_use_container`` switches. A path key
(``shield_exe``, ``ufs_bin_dir``) is required and checked only when its switch is
false. The module imports nothing from the workflow, so case_submit.py can use it
without a case directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT_DIR / "templates"
# Add a release here together with its templates/<release>.*.yaml files.
VALID_RELEASES = ("202411", "202604")
DEFAULT_RELEASE = "202411"


def resolve_release(value: object) -> str:
    """Return the release name for a ``shield_release`` value.

    The value is converted to a string, so an unquoted ``202604`` in YAML (an
    integer) is accepted. An unset or empty value gives the default release.
    Raises ValueError for a release that is not in VALID_RELEASES.
    """
    release = "" if value is None else str(value).strip()
    if not release:
        return DEFAULT_RELEASE
    if release not in VALID_RELEASES:
        raise ValueError(
            f"shield_release {release!r} is not supported; choose one of: "
            + ", ".join(VALID_RELEASES)
        )
    return release


def template(release: str, name: str) -> Path:
    return TEMPLATE_DIR / f"{resolve_release(release)}.{name}"


def coupling_step_key(release: str) -> str:
    if resolve_release(release) == "202411":
        return "dt_ocean"
    return "dt_cpld"


def restart_overrides(release: str) -> dict[str, dict]:
    if resolve_release(release) == "202411":
        keys = {"checksum_required": False, "restart_checksums_required": False}
        return {"fms_io_nml": keys, "fms2_io_nml": keys}
    return {}


def _use_container(cfg: Mapping, component: str) -> bool:
    """`<component>_use_container` as a strict boolean; strings such as "false" are rejected."""
    key = f"{component}_use_container"
    value = cfg.get(key, False)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be true or false, got {value!r}")
    return value


def resolve_runtime(cfg: Mapping, n_nodes: int) -> dict:
    """Resolve and check the execution mode of a merged case configuration.

    Returns the two switches plus ``shield_exe`` and ``ufs_bin_dir`` with
    environment variables expanded. A path is returned empty in container mode.
    Raises ``ValueError`` with an actionable message when the combination
    cannot run: a missing or nonexistent native path, a non-boolean switch, or
    the SHiELD container on more than one node (the image is single node only).
    """
    shield_container = _use_container(cfg, "shield")
    prepro_container = _use_container(cfg, "preprocess")

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
