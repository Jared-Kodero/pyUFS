"""SHiELD release selection and the release-dependent settings.

``shield_release`` in a case ``run_config.yaml`` picks the default configuration
(``config/<release>/run_config.yaml``), the namelist templates
(``templates/<release>/``) and the ``coupler_nml`` time-step key. This module is
the only place that reads and validates the value.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_RELEASE = "202411"

# The release the shield container image is built from (gfdlfv3/shield). A
# native executable is required for any other release.
CONTAINER_RELEASE = "202411"


def available_releases() -> list[str]:
    """Releases that have a default configuration in config/<release>/."""
    return sorted(p.parent.name for p in CONFIG_DIR.glob("*/run_config.yaml"))


def resolve_release(value: object) -> str:
    """Return the release name for a ``shield_release`` value.

    The value is converted to a string, so an unquoted ``202604`` in YAML (an
    integer) is accepted. An unset or empty value gives the default release.
    Raises ValueError for a release without a configuration directory.
    """
    release = "" if value is None else str(value).strip()
    if not release:
        return DEFAULT_RELEASE
    valid = available_releases()
    if release not in valid:
        raise ValueError(
            f"shield_release {release!r} is not supported; choose one of: "
            + ", ".join(valid)
        )
    return release


def case_release(case_file: Path) -> str:
    """Release named by a case file, or the default release when it names none.

    A file that cannot be parsed gives the default release; the YAML error is
    reported where the file is read for its keys.
    """
    try:
        case = yaml.safe_load(case_file.read_text()) or {}
    except yaml.YAMLError:
        return DEFAULT_RELEASE
    if not isinstance(case, dict):
        return DEFAULT_RELEASE
    return resolve_release(case.get("shield_release"))


def template(release: str, name: str) -> Path:
    return ROOT_DIR / "templates" / resolve_release(release) / name


def coupling_step_key(release: str) -> str:
    if resolve_release(release) == "202411":
        return "dt_ocean"
    return "dt_cpld"


def restart_overrides(release: str) -> dict[str, dict]:
    if resolve_release(release) == "202411":
        keys = {"checksum_required": False, "restart_checksums_required": False}
        return {"fms_io_nml": keys, "fms2_io_nml": keys}
    return {}


def resolve_shield_exe(
    shield_exe: object, release: str, multi_node: bool
) -> tuple[str, str | None]:
    """Decide how the model is run: a native executable or the container.

    Returns ``(exe, message)``. ``exe`` is the executable path with variables
    expanded, or "" to use the container image; ``message`` is a warning for the
    caller to log, or None.

    * ``shield_exe`` empty: the container, no message.
    * ``shield_exe`` is a file: used as given.
    * ``shield_exe`` is missing on a single node and the release is the one the
      container is built from: the container is used, with a message.

    Raises FileNotFoundError when the executable is missing and the container is
    not an option: a multi-node run (the container image is single-node only) or
    a release the container image does not match.
    """
    raw = "" if shield_exe is None else str(shield_exe).strip()
    if not raw:
        return "", None

    exe = os.path.expandvars(raw)
    if Path(exe).is_file():
        return exe, None

    build = "Build it with tools/build_shield.py (README section 19)"
    if multi_node:
        raise FileNotFoundError(
            f"shield_exe not found: {exe}. Multi-node runs need a native "
            f'executable. {build}, or set shield_exe: "" to use the container '
            "image on a single node."
        )
    release = resolve_release(release)
    if release != CONTAINER_RELEASE:
        raise FileNotFoundError(
            f"shield_exe not found: {exe}. The container image is the "
            f"{CONTAINER_RELEASE} build and cannot run shield_release {release}. "
            f"{build}."
        )
    return "", (
        f"shield_exe not found: {exe}; the {CONTAINER_RELEASE} container image "
        "will be used instead."
    )
