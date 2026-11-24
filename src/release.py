"""SHiELD release selection and the release-dependent settings.

``shield_release`` in a case ``run_config.yaml`` picks the default configuration
(``config/<release>/run_config.yaml``, which includes ``config/shared/run_config.yaml``), the namelist templates
(``templates/<release>/``) and the ``coupler_nml`` time-step key. This module is
the only place that reads and validates the value.
"""

from __future__ import annotations

from pathlib import Path

from fv3_yaml import YamlError, open_yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_RELEASE = "202411"


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
        case = open_yaml(case_file)
    except YamlError:
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
