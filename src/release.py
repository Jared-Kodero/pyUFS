"""SHiELD release selection and the release-dependent settings.

``shield_release`` in a case ``run_config.yaml`` (one of VALID_RELEASES) picks the namelist templates
(``templates/<release>.<name>``) and the ``coupler_nml`` time-step key. The defaults
for all other keys are in ``config/run_config.yaml``, which is the same for every
release. This module is the only place that reads and validates the value.
"""

from __future__ import annotations

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
