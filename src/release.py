from pathlib import Path

import yaml

DEFAULT_RELEASE = "202411"


def case_release(case_file: Path) -> str:
    try:
        case = yaml.safe_load(case_file.read_text()) or {}
    except yaml.YAMLError:
        return DEFAULT_RELEASE
    return case.get("shield_release") or DEFAULT_RELEASE


def expand_shield_exe(cfg: dict) -> dict:
    exe = (cfg["shield_exe"] or "").replace("{shield_release}", cfg["shield_release"])
    return {**cfg, "shield_exe": exe}


def template(release: str, name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "templates" / release / name


def coupling_step_key(release: str) -> str:
    if release == "202411":
        return "dt_ocean"
    return "dt_cpld"


def restart_overrides(release: str) -> dict[str, dict]:
    if release == "202411":
        keys = {"checksum_required": False, "restart_checksums_required": False}
        return {"fms_io_nml": keys, "fms2_io_nml": keys}
    return {}
