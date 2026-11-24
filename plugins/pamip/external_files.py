"""User SST, sea-ice and land-mask files that replace the SHiELD surface climatologies.

They work without a PAMIP experiment; the pamip hooks call these functions."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("PLUGINS")

EXTERNAL_FILES = {
    "external_sst": "fntsfc",
    "external_sic": "fnaisc",
    "external_mask": "fnmskh",
}


def validate(config: dict) -> None:
    given = []
    for key in EXTERNAL_FILES:
        value = config[key]
        if value is None or value == "":
            config[key] = None
            continue
        if not isinstance(value, str):
            raise TypeError(f"`{key}` must be a file path as text. Got {value!r}")
        path = Path(os.path.expandvars(value)).expanduser()
        given.append(path.name)
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"`{key}` is {value!r}, which resolves to {path}.")

    if len(set(given)) != len(given):
        raise ValueError(
            "`external_sst`, `external_sic` and `external_mask` need three different file names."
        )


def stage(config: dict, state) -> dict[str, dict]:
    """Copy the files to FIXED/external and point namsfc at the copies."""
    namsfc = {}
    for key, entry in EXTERNAL_FILES.items():
        if not config[key]:
            continue
        path = Path(os.path.expandvars(config[key])).expanduser()
        destination = Path(state.fix) / "external" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        shutil.copyfile(path.resolve(), destination)
        namsfc[entry] = f"FIXED/external/{path.name}"
        log.info("%s: %s", key, path.resolve())
    return {"namsfc": namsfc}
