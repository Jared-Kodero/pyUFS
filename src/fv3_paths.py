from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fv3_state import FV3State


def _env_path(name: str) -> Path | None:
    """Path from an environment variable set by the launchers, or None.

    Reading at import must not fail where the job environment is absent (tools,
    tests); a missing value surfaces where the path is used.
    """
    value = os.getenv(name)
    return Path(value) if value else None


env_paths = {
    "fix_src": _env_path("FIX_SRC"),
    "ufs_exe": Path("/UFS_UTILS/exec"),
    "run_dir": _env_path("CASE_PWD"),
    "case_dir": _env_path("CASE_DIR"),
    "archive_dir": _env_path("ARCHIVE_DIR"),
    "ufs_utils": Path(__file__).resolve().parent.parent,
}
env_paths["configs"] = env_paths["ufs_utils"] / "config"

# Subdirectories of the case directory (None while CASE_DIR is unset).
CASE_SUBDIRS = {
    "tmp": "TMP",
    "hist": "HIST",
    "grid": "GRID",
    "logs": "LOGS",
    "fix": "FIXED",
    "input": "INPUT",
    "output": "OUTPUT",
    "restarts": "RESTART",
    "ic_data": "IC",
    "bc_data": "BC",
}
case_paths = {
    key: env_paths["case_dir"] / sub if env_paths["case_dir"] else None
    for key, sub in CASE_SUBDIRS.items()
}

paths = {**env_paths, **case_paths}


def configure_directories(state: FV3State) -> dict:
    state = parse_dirs(state)

    config_restart_dir(paths, state)

    def _clear(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if state.warm_start:
        # A resumed segment (case_continue.sh) clears nothing: HIST and RESTART
        # may hold the finished work of the failed segment.
        if not state.get("resume_run"):
            _clear(paths["restarts"])
            _clear(paths["hist"])

    else:
        seg_dir = Path(paths["output"]) / "seg"
        if seg_dir.is_dir() and any(seg_dir.iterdir()):
            raise RuntimeError(
                f"{seg_dir} holds regridded output of earlier segments, which a cold "
                + "start would delete. Run case_continue.sh to resume the case, or "
                + "remove the case directory to start over."
            )
        _clear(paths["output"])
        _clear(paths["hist"])
        _clear(paths["restarts"])

    if int(state.get("restart_no", 0)) == 0:
        # A cold start writes INPUT afresh: from the generated ICs, or from an
        # external bundle copied in by fv3_external_ic. Without either, the
        # case's own INPUT holds the ICs and is kept.
        external = state.get("external_ic_dir")
        in_place = not state.generate_ic_data and (
            not external
            or Path(external).resolve() == Path(paths["case_dir"]).resolve()
        )
        if not in_place:
            _clear(paths["input"])
            _clear(paths["ic_data"] / "INPUT")
            for archived_input in paths["ic_data"].glob("R*_INPUT"):
                _clear(archived_input)
        elif (paths["input"] / "coupler.res").exists():
            # INPUT then holds a promoted restart: the coupler would start the
            # clock at its date while the atmosphere is cold started.
            raise RuntimeError(
                f"{paths['input']} holds a model restart (coupler.res) from earlier "
                + "segments; the cold-start inputs are archived in IC/INPUT. Restore "
                + "them to INPUT, set external_ic_dir, or set generate_ic_data: true."
            )

    for d in case_paths.values():
        d.mkdir(parents=True, exist_ok=True)

    return paths


def config_restart_dir(paths: dict, params: FV3State) -> None:
    """
    Archive the previous INPUT directory and promote RESTART to INPUT
    for warm-start continuation runs.

    Archive naming convention:
    - restart_no == 1  -> IC/INPUT
    - restart_no >= 2  -> IC/RXX_INPUT, where XXX = restart_no - 1
    """

    if not params.get("warm_start") or int(params.get("restart_no", 0)) == 0:
        return

    case_dir = Path(paths["case_dir"])
    archive_dir = Path(paths["ic_data"])
    archive_dir.mkdir(parents=True, exist_ok=True)

    prev_input_data = Path(paths["input"])
    prev_model_restart = Path(paths["restarts"])
    curr_input_data = case_dir / "INPUT"

    restart_no = int(params.restart_no)
    archive_index = restart_no - 1

    if archive_index == 0:
        prev_ic_data = archive_dir / "INPUT"
    else:
        prev_ic_data = archive_dir / f"R{archive_index:03d}_INPUT"

    # A resumed segment (case_continue.sh) whose failed attempt already promoted
    # its restart: INPUT holds it, and IC/R<n>_INPUT the input it replaced.
    if (
        params.get("resume_run")
        and prev_ic_data.exists()
        and (curr_input_data / "coupler.res").exists()
    ):
        return

    if not prev_model_restart.exists() or not any(prev_model_restart.iterdir()):
        raise FileNotFoundError(
            f"Restart directory missing or empty: {prev_model_restart}"
        )

    if prev_ic_data.exists():
        raise FileExistsError(
            f"{prev_ic_data} already exists; restart counter inconsistent."
        )

    if not prev_input_data.exists():
        raise FileNotFoundError(
            f"Expected INPUT directory not found: {prev_input_data}"
        )

    # Archive previous INPUT
    prev_input_data.rename(prev_ic_data)

    # Promote RESTART -> INPUT
    prev_model_restart.rename(curr_input_data)

    relink_static_inputs(archive_dir / "INPUT", curr_input_data)


# Cold-start inputs that a warm start must not reuse: the initial conditions
# (replaced by RESTART) and the regional boundary files (relinked from
# state.bc_data by regional_bc.link_bc_to_input).
COLD_START_ONLY = ("gfs_data", "sfc_data", "gfs_bndy")


def relink_static_inputs(initial_input: Path, input_dir: Path) -> None:
    """Link the static inputs of the cold-start INPUT into a promoted INPUT.

    RESTART holds only the model state. The files the model also reads at a
    warm start (oro_data*, the GSL oro_data_ls/ss tiles, grid and mosaic
    links, gfs_ctrl.nc for regional boundaries, fix-file links) live in the
    cold-start INPUT, archived as IC/INPUT. Without them the physics sets the
    orographic fields to zero (FV3GFS_io.F90, sfc_prop_restart_read). Files
    already provided by RESTART are kept.

    Symbolic links are copied verbatim: INPUT and IC/INPUT links are relative
    to the original INPUT directory, which the promoted INPUT replaces.
    Regular files are linked to their archived copy.
    """
    if not initial_input.exists():
        return

    for f in sorted(initial_input.iterdir()):
        if f.name.startswith(COLD_START_ONLY):
            continue

        target = input_dir / f.name
        if target.exists() or target.is_symlink():
            continue

        if f.is_symlink():
            target.symlink_to(os.readlink(f))
        else:
            target.symlink_to(os.path.relpath(f, start=input_dir))


def parse_dirs(cfg: dict) -> dict:

    dir_keys = (
        "case_root",
        "fix_src",
        "shield_image",
        "fregrid_image",
        "preprocess_image",
        "containers_root",
        "archive_root",
    )

    for k in dir_keys:
        if cfg[k] is None:
            continue
        cfg[k] = str(Path(os.path.expandvars(cfg[k])))
    return cfg
