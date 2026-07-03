from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fv3_state import FV3State

env_paths = {}
env_paths["work_dir"] = Path(os.getenv("WORK_DIR"))
env_paths["fix_src"] = Path(os.getenv("FIX_SRC"))
env_paths["ufs_exe"] = Path("/UFS_UTILS/exec")
env_paths["run_dir"] = Path(os.getenv("CASE_PWD"))
env_paths["case_dir"] = Path(os.getenv("CASE_DIR"))
env_paths["archive_dir"] = Path(os.getenv("ARCHIVE_DIR"))


case_paths = {}
case_paths["tmp"] = env_paths["work_dir"] / "TMP"
case_paths["hist"] = env_paths["work_dir"] / "HIST"
case_paths["grid"] = env_paths["work_dir"] / "GRID"
case_paths["logs"] = env_paths["work_dir"] / "LOGS"
case_paths["fix"] = env_paths["work_dir"] / "FIXED"
case_paths["input"] = env_paths["work_dir"] / "INPUT"
case_paths["output"] = env_paths["work_dir"] / "OUTPUT"
case_paths["restarts"] = env_paths["work_dir"] / "RESTART"
case_paths["ic_data"] = env_paths["work_dir"] / "IC"

paths = {**env_paths, **case_paths}


def configure_directories(params: FV3State) -> dict:
    params = parse_dirs(params)

    config_restart_dir({**env_paths, **case_paths}, params)

    def _clear(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if params.warm_start:
        _clear(paths["restarts"])
        _clear(paths["hist"])

    else:
        _clear(paths["output"])
        _clear(paths["hist"])

    for _, d in case_paths.items():
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

    work_dir = Path(paths["work_dir"])
    archive_dir = Path(paths["ic_data"])
    archive_dir.mkdir(parents=True, exist_ok=True)

    prev_input_data = Path(paths["input"])
    prev_model_restart = Path(paths["restarts"])
    curr_input_data = work_dir / "INPUT"

    restart_no = int(params.restart_no)
    archive_index = restart_no - 1

    if archive_index == 0:
        prev_ic_data = archive_dir / "INPUT"
    else:
        prev_ic_data = archive_dir / f"R{archive_index:03d}_INPUT"

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

    # Re-link static (non-netCDF) files from initial archived INPUT if present
    initial_input = archive_dir / "INPUT"
    if initial_input.exists():
        for f in initial_input.iterdir():
            if f.is_file() and f.suffix != ".nc":
                target = curr_input_data / f.name

                if target.exists() or target.is_symlink():
                    target.unlink()

                rel_target = os.path.relpath(f, start=target.parent)
                target.symlink_to(rel_target)


def parse_dirs(cfg: dict) -> dict:

    dir_keys = (
        "jobtmp",
        "case_root",
        "fix_src",
        "ufs_utils",
        "archive_root",
        "shield_image",
        "fregrid_image",
        "preprocess_image",
        "containers_root",
    )

    for k in dir_keys:
        cfg[k] = str(Path(os.path.expandvars(cfg[k])))
    return cfg
