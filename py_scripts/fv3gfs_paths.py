import os
import shutil
from pathlib import Path

env_paths = {}
env_paths["fix"] = Path(os.getenv("FIX_DIR"))
env_paths["home"] = Path(os.getenv("WORK_DIR"))
env_paths["fix_am"] = env_paths["fix"] / "am"
env_paths["ufs_exe"] = Path("/UFS_UTILS/exec")
env_paths["rundir"] = Path(os.getenv("RUN_DIR"))
env_paths["case_dir"] = Path(os.getenv("CASE_DIR"))
env_paths["archive_dir"] = Path(os.getenv("ARCHIVE_DIR"))


case_paths = {}
case_paths["tmp"] = env_paths["home"] / "TMP"
case_paths["hist"] = env_paths["home"] / "HIST"
case_paths["grid"] = env_paths["home"] / "GRID"
case_paths["logs"] = env_paths["home"] / "LOGS"
case_paths["fixed"] = env_paths["home"] / "FIXED"
case_paths["input"] = env_paths["home"] / "INPUT"
case_paths["output"] = env_paths["home"] / "OUTPUT"
case_paths["restarts"] = env_paths["home"] / "RESTART"
case_paths["init_data"] = env_paths["home"] / "INIT_DATA"

paths = {**env_paths, **case_paths}


def configure_directories(params) -> dict:
    config_restart_dir({**env_paths, **case_paths}, params)

    def _clear(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if not (params["warm_start"]):
        for item in case_paths["home"].iterdir():
            _clear(item)

    elif params.update_nml_only:
        if params.warm_start:
            keys = ["restarts"]
        else:
            keys = ["hist", "logs", "output", "restarts"]

        for key in keys:
            _clear(case_paths[key])

    for k, d in case_paths.items():
        d.mkdir(parents=True, exist_ok=True)

    return {**env_paths, **case_paths}


def config_restart_dir(paths: dict, params: dict) -> None:
    """
    Archive the previous INPUT directory and promote RESTART to INPUT
    for warm-start continuation runs.

    Archive naming convention:
    - restart_no == 1  -> INIT_DATA/INIT_INPUT
    - restart_no >= 2  -> INIT_DATA/RXX_INPUT, where XXX = restart_no - 1
    """

    if not params.get("warm_start") or int(params.get("restart_no", 0)) == 0:
        return

    home = Path(paths["home"])
    archive_dir = home / "INIT_DATA"
    archive_dir.mkdir(parents=True, exist_ok=True)

    prev_input_data = Path(paths["input"])
    prev_model_restart = Path(paths["restarts"])
    curr_input_data = home / "INPUT"

    restart_no = int(params["restart_no"])
    archive_index = restart_no - 1

    if archive_index == 0:
        prev_init_data = archive_dir / "INIT_INPUT"
    else:
        prev_init_data = archive_dir / f"R{archive_index:03d}_INPUT"

    if not prev_model_restart.exists() or not any(prev_model_restart.iterdir()):
        raise FileNotFoundError(
            f"Restart directory missing or empty: {prev_model_restart}"
        )

    if prev_init_data.exists():
        raise FileExistsError(
            f"{prev_init_data} already exists; restart counter inconsistent."
        )

    if not prev_input_data.exists():
        raise FileNotFoundError(
            f"Expected INPUT directory not found: {prev_input_data}"
        )

    # Archive previous INPUT
    prev_input_data.rename(prev_init_data)

    # Promote RESTART -> INPUT
    prev_model_restart.rename(curr_input_data)

    # Re-link static (non-netCDF) files from initial archived INPUT if present
    initial_input = archive_dir / "INIT_INPUT"
    if initial_input.exists():
        for f in initial_input.iterdir():
            if f.is_file() and f.suffix != ".nc":
                target = curr_input_data / f.name

                if target.exists() or target.is_symlink():
                    target.unlink()

                rel_target = os.path.relpath(f, start=target.parent)
                target.symlink_to(rel_target)
