import hashlib
import os

from fv3gfs_namelists import restart_config
from fv3gfs_runscripts import gen_shield_run_sh
from fv3gfs_stage_data import update_table_files
from fv3gfs_state import prev_state, save_state, state
from fv3gfs_utils import env_setup
from sm_pertubutions import apply_perturbations


def check_prev_state(params: dict) -> None:

    _hash_keys = [
        "res",
        "gtype",
        "levels",
        "target_lon",
        "target_lat",
        "stretch_factor",
        "refine_ratio",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "datetime",
        "chgres_config",
    ]

    # ------------------------------------------------------------
    # Compute checksum
    # ------------------------------------------------------------

    _hash_data_str = ",".join(str(params.get(k)) for k in _hash_keys)
    checksum = hashlib.sha256(_hash_data_str.encode("utf-8")).hexdigest()[32:].upper()
    params["checksum"] = checksum

    run_hours = params["run_nhours"]

    # ------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------

    if not prev_state:
        params["restart_no"] = 0
        params["update_nml_only"] = False

        resubmit = int(os.getenv("RESUBMIT_COUNT", 0))
        params["resubmit"] = resubmit
        params["total_restarts"] = resubmit + 1

        if isinstance(run_hours, list):
            params["run_nhours"] = run_hours[0]

        if isinstance(run_hours, int):
            total_run_hours = (resubmit + 1) * run_hours
        else:
            total_run_hours = sum(run_hours)

        params["total_run_hours"] = total_run_hours

        return

    # ------------------------------------------------------------
    # Warm start continuation
    # ------------------------------------------------------------

    if params.get("warm_start", False):
        params["update_nml_only"] = True

        prev_restart = prev_state.get("restart_no", 0)
        restart_no = prev_restart + 1
        params["restart_no"] = restart_no

        if isinstance(run_hours, list):
            idx = min(restart_no, len(run_hours) - 1)
            params["run_nhours"] = run_hours[idx]
        else:
            params["run_nhours"] = run_hours

        prev_resubmit = prev_state.get("resubmit", 0)
        params["resubmit"] = max(prev_resubmit - 1, 0)

        return

    # ------------------------------------------------------------
    # Non warm start continuation
    # ------------------------------------------------------------

    params["restart_no"] = 0

    if isinstance(run_hours, list):
        params["run_nhours"] = run_hours[0]

    ic_and_grid = (
        bool(prev_state.get("ic_and_grid_generated", False))
        and prev_state.get("checksum") == checksum
    )

    params["update_nml_only"] = bool(ic_and_grid)


def restart_driver():
    env_setup()

    for k, v in state.items():
        prev_state[k] = v
    state.update(dict(prev_state))

    for file in state.home.glob("*"):
        if str(file).endswith((".out")):
            file.unlink()

    restart_config()
    update_table_files()
    apply_perturbations()
    save_state()
    gen_shield_run_sh()
