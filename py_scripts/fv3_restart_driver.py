import os

from fv3_namelists import restart_config, update_table_files
from fv3_runscripts import gen_shield_run_sh
from fv3_state import FV3State, compute_checksum, prev_state, save_state, state
from fv3_utils import env_setup
from sm_perturbations import apply_perturbations


def check_prev_state(params: FV3State) -> None:

    checksum = compute_checksum(params)
    params.checksum = checksum

    run_hours = params.run_nhours

    # ------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------

    if not prev_state:
        params.restart_no = 0

        resubmit = int(os.getenv("CASE_RESUBMIT_COUNT", 0))
        params.resubmit = resubmit
        params.total_restarts = resubmit + 1

        if isinstance(run_hours, list):
            params.run_nhours = run_hours[0]

        if isinstance(run_hours, int):
            total_run_hours = (resubmit + 1) * run_hours
        else:
            total_run_hours = sum(run_hours)

        params.total_run_hours = total_run_hours

        return

    # ------------------------------------------------------------
    # Warm start continuation
    # ------------------------------------------------------------

    if params.get("warm_start", False):
        prev_restart = prev_state.get("restart_no", 0)
        restart_no = prev_restart + 1
        params.restart_no = restart_no

        if isinstance(run_hours, list):
            idx = min(restart_no, len(run_hours) - 1)
            params.run_nhours = run_hours[idx]
        else:
            params.run_nhours = run_hours

        prev_resubmit = prev_state.get("resubmit", 0)
        params.resubmit = max(prev_resubmit - 1, 0)

        return

    # ------------------------------------------------------------
    # Non warm start continuation
    # ------------------------------------------------------------

    params.restart_no = 0

    if isinstance(run_hours, list):
        params.run_nhours = run_hours[0]


def restart_driver():
    env_setup()

    for k, v in state.items():
        prev_state[k] = v
    state.update(dict(prev_state))

    for file in state.case_home.glob("*"):
        if str(file).endswith((".out")):
            file.unlink()

    restart_config()
    update_table_files()
    apply_perturbations()
    save_state()
    gen_shield_run_sh()
