from fv3_namelists import restart_config, update_table_files
from fv3_runscripts import gen_shield_run_sh
from fv3_state import FV3State, compute_checksum, prev_state, save_fv3_state, state
from fv3_utils import env_setup
from sm_perturbations import apply_perturbations


def check_prev_state(params: FV3State) -> None:
    idx = state.resubmit_idx
    max_idx = state.resubmit

    if not 0 <= idx <= max_idx:
        raise ValueError(f"Invalid resubmit state: {idx=} {max_idx=}")

    checksum = compute_checksum(params)
    if idx > 0 and prev_state.get("checksum") != checksum:
        raise RuntimeError("Restart configuration does not match previous state.")

    run_hours = params.run_nhours

    if isinstance(run_hours, list):
        if len(run_hours) != max_idx + 1:
            raise ValueError(
                f"run_nhours needs {max_idx + 1} values, got {len(run_hours)}"
            )

        params.run_nhours = run_hours[idx]
        params.total_run_hours = sum(run_hours)

    else:
        params.total_run_hours = (max_idx + 1) * run_hours

    params.checksum = checksum
    params.restart_no = idx
    params.resubmit_idx = idx
    params.resubmit = max_idx
    params.total_restarts = max_idx + 1
    params.warm_start = idx > 0


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
    save_fv3_state()
    gen_shield_run_sh()
