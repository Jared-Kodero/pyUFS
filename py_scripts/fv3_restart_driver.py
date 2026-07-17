from __future__ import annotations

from fv3_namelists import restart_config, update_table_files
from fv3_paths import configure_directories
from fv3_runscripts import gen_shield_run_sh
from fv3_state import (
    compute_checksum,
    load_fv3_state,
    log,
    save_fv3_state,
    state,
)
from fv3_utils import (
    env_setup,
    require_minimum_cpus,
    runtime_env_vars,
)
from sm_perturbations import apply_perturbations


def _load_restart_state() -> None:
    require_minimum_cpus()

    runtime_env = runtime_env_vars()
    restart_index = int(runtime_env["resubmit_idx"])

    if restart_index <= 0:
        raise RuntimeError(
            "Restart driver requires CASE_RESUBMIT_INDEX to be greater than zero."
        )

    load_fv3_state()

    persisted_checksum = state.get("checksum")
    if not persisted_checksum:
        raise RuntimeError("state.yaml does not contain a configuration checksum.")

    state.update(runtime_env)

    state.restart_no = restart_index
    state.resubmit_idx = restart_index
    state.total_restarts = state.resubmit + 1
    state.continue_run = True
    state.warm_start = True

    if not 0 <= state.resubmit_idx <= state.resubmit:
        raise ValueError(
            f"Invalid resubmit state: {state.resubmit_idx=} {state.resubmit=}"
        )

    state.update(configure_directories(state))

    current_checksum = compute_checksum(state)

    if persisted_checksum != current_checksum:
        raise RuntimeError(
            "Restart configuration does not match the initial configuration."
        )

    state.checksum = current_checksum

    log.info("Restart = %s", state.restart_no)


def restart_driver() -> None:
    _load_restart_state()
    env_setup()

    for file in state.work_dir.glob("*.out"):
        file.unlink()

    restart_config()
    update_table_files()
    apply_perturbations()

    gen_shield_run_sh()
    save_fv3_state()
