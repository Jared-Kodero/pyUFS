from __future__ import annotations

from fv3_fixed_files import update_fixed_files
from fv3_namelists import restart_config, update_table_files
from fv3_paths import configure_directories
from fv3_runscripts import gen_shield_run_sh
from fv3_runtime import merged_run_config
from fv3_state import (
    compute_checksum,
    load_fv3_state,
    log,
    save_fv3_state,
    state,
)
from fv3_utils import env_setup, require_minimum_cpus, runtime_env_vars
from regional_bc import link_bc_to_input
from sm_perturbations import apply_sm_perturbations


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

    # Both checks run before configure_directories moves INPUT and RESTART,
    # so a refused restart leaves the case as the previous segment wrote it.
    current_checksum = compute_checksum(state)
    if persisted_checksum != current_checksum:
        raise RuntimeError("state.yaml does not match its recorded checksum.")

    if state.config_checksum is None:
        log.warning(
            "state.yaml has no run_config checksum (written by an earlier "
            + "version); run_config.yaml edits are not checked"
        )
    elif compute_checksum(merged_run_config()) != state.config_checksum:
        raise RuntimeError(
            "The grid or start-time settings of run_config.yaml changed after the "
            + "cold start (c_res, gtype, levels, target_lon/lat, stretch_factor, "
            + "refine_ratio, nest boxes, init_datetime, forecast_hour or "
            + "tgrad_perturbations). "
            + "Restore them or start a new case."
        )

    state.update(configure_directories(state))
    state.checksum = current_checksum

    log.info("Restart = %s", state.restart_no)


def restart_driver() -> None:
    _load_restart_state()
    env_setup()

    for file in state.work_dir.glob("*.out"):
        file.unlink()

    restart_config()
    update_fixed_files()
    update_table_files()

    # Promoting RESTART to INPUT drops the previous segment's boundary links, so
    # relink the full sequence from state.bc_data (no-op for non-regional grids).
    link_bc_to_input()

    apply_sm_perturbations()

    gen_shield_run_sh()
    save_fv3_state()
