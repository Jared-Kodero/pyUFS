from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

import fv3_plugins
from fv3_diag_files import update_table_files
from fv3_fixed_files import update_fixed_files
from fv3_namelists import restart_config
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
from fv3_utils import (
    env_setup,
    get_allocated_nodes,
    require_minimum_cpus,
    runtime_env_vars,
)
from regional_bc import link_bc_to_input


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
            + "the hashed options of a plugin (HASH_KEYS)). "
            + "Restore them or start a new case."
        )

    state.update(configure_directories(state))
    state.checksum = current_checksum
    state.node_list[state.restart_no] = get_allocated_nodes()


def restart_driver() -> None:
    _load_restart_state()
    env_setup()

    if state.resume_run:  # set by case_continue.sh for the segment it resumes
        log.info(
            f"Resuming {state.case_name} at segment {state.restart_no} "
            f"({state.total_restarts} segments in total)"
        )

    for file in state.case_dir.glob("*.out"):
        file.unlink()

    restart_config()
    update_fixed_files()
    update_table_files()

    # Promoting RESTART to INPUT drops the previous segment's boundary links, so
    # relink the full sequence from state.bc_data (no-op for non-regional grids).
    link_bc_to_input()

    gen_shield_run_sh()

    fv3_plugins.do_plugins(restart=True)
    state.resume_run = False  # applies to this segment only
    save_fv3_state()


def verify_restart_config(state_file: Path) -> None:
    """Refuse a restart whose grid or start-time settings differ from the archived case."""
    with open(state_file) as f:
        previous = yaml.safe_load(f).get("config_checksum")
    if previous is not None and previous != state.config_checksum:
        raise RuntimeError(
            f"run_config.yaml differs from the archived case ({state_file}) in the "
            + "grid or start-time settings (c_res, gtype, levels, target_lon/lat, "
            + "stretch_factor, refine_ratio, nest boxes, init_datetime, forecast_hour "
            + "or the pamip keys). Restore them."
        )


def stage_restart(restart_dir: Path, state_file: Path) -> None:
    """Continue the archived case in the freshly preprocessed case directory.

    The preprocess has rebuilt the grid, orography and initial conditions. The
    archived state.yaml is installed with the values that follow from the
    current run_config.yaml (resubmit and the run length) and the restart files
    are copied to RESTART, which the restart driver promotes to INPUT.
    """
    restart_files = Path(restart_dir) / "RESTART"
    if not (restart_files / "coupler.res").exists():
        raise FileNotFoundError(f"{restart_files / 'coupler.res'} not found")

    with open(state_file) as f:
        data = yaml.safe_load(f)
    for key in ("resubmit", "total_run_hours", "forecast_length"):
        data[key] = state[key]
    with open(Path(state.case_dir) / "state.yaml", "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)

    shutil.copytree(restart_files, state.restarts, dirs_exist_ok=True)

    # From the second restart the driver relinks the static inputs from IC/INPUT,
    # the cold-start INPUT, which it would otherwise archive as IC/R<n>_INPUT.
    if int(os.environ["CASE_RESUBMIT_INDEX"]) >= 2:
        shutil.move(state.input, Path(state.ic_data) / "INPUT")
        Path(state.input).mkdir()
