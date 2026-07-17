from __future__ import annotations

import os
from pathlib import Path

from chgres_cube import run_chgres_cube
from fv3_driver_grid import run_driver
from fv3_ensemble_driver import ensemble_config
from fv3_external_ic import init_external_ic
from fv3_ic_data import preprocess_only
from fv3_namelists import update_nml_configs
from fv3_nesting import nest_info, validate_nests
from fv3_paths import configure_directories, paths
from fv3_pes_config import calc_cpu_alloc
from fv3_plot_grid import plot_grid
from fv3_runscripts import gen_shield_run_sh
from fv3_runtime import log, read_namelist, to_list
from fv3_state import compute_checksum, save_fv3_state, state
from fv3_utils import (
    cres_to_deg,
    format_forecast_length,
    parse_datetime,
    parse_resolution,
    require_minimum_cpus,
    runtime_env_vars,
)
from sm_perturbations import apply_perturbations


def _log_initial_state() -> None:
    log.info("Configuration file: %s", state.run_config)
    log.info("Case directory: %s", state.case_dir)
    log.info("Current directory: %s", state.run_dir)
    log.info("Working directory: %s", state.work_dir)
    log.info("Archive directory: %s", state.archive_dir)
    log.info("Fixed/static directory: %s", state.fix_src)

    log_path = str(state.logs).replace(str(state.work_dir), str(state.case_dir))
    log.info("Logs directory: %s", log_path)

    if "shield_exe" in state:
        log.info("Model executable: %s", state.shield_exe)
    else:
        log.info("Model executable: container image (SHiELD)")

    log.info("Description: %s", state.description)
    log.info("Initial run mode selected")
    log.info("Full Grid/IC regeneration will be performed.")

    if state.preprocess_only:
        log.info(
            "Preprocess-only mode selected. Will exit after preprocessing IC data."
        )

    if state.ensemble_run:
        log.info(
            "Ensemble run [%s/%s]",
            state.ensemble_id,
            state.n_ensembles,
        )

    log.info("Model initialization time: %s UTC", state.init_datetime)

    if state.resubmit > 0:
        log.info("Total run segments: %s", state.total_restarts)

    log.info(
        "Forecast length for this segment: %s hours",
        state.run_nhours,
    )
    log.info("Total forecast length: %s", state.forecast_length)
    log.info("Vertical levels: %s", state.levels)
    log.info("Grid type: %s", state.gtype)
    log.info("Global cubed-sphere resolution: C%s", state.c_res)

    for tile in range(1, 7):
        log.info(
            "Global tile %s resolution: %.2f km",
            tile,
            state.res_km[0],
        )

    if state.gtype == "nest":
        for message in nest_info:
            log.info(message)

        log.info("Number of nests: %s", state.n_nests)
        log.info("Refinement ratio: %s", state.refine_ratio)

    log.info("Target longitude: %s", state.target_lon)
    log.info("Target latitude: %s", state.target_lat)


def _load_initial_state() -> None:
    require_minimum_cpus()

    runtime_env = runtime_env_vars()
    default_config_path = Path(paths["configs"]) / "run_config.yaml"
    runtime_config_path = Path(paths["run_dir"]) / "run_config.yaml"

    default_config = read_namelist(default_config_path)
    runtime_config = read_namelist(runtime_config_path)

    merged_config = dict(runtime_config)

    for key, value in default_config.items():
        if merged_config.get(key) is None and value is not None:
            merged_config[key] = value

    state.clear()
    state.update(merged_config)
    state.update(runtime_env)

    state.run_config = runtime_config_path
    state.c_res = parse_resolution(state.c_res)

    state.k_split = to_list(state.get("k_split"))
    state.n_split = to_list(state.get("n_split"))

    state.case_description = state.get("description", "")
    state.continue_run = False
    state.warm_start = False
    state.restart_no = 0
    state.resubmit_idx = 0
    state.total_restarts = state.resubmit + 1

    parse_datetime(state)

    description = [state.init_datetime, state.case_name]
    state.description = "_".join(str(value).upper() for value in description if value)

    segment_count = state.resubmit + 1
    state.total_run_hours = state.run_nhours * segment_count
    state.forecast_length = format_forecast_length(state.total_run_hours)

    state.update(configure_directories(state))

    refine_ratio = state.refine_ratio
    state.refine_ratio = to_list(refine_ratio)

    if len(state.refine_ratio) == 1:
        state.res_km = [cres_to_deg(state.c_res).km]
    else:
        state.res_km = [0.0] * (len(state.refine_ratio) + 1)
        state.res_km[0] = cres_to_deg(state.c_res).km

    if state.gtype == "nest":
        state.n_nests = len(state.refine_ratio)
        validate_nests(state)
    else:
        state.n_nests = 0
        state.refine_ratio = 1

    state.checksum = compute_checksum(state)

    _log_initial_state()


def init_driver() -> None:
    _load_initial_state()

    os.chdir(state.work_dir)

    if not state.generate_ic_data:
        init_external_ic()
    else:
        log.info("Starting FV3 Grid and IC generation driver")

        run_driver(
            c_res=state.c_res,
            gtype=state.gtype,
            add_lake=state.add_lake,
            lake_cutoff=state.lake_cutoff,
            make_gsl_orog=state.make_gsl_orog,
            stretch_factor=state.stretch_factor,
            target_lon=state.target_lon,
            target_lat=state.target_lat,
            refine_ratio=state.refine_ratio,
            istart_nest=state.istart_nest,
            jstart_nest=state.jstart_nest,
            iend_nest=state.iend_nest,
            jend_nest=state.jend_nest,
            parent_tile=state.parent_tile,
            lon_min=state.lon_min,
            lon_max=state.lon_max,
            lat_min=state.lat_min,
            lat_max=state.lat_max,
            n_nests=state.n_nests,
            halo=state.halo,
            idim=state.idim,
            jdim=state.jdim,
            delx=state.delx,
            dely=state.dely,
            orog_dir=state.fix_src / "orog",
            tmp=state.tmp,
            exe_dir=state.ufs_exe,
            fix_dir=state.fix_src,
        )

        run_chgres_cube()
        ensemble_config()
        plot_grid()

        if state.preprocess_only:
            preprocess_only()
            save_fv3_state()
            return

    os.chdir(state.work_dir)

    calc_cpu_alloc(state.grid)
    update_nml_configs()
    apply_perturbations()
    gen_shield_run_sh()

    save_fv3_state()

    log.info("Starting initial run")
