from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from chgres_cube import run_chgres_cube
from cpld_gridgen import gridgen_res, run_cpld_gridgen
from fv3_driver_grid import run_driver
from fv3_ensemble_driver import ensemble_config
from fv3_external_ic import init_external_ic
from fv3_ic_data import preprocess_only
from fv3_namelists import update_nml_configs
from fv3_nesting import get_centers, nest_info, validate_nests
from fv3_paths import configure_directories, paths
from fv3_pes_config import calc_cpu_alloc
from fv3_plot_grid import plot_grid
from fv3_runscripts import gen_shield_run_sh
from fv3_runtime import log, merged_run_config, to_list
from fv3_state import compute_checksum, save_fv3_state, state
from fv3_utils import (
    cres_to_deg,
    format_forecast_length,
    parse_datetime,
    parse_resolution,
    require_minimum_cpus,
    runtime_env_vars,
    segment_hours,
)
from global_cycle_driver import GLOBAL_GTYPES, run_surface_cycle
from regional_bc import BC_INTERVAL_HOURS
from sm_perturbations import apply_sm_perturbations
from tgrad_perturbations import (
    apply_tgrad_perturbations,
    validate_tgrad_perturbations,
)


def _log_initial_state() -> None:
    log.info("Configuration file: %s", state.run_config)
    log.info("Case directory: %s", state.case_dir)
    log.info("Current directory: %s", state.run_dir)
    log.info("Working directory: %s", state.work_dir)
    log.info("Archive directory: %s", state.archive_dir)
    log.info("Fixed/static directory: %s", state.fix_src)

    log_path = str(state.logs).replace(str(state.work_dir), str(state.case_dir))
    log.info("Logs directory: %s", log_path)

    if state.shield_exe:
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
    if state.forecast_hour:
        log.info(
            "Initial conditions: f%03d forecast of the %s UTC cycle",
            state.forecast_hour,
            state.ic_cycle,
        )

    if state.resubmit > 0:
        log.info("Total run segments: %s", state.total_restarts)

    log.info(
        "Forecast length for each segment: %s %s",
        state.run_length,
        state.run_length_units,
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


def validate_ufs_utils_options() -> None:
    """Check the global_cycle, emcsfc and cpld_gridgen settings (section 5a)."""
    for flag, needs in (
        ("run_emcsfc_snow", ("ims_snow_file",)),
        ("run_emcsfc_ice_blend", ("ims_ice_file", "five_min_ice_file")),
    ):
        if not state[flag]:
            continue
        if not state.run_global_cycle:
            raise ValueError(
                f"{flag} builds a global_cycle input; set run_global_cycle"
            )
        for key in needs:
            if not state[key]:
                raise ValueError(f"{flag} requires {key}")
    for key in (
        "global_cycle_sst_file",
        "global_cycle_ice_file",
        "global_cycle_snow_file",
        "ims_snow_file",
        "ims_ice_file",
        "five_min_ice_file",
    ):
        if state[key] and not Path(os.path.expandvars(str(state[key]))).exists():
            raise FileNotFoundError(f"{key} not found: {state[key]}")
    for flag in ("run_global_cycle", "run_cpld_gridgen"):
        if state[flag] and state.gtype not in GLOBAL_GTYPES:
            raise ValueError(f"{flag} supports gtype {', '.join(GLOBAL_GTYPES)}")
    if state.run_cpld_gridgen:
        state.cpld_gridgen_res = gridgen_res(state.cpld_gridgen_res)


def _load_initial_state() -> None:
    require_minimum_cpus()

    runtime_env = runtime_env_vars()
    runtime_config_path = Path(paths["run_dir"]) / "run_config.yaml"
    merged_config = merged_run_config()

    state.clear()
    state.update(merged_config)
    # Checked by every restart segment, so edits to the grid or start-time
    # keys of run_config.yaml after the cold start are caught.
    state.config_checksum = compute_checksum(merged_config)
    state.update(runtime_env)
    for key in ("init_datetime", "run_length"):
        if state[key] is None:
            raise ValueError(f"run_config.yaml must set {key}")
    state.case_description = state.get("description", "")
    description = [state.init_datetime, state.case_name]
    state.description = "_".join(str(value).upper() for value in description if value)
    # init_datetime names the source cycle; the ICs are its forecast_hour
    # forecast, so the model clock (coupler_nml current_date, the chgres_cube
    # date and the regional boundary hours) starts forecast_hour hours later.
    state.forecast_hour = int(state.forecast_hour or 0)
    if state.forecast_hour < 0:
        raise ValueError(f"forecast_hour must be >= 0, got {state.forecast_hour}")
    state.ic_cycle = parse_datetime(state.init_datetime)
    state.init_datetime = state.ic_cycle + pd.Timedelta(hours=state.forecast_hour)
    state.run_config = runtime_config_path
    state.c_res = parse_resolution(state.c_res)
    # Fail before grid and IC generation; files are written after preprocessing.
    validate_tgrad_perturbations(state.tgrad_perturbations)
    if state.generate_ic_data:
        validate_ufs_utils_options()
    state.continue_run = False
    state.warm_start = False
    state.restart_no = 0
    state.resubmit_idx = 0
    state.total_restarts = state.resubmit + 1

    state.total_run_hours = sum(
        segment_hours(
            state.init_datetime,
            state.run_length,
            state.run_length_units,
            state.total_restarts,
        )
    )
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
        # With external ICs the nest layout comes from the bundle
        # (fv3_external_ic), so the case's own boxes are not used.
        if state.generate_ic_data:
            validate_nests(state)
    else:
        state.n_nests = 0
        # A regional_gfdl domain is refined from its parent tile; other
        # non-nested grids have no refinement.
        state.refine_ratio = (
            int(state.refine_ratio[0]) if state.gtype == "regional_gfdl" else 1
        )

        # The model reads the boundary grid and orography under the fixed names
        # grid.tile7.halo4.nc and oro_data.tile7.halo4.nc (fv_regional_bc.F90),
        # which a halo of 3 rows produces.
        if state.gtype in ("regional_gfdl", "regional_esg") and state.halo != 3:
            raise ValueError(f"Regional grids require halo: 3, got {state.halo}")

        # A regional restart reads the boundary file of its start hour
        # (fv_regional_bc.F90: bc_hour = nint(current_time / 3600)), and
        # boundary files exist every BC_INTERVAL_HOURS.
        if state.gtype in ("regional_gfdl", "regional_esg"):
            hours = segment_hours(
                state.init_datetime,
                state.run_length,
                state.run_length_units,
                state.total_restarts,
            )
            if state.forecast_hour % BC_INTERVAL_HOURS:
                raise ValueError(
                    "Regional runs need forecast_hour to be a multiple of "
                    + f"{BC_INTERVAL_HOURS} h (boundary file interval); "
                    + f"got {state.forecast_hour}"
                )
            starts = [sum(hours[:i]) for i in range(1, len(hours))]
            if any(h % BC_INTERVAL_HOURS for h in starts):
                raise ValueError(
                    "Regional restart segments must start on multiples of "
                    + f"{BC_INTERVAL_HOURS} h (boundary file interval); "
                    + f"segments start at {starts} h"
                )

        if state.gtype == "regional_gfdl" and state.generate_ic_data:
            # A regional_gfdl domain is carved from tile 6 like a single
            # length-1 nest. The bounding box is coerced to length-1 lists so
            # get_nest_indices consumes it as nested runs do, and the cube is
            # centred on it as for nests, so the box lies inside tile 6. The
            # domain is standalone in the model, so n_nests stays 0 and no
            # fv_nest_nml is written. regional_esg is defined by idim, jdim,
            # delx and dely and does not use the box.
            for key in ("lon_min", "lon_max", "lat_min", "lat_max"):
                state[key] = to_list(state[key])
            bounds = state.lon_min + state.lon_max + state.lat_min + state.lat_max
            if len(bounds) != 4 or any(v is None for v in bounds):
                raise ValueError(
                    "regional_gfdl requires one bounding box: set lon_min, lon_max, "
                    + "lat_min and lat_max in run_config.yaml."
                )
            get_centers(state)

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
        run_surface_cycle()
        run_cpld_gridgen()
        ensemble_config()
        plot_grid()

        log.info("Finished generating grid and IC files")
        if state.preprocess_only:
            preprocess_only()
            save_fv3_state()
            return

    state.checksum = compute_checksum(state)
    os.chdir(state.work_dir)
    calc_cpu_alloc(state.grid)
    apply_sm_perturbations()
    apply_tgrad_perturbations()
    update_nml_configs()
    gen_shield_run_sh()

    save_fv3_state()

    log.info("Starting initial run")
