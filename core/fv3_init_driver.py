from __future__ import annotations

import os
import socket
from pathlib import Path

import fv3_plugins
import pandas as pd
from chgres_cube import run_chgres_cube
from cpld_gridgen import gridgen_res, run_cpld_gridgen
from fv3_diag_files import update_table_files
from fv3_driver_grid import run_driver
from fv3_ensemble_driver import ensemble_config
from fv3_external_ic import init_external_ic
from fv3_fixed_files import update_fixed_files
from fv3_ic_data import preprocess_only, validate_ic_grib_files
from fv3_namelists import update_nml_configs
from fv3_nesting import get_centers, nest_info, validate_nests
from fv3_paths import CONTAINER_UFS_BIN_DIR, configure_directories, paths
from fv3_pes_config import calc_cpu_alloc, get_grid_info
from fv3_plot_grid import plot_grid
from fv3_regional_bc import BC_INTERVAL_HOURS
from fv3_restart_driver import restart_driver, stage_restart, verify_restart_config
from fv3_runscripts import gen_shield_run_sh
from fv3_runtime import log
from fv3_state import compute_checksum, save_fv3_state, state
from fv3_utils import (
    cres_to_deg,
    format_forecast_length,
    get_allocated_nodes,
    load_config,
    parse_datetime,
    parse_resolution,
    require_minimum_cpus,
    runtime_env_vars,
    segment_hours,
    to_list,
)
from global_cycle_driver import GLOBAL_GTYPES, run_surface_cycle
from release import resolve_release

# Files the user may supply; logged with their full path when set.
USER_FILES = {
    "model_input_dir": "Model input directory",
    "ims_snow_file": "IMS snow file",
    "ims_ice_file": "IMS ice file",
    "five_min_ice_file": "Five-minute ice file",
    "global_cycle_sst_file": "Global cycle SST file",
    "global_cycle_ice_file": "Global cycle ice file",
    "global_cycle_snow_file": "Global cycle snow file",
}


def _log_initial_state() -> None:
    log.info("Configuration file: %s", state.run_config)
    log.info("Case directory: %s", state.case_dir)
    log.info("Submit directory: %s", state.submit_dir)
    if state.archive_data:
        log.info("Archive directory: %s", state.archive_dir)
    else:
        log.info("Archive directory: none (archive_data is false)")
    log.info("Fixed/static directory: %s", state.fix_src)

    log.info("Logs directory: %s", state.logs)

    if state.shield_use_container:
        log.info("Model executable: container image (SHiELD)")
    else:
        log.info("Model executable: %s", state.shield_exe)
    if state.preprocess_use_container:
        log.info("Preprocessing executables: container image (%s)", state.ufs_bin_dir)
    else:
        log.info("Preprocessing executables: %s", state.ufs_bin_dir)
    log.info("Python executable: %s", state.python)

    log.info("Description: %s", state.description)
    log.info("Initial run mode selected")
    if state.generate_ic_data:
        log.info("Full Grid/IC regeneration will be performed.")
    else:
        log.info(
            "Grid and initial conditions are taken from model_input_dir or the case directory."
        )

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

    log.info("Global tiles 1-6 resolution: %.2f km", state.res_km[0])

    if state.gtype == "nest":
        for message in nest_info:
            log.info(message)

        log.info("Number of nests: %s", state.n_nests)
        log.info("Refinement ratio: %s", state.refine_ratio)

    log.info("Target longitude: %s", state.target_lon)
    log.info("Target latitude: %s", state.target_lat)

    for key, label in USER_FILES.items():
        if state[key]:
            path = Path(os.path.expandvars(str(state[key]))).expanduser().resolve()
            modified = pd.Timestamp(path.stat().st_mtime, unit="s")
            log.info("%s: %s (%s)", label, path, modified.strftime("%Y-%m-%d"))


def validate_pyufs_options() -> None:
    """Check ic_grib_files and the global_cycle, emcsfc and cpld_gridgen settings."""
    validate_ic_grib_files()
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
        if (
            state[key]
            and not Path(os.path.expandvars(str(state[key]))).expanduser().exists()
        ):
            raise FileNotFoundError(f"{key} not found: {state[key]}")
    for flag in ("run_global_cycle", "run_cpld_gridgen"):
        if state[flag] and state.gtype not in GLOBAL_GTYPES:
            raise ValueError(f"{flag} supports gtype {', '.join(GLOBAL_GTYPES)}")
    if state.run_cpld_gridgen:
        state.cpld_gridgen_res = gridgen_res(state.cpld_gridgen_res)


def _load_initial_state() -> None:

    runtime_env = runtime_env_vars()
    runtime_config_path = Path(paths["submit_dir"]) / "run_config.yaml"
    run_config_includes: list[str] = []
    merged_config = load_config(
        runtime_config_path,
        includes=run_config_includes,
        defaults=Path(paths["configs"]) / "run_config.yaml",
    )
    merged_config["shield_release"] = resolve_release(
        merged_config.get("shield_release")
    )

    plugins_file = merged_config.pop("plugins_config", None)
    state.clear()
    state.update(merged_config)
    state.run_config_includes = run_config_includes
    # Checked by every restart segment, so edits to the grid or start-time
    # keys of run_config.yaml after the cold start are caught.
    state.config_checksum = compute_checksum(merged_config)
    state.update(runtime_env)
    # case_submit.py (release.resolve_runtime) checked the native paths. The use_container
    # switches alone choose the mode; the path of a containerized side is ignored.
    state.shield_exe = (
        None if state.shield_use_container else os.path.expandvars(state.shield_exe)
    )
    _ufs_bin_dir = (
        CONTAINER_UFS_BIN_DIR if state.preprocess_use_container else state.ufs_bin_dir
    )
    state.ufs_bin_dir = Path(os.path.expandvars(str(_ufs_bin_dir)))

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
    if plugins_file:
        plugins_path = (Path(os.path.expandvars(plugins_file)).expanduser()).resolve()
        plugins_config_includes: list[str] = []
        state.plugins = fv3_plugins.load_plugins(plugins_path, plugins_config_includes)
        state.plugins_config_includes = plugins_config_includes
        log.info("Plugins file: %s", plugins_path)
    # Fail before grid and IC generation; files are written after preprocessing.
    fv3_plugins.validate()
    if state.generate_ic_data:
        validate_pyufs_options()
    state.continue_run = False
    state.warm_start = False
    state.restart_no = 0
    state.resubmit_idx = 0
    state.total_restarts = state.resubmit + 1
    state.merged_files = []
    state.node_list = {0: get_allocated_nodes()}

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

    available_cpus = require_minimum_cpus()
    if available_cpus < 32 and not os.environ.get("CASE_MEMORY_FLAG"):
        state.preprocess_dask_scheduler = "synchronous"
    else:
        state.preprocess_dask_scheduler = "processes"

    state.preprocess_node_cpu_count = available_cpus
    state.preprocess_node_hostname = socket.gethostname()

    _log_initial_state()


def init_driver() -> None:
    _load_initial_state()

    restart_dir = os.environ.get("CASE_RESTART_DIR")
    restart_state = os.environ.get("CASE_RESTART_STATE")
    if restart_dir:
        verify_restart_config(Path(restart_state))

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
            exe_dir=state.ufs_bin_dir,
            fix_dir=state.fix_src,
        )

        get_grid_info(state.tmp / "grid")
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
    calc_cpu_alloc(state.grid)
    update_nml_configs()
    update_table_files()
    update_fixed_files()
    gen_shield_run_sh()
    fv3_plugins.apply("cold_start")

    save_fv3_state()
    log.info(f"Total PEs needed for run: {state.total_pes}")
    log.info("Preprocessing complete")

    if restart_dir:
        stage_restart(Path(restart_dir), Path(restart_state))
        restart_driver()
