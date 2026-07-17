import os

from chgres_cube import run_chgres_cube
from fv3_driver_grid import run_driver
from fv3_ensemble_driver import ensemble_config
from fv3_external_ic import init_external_ic
from fv3_ic_data import preprocess_only
from fv3_namelists import update_nml_configs
from fv3_pes_config import calc_cpu_alloc
from fv3_plot_grid import plot_grid
from fv3_runscripts import gen_shield_run_sh
from fv3_runtime import log
from fv3_state import save_fv3_state, state
from sm_perturbations import apply_perturbations


def init_driver():
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
            tmp=state.ic_data,
            exe_dir=state.ufs_exe,
            fix_dir=state.fix_src,
        )

        # Generate ICs
        run_chgres_cube()
        ensemble_config()

        plot_grid()

    if state.preprocess_only:
        preprocess_only()
        return

    os.chdir(state.work_dir)
    calc_cpu_alloc(state.grid)
    update_nml_configs()
    apply_perturbations()
    save_fv3_state()
    gen_shield_run_sh()

    log.info("Starting init Run")
