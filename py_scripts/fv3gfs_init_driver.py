from chgres_cube import run_chgres_cube
from fv3gfs_driver_grid import run_driver
from fv3gfs_ensemble_driver import ensemble_config
from fv3gfs_ic_data import ic_only, initialize_ic_from_existing_case
from fv3gfs_namelists import update_nml_configs
from fv3gfs_runscripts import gen_shield_run_sh
from fv3gfs_runtime import log
from fv3gfs_state import save_state, state
from sm_pertubutions import apply_perturbations


def init_driver():

    if not state.ic_gen:
        initialize_ic_from_existing_case()

    else:
        log.info("Starting FV3 Grid and IC generation driver")

        run_driver(
            res=state.res,
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
            n_nests=state.n_nests,
            halo=state.halo,
            idim=state.idim,
            jdim=state.jdim,
            delx=state.delx,
            dely=state.dely,
            orog_dir=state.fix / "orog",
            tmp=state.tmp,
            exe_dir=state.ufs_exe,
            fix_dir=state.fix,
        )

        # Generate ICs
        run_chgres_cube()
        ensemble_config()
    log.info(f"Diag files staged at: {state.home}")
    log.info(f"IC files staged at: {state.input}")
    log.info(f"Fix files staged at: {state.fixed}")
    log.info(f"Grid files staged at: {state.grid}")
    log.info(f"Mosaic files staged at: {state.grid}")

    if state.ic_only:
        ic_only()
        return

    log.info("Init Run")
    update_nml_configs()
    apply_perturbations()
    save_state()
    gen_shield_run_sh()
