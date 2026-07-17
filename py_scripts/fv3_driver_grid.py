import sys
from pathlib import Path

from fv3_filter_topo import run_filter_topo
from fv3_make_grid import run_make_grid
from fv3_make_lake import run_add_lakefrac
from fv3_make_mosaic import run_make_mosaic
from fv3_make_orog import run_make_orog
from fv3_make_orog_gsl import run_make_orog_gsl
from fv3_nesting import get_nest_indices
from fv3_runtime import get_newres, log
from fv3_shave import run_shave
from fv3_state import save_fv3_state, state
from fv3_utils import clear_dir, cp
from sfc_climo_gen import run_sfc_climo_gen


def run_driver(
    c_res: int = None,
    gtype: str = None,
    add_lake: bool = None,
    lake_cutoff: float = None,
    make_gsl_orog: bool = None,
    stretch_factor: float = None,
    target_lon: float = None,
    target_lat: float = None,
    refine_ratio: float = None,
    istart_nest: float = None,
    jstart_nest: float = None,
    iend_nest: float = None,
    jend_nest: float = None,
    parent_tile: float = None,
    lon_min: float = None,
    lon_max: float = None,
    lat_min: float = None,
    lat_max: float = None,
    n_nests: int = None,
    halo: int = None,
    idim: int = None,
    jdim: int = None,
    delx: float = None,
    dely: float = None,
    # Paths
    tmp: Path = None,
    exe_dir: Path = None,
    orog_dir: Path = None,
    fix_dir: Path = None,
):
    """
    Python driver for FV3 grid/orography/sfc_climo generation.
    Clean, fully ordered, consistent with original bash workflow.
    """

    tmp_ic_dir = tmp / "input"
    tmp_ic_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # === GLOBAL GRIDS: uniform, stretch, nest =================
    # ==========================================================
    log.info("Generating Grid and Orography files")
    if gtype in ["uniform", "stretch", "nest"]:
        run_make_grid(
            c_res=c_res,
            gtype=gtype,
            exec_dir=exe_dir,
            out_dir=tmp / "grid",
            stretch_factor=stretch_factor,
            target_lon=target_lon,
            target_lat=target_lat,
            refine_ratio=refine_ratio,
            istart_nest=istart_nest,
            jstart_nest=jstart_nest,
            iend_nest=iend_nest,
            jend_nest=jend_nest,
            halo=halo,
            idim=idim,
            jdim=jdim,
            delx=delx,
            dely=dely,
            parent_tile=parent_tile,
            mod_dir=state.ic_data / "grid",
        )

        run_make_mosaic(
            c_res=c_res,
            gtype=gtype,
            exec_dir=exe_dir,
            out_dir=tmp / "grid",
            mod_dir=state.ic_data / "grid",
        )

        if state.preprocess_grid_only:
            cp(tmp / "grid", state.ic_data / "grid")
            clear_dir(tmp / "grid")

            path = str(state.ic_data / "grid").replace(
                str(state.work_dir), str(state.case_dir)
            )

            state.preprocess_grid_only = False
            save_fv3_state()
            log.info(f"Grid files staged in {path}")
            sys.exit(0)
            return

        if gtype == "nest":
            n_tiles = 6 + n_nests
        else:
            n_tiles = 6

        # --- Make orography per tile ---

        tiles = [i + 1 for i in range(n_tiles)]

        run_make_orog(
            c_res=c_res,
            tiles=tiles,
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            orog_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
            mod_dir=state.ic_data / "orography",
        )

        run_make_orog_gsl(
            make_gsl_orog=make_gsl_orog,
            c_res=c_res,
            tiles=tiles,
            halo=-999,  # no-halo mode
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            topo_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
            mod_dir=state.ic_data / "orography",
        )

        if state.preprocess_orog_only:
            cp(tmp / "orog", state.ic_data / "orography")
            clear_dir(tmp / "orog")
            path = str(state.ic_data / "orography").replace(
                str(state.work_dir), str(state.case_dir)
            )
            state.preprocess_orog_only = False
            save_fv3_state()
            log.info(f"Orography files staged in {path}")
            sys.exit(0)
            return

        # --- Add lake fraction if requested ---

        run_add_lakefrac(
            add_lake=add_lake,
            c_res=c_res,
            gtype=gtype,
            exec_dir=exe_dir,
            orog_dir=tmp / "orog",
            grid_dir=tmp / "grid",
            topo=orog_dir,
            lake_cutoff=lake_cutoff,
            tmp=tmp,
        )

        if gtype in ["uniform", "stretch"]:
            run_filter_topo(
                c_res=c_res,
                gtype=gtype,
                exec_dir=exe_dir,
                grid_dir=tmp / "grid",
                orog_dir=tmp / "orog",
                tmp_dir=tmp / "filter_topo",
                stretch_factor=stretch_factor,
            )
        elif gtype == "nest":
            run_filter_topo(
                c_res=c_res,
                gtype=gtype,
                exec_dir=exe_dir,
                grid_dir=tmp / "grid",
                orog_dir=tmp / "orog",
                tmp_dir=tmp / "filter_topo",
                stretch_factor=stretch_factor,
            )

        # --- Copy outputs to tmp_ic_dir ---
        grid_files = list((tmp / "grid").glob(f"C{c_res}_grid.tile*.nc"))
        mosaic_files = list((tmp / "grid").glob(f"C{c_res}_*mosaic*.nc"))
        filter_topo_files = list((tmp / "filter_topo").glob("*.nc"))

        for f in grid_files + mosaic_files + filter_topo_files:
            cp(f, tmp_ic_dir)

        if gtype == "nest":
            for tile in range(7, 7 + n_nests):
                cp(tmp / "orog" / f"oro.C{c_res}.tile{tile}.nc", tmp_ic_dir)

        if make_gsl_orog:
            gsl_orog_files = list((tmp / "orog").glob("*.nc"))
            for f in gsl_orog_files:
                cp(f, tmp_ic_dir)

        # --- Surface climatology ---

        run_sfc_climo_gen(
            c_res=c_res,
            input_sfc_climo_dir=fix_dir / "sfc_climo",
            exec_dir=exe_dir,
            tmp_dir=tmp / "fix_sfc",
            out_dir=tmp_ic_dir / "fix_sfc",
            fix_dir=fix_dir,
            mosaic_dir=tmp_ic_dir,
            orog_dir=tmp_ic_dir,
            grid_type=gtype,
            halo=halo if gtype == "nest" else 0,
            n_nests=n_nests,
        )

    # ==========================================================
    # === REGIONAL GRIDS: gfdl, esg ============================
    # ==========================================================
    elif gtype in ["regional_gfdl", "regional_esg"]:
        tile = 7
        halop1 = halo + 1 if halo else 4

        get_nest_indices(
            c_res=c_res,
            tile_idx=0,  # parent tile 6 is always tile_idx 0 for nesting
            grid_dir=None,
            parent_tile=6,
            i_refine_ratio=refine_ratio[0]
            if isinstance(refine_ratio, list)
            else refine_ratio,
        )

        istart_nest = state.istart_nest[0]
        iend_nest = state.iend_nest[0]
        jstart_nest = state.jstart_nest[0]
        jend_nest = state.jend_nest[0]
        parent_tile = state.parent_tile[0]

        # --- Expand halo region for regional_gfdl ---
        if gtype == "regional_gfdl":
            nptsx = int(iend_nest - istart_nest + 1)
            nptsy = int(jend_nest - jstart_nest + 1)
            idim = int(nptsx * refine_ratio / 2)
            jdim = int(nptsy * refine_ratio / 2)

            add = 0
            while True:
                add += 1
                iend_halo = iend_nest + add
                istart_halo = istart_nest - add
                jend_halo = jend_nest + add
                jstart_halo = jstart_nest - add
                new_nptsx = iend_halo - istart_halo + 1
                new_idim = int(new_nptsx * refine_ratio / 2)
                if new_idim - idim >= 10:
                    break
            istart_nest, iend_nest, jstart_nest, jend_nest = (
                istart_halo,
                iend_halo,
                jstart_halo,
                jend_halo,
            )

            # --- Make grid ---
            run_make_grid(
                c_res=c_res,
                gtype=gtype,
                exec_dir=exe_dir,
                out_dir=tmp / "grid",
                stretch_factor=stretch_factor,
                target_lon=target_lon,
                target_lat=target_lat,
                refine_ratio=refine_ratio,
                istart_nest=istart_nest,
                jstart_nest=jstart_nest,
                iend_nest=iend_nest,
                jend_nest=jend_nest,
                parent_tile=6,
                halo=halo,
                idim=idim,
                jdim=jdim,
                delx=delx,
                dely=dely,
            )

            run_make_mosaic(
                c_res=c_res,
                gtype=gtype,
                exec_dir=exe_dir,
                out_dir=tmp / "grid",
            )

        elif gtype == "regional_esg":
            # --- Make grid ---
            run_make_grid(
                c_res=c_res,
                gtype=gtype,
                exec_dir=exe_dir,
                out_dir=tmp / "grid",
                stretch_factor=stretch_factor,
                target_lon=target_lon,
                target_lat=target_lat,
                refine_ratio=refine_ratio,
                istart_nest=istart_nest,
                jstart_nest=jstart_nest,
                iend_nest=iend_nest,
                jend_nest=jend_nest,
                parent_tile=parent_tile,
                halo=halo,
                idim=idim,
                jdim=jdim,
                delx=delx,
                dely=dely,
            )

            run_make_mosaic(
                c_res=c_res,
                gtype=gtype,
                exec_dir=exe_dir,
                out_dir=tmp / "grid",
            )

            if state.preprocess_grids_only:
                state.grids_preprocessed = True
                log.info(
                    "Preprocess grids only requested. Exiting after grid generation."
                )
                save_fv3_state()
                return

        # --- Replace c_res with derived resolution ---
        old_res = c_res
        c_res = get_newres(tmp / "grid" / f"C{c_res}_grid.tile7.nc")

        # replace c_res part in the prev generated grid/orog files with the new c_res for consistency
        for f in (tmp / "grid").glob(f"*{old_res}*"):
            new_name = f.name.replace(f"{old_res}", f"{c_res}")
            f.rename(tmp / "grid" / new_name)

        # --- Make orography ---
        run_make_orog(
            c_res=c_res,
            tiles=[tile],
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            orog_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
        )

        if state.preprocess_orography_only:
            state.orography_preprocessed = True
            log.info(
                "Preprocess orography only requested. Exiting after orography generation."
            )
            save_fv3_state()
            return

        run_add_lakefrac(
            add_lake=add_lake,
            c_res=c_res,
            gtype=gtype,
            exec_dir=exe_dir,
            orog_dir=tmp / "orog",
            grid_dir=tmp / "grid",
            topo=orog_dir,
            lake_cutoff=lake_cutoff,
            tmp=tmp,
        )

        # --- Filter topography ---
        run_filter_topo(
            c_res=c_res,
            gtype=gtype,
            exec_dir=exe_dir,
            grid_dir=tmp / "grid",
            orog_dir=tmp / "orog",
            tmp_dir=tmp / "filter_topo",
            stretch_factor=stretch_factor,
        )

        run_shave(
            idim=idim,
            jdim=jdim,
            halo=halo,
            halop1=halop1,
            c_res=c_res,
            tile=7,
            exec_dir=exe_dir,
            tmp_dir=tmp / "filter_topo",
            grid_dir=tmp / "grid",
            tmp_ic_dir=tmp_ic_dir,
        )

        # --- Copy mosaics ---
        for f in (tmp / "grid").glob(f"C{c_res}_*mosaic.nc"):
            cp(f, tmp_ic_dir)

        # --- Run GSL orography (after halo0 shave) ---

        run_make_orog_gsl(
            make_gsl_orog=make_gsl_orog,
            c_res=c_res,
            tiles=[tile],
            halo=0,
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            topo_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
        )
        for f in (tmp / "orog").glob(f"C{c_res}_oro_data_*tile{tile}*.nc"):
            cp(f, tmp_ic_dir)

        # --- Regional surface climatology ---
        grid_symlink = tmp_ic_dir / f"C{c_res}_grid.tile7.nc"
        oro_symlink = tmp_ic_dir / f"C{c_res}_oro_data.tile7.nc"
        grid_symlink.symlink_to(tmp_ic_dir / f"C{c_res}_grid.tile7.halo{halop1}.nc")
        oro_symlink.symlink_to(tmp_ic_dir / f"C{c_res}_oro_data.tile7.halo{halop1}.nc")

        run_sfc_climo_gen(
            c_res=c_res,
            input_sfc_climo_dir=fix_dir / "sfc_climo",
            exec_dir=exe_dir,
            tmp_dir=tmp / "fix_sfc",
            out_dir=tmp_ic_dir / "fix_sfc",
            fix_dir=fix_dir,
            mosaic_dir=tmp_ic_dir,
            orog_dir=tmp_ic_dir,
            grid_type="regional",
            halo=halop1,
            n_nests=0,
        )

        grid_symlink.unlink(missing_ok=True)
        oro_symlink.unlink(missing_ok=True)

    else:
        raise ValueError(f"Unsupported grid type: {gtype}")

    # set flag indicating grid generation complete
    save_fv3_state()  # Save configuration state
