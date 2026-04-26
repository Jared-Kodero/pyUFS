from pathlib import Path

from fv3gfs_filter_topo import run_filter_topo
from fv3gfs_make_grid import run_make_grid
from fv3gfs_make_lake import run_add_lakefrac
from fv3gfs_make_mosaic import run_make_mosaic
from fv3gfs_make_orog import run_make_orog
from fv3gfs_make_orog_gsl import run_make_orog_gsl
from fv3gfs_nesting import get_nest_indices
from fv3gfs_runtime import get_newres, log
from fv3gfs_shave import run_shave
from fv3gfs_state import save_state, state
from fv3gfs_utils import cp
from sfc_climo_gen import run_sfc_climo_gen


def run_driver(
    res: int = None,
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

    tmp_ic_dir = tmp / "ic"
    tmp_ic_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # === GLOBAL GRIDS: uniform, stretch, nest =================
    # ==========================================================
    log.info("Generating Grid and Orography files")
    if gtype in ["uniform", "stretch", "nest"]:
        run_make_grid(
            res=res,
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
        )

        run_make_mosaic(
            res=res,
            gtype=gtype,
            exec_dir=exe_dir,
            out_dir=tmp / "grid",
        )

        if gtype == "nest":
            n_tiles = 6 + n_nests
        else:
            n_tiles = 6

        # --- Make orography per tile ---

        tiles = [i + 1 for i in range(n_tiles)]

        run_make_orog(
            res=res,
            tiles=tiles,
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            orog_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
        )

        run_make_orog_gsl(
            make_gsl_orog=make_gsl_orog,
            res=res,
            tiles=tiles,
            halo=-999,  # no-halo mode
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            topo_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
        )

        # --- Add lake fraction if requested ---

        run_add_lakefrac(
            add_lake=add_lake,
            res=res,
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
                res=res,
                gtype=gtype,
                exec_dir=exe_dir,
                grid_dir=tmp / "grid",
                orog_dir=tmp / "orog",
                tmp_dir=tmp / "filter_topo",
                stretch_factor=stretch_factor,
            )
        elif gtype == "nest":
            run_filter_topo(
                res=res,
                gtype="stretch",
                exec_dir=exe_dir,
                grid_dir=tmp / "grid",
                orog_dir=tmp / "orog",
                tmp_dir=tmp / "filter_topo",
                stretch_factor=stretch_factor,
            )

        # --- Copy outputs to tmp_ic_dir ---
        grid_files = list((tmp / "grid").glob(f"C{res}_grid.tile*.nc"))
        mosaic_files = list((tmp / "grid").glob(f"C{res}_*mosaic*.nc"))
        filter_topo_files = list((tmp / "filter_topo").glob("*.nc"))

        for f in grid_files + mosaic_files + filter_topo_files:
            cp(f, tmp_ic_dir)

        if gtype == "nest":
            for tile in range(7, 7 + n_nests):
                cp(tmp / "orog" / f"oro.C{res}.tile{tile}.nc", tmp_ic_dir)

        if make_gsl_orog:
            gsl_orog_files = list((tmp / "orog").glob("*.nc"))
            for f in gsl_orog_files:
                cp(f, tmp_ic_dir)

        # --- Surface climatology ---

        run_sfc_climo_gen(
            res=res,
            input_sfc_climo_dir=fix_dir / "sfc_climo",
            exec_dir=exe_dir,
            tmp_dir=tmp / "fix_sfc",
            out_dir=tmp_ic_dir / "fix_sfc",
            fix_dir=fix_dir,
            mosaic_dir=tmp_ic_dir,
            orog_dir=tmp_ic_dir,
            grid_type=gtype,
            n_nests=n_nests,
        )

    # ==========================================================
    # === REGIONAL GRIDS: gfdl, esg ============================
    # ==========================================================
    elif gtype in ["regional_gfdl", "regional_esg"]:
        tile = 7
        halop1 = halo + 1 if halo else 4

        offsets = get_nest_indices(
            parent_res=res,
            refine_ratio=refine_ratio,
            lon_min=state.lon_min,
            lon_max=state.lon_max,
            lat_min=state.lat_min,
            lat_max=state.lat_max,
            parent_tile=6,
            grid_dir=None,
        )

        istart_nest = offsets.istart_nest
        iend_nest = offsets.iend_nest
        jstart_nest = offsets.jstart_nest
        jend_nest = offsets.jend_nest
        parent_tile = offsets.parent_tile

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
                res=res,
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
                res=res,
                gtype=gtype,
                exec_dir=exe_dir,
                out_dir=tmp / "grid",
            )

        elif gtype == "regional_esg":
            # --- Make grid ---
            run_make_grid(
                res=res,
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
                res=res,
                gtype=gtype,
                exec_dir=exe_dir,
                out_dir=tmp / "grid",
            )

        # --- Replace res with derived resolution ---
        res = get_newres(tmp / "grid" / f"C{res}_grid.tile7.nc")

        # --- Make orography ---
        run_make_orog(
            res=res,
            tile=tile,
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            orog_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
        )

        run_add_lakefrac(
            add_lake=add_lake,
            res=res,
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
            res=res,
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
            res=res,
            tile=7,
            exec_dir=exe_dir,
            tmp_dir=tmp / "filter_topo",
            grid_dir=tmp / "grid",
            tmp_ic_dir=tmp_ic_dir,
        )

        # --- Copy mosaics ---
        for f in (tmp / "grid").glob(f"C{res}_*mosaic.nc"):
            cp(f, tmp_ic_dir)

        # --- Run GSL orography (after halo0 shave) ---

        run_make_orog_gsl(
            make_gsl_orog=make_gsl_orog,
            res=res,
            tile=tile,
            halo=0,
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            topo_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
        )
        for f in (tmp / "orog").glob(f"C{res}_oro_data_*tile{tile}*.nc"):
            cp(f, tmp_ic_dir)

        # --- Regional surface climatology ---
        grid_symlink = tmp_ic_dir / f"C{res}_grid.tile7.nc"
        oro_symlink = tmp_ic_dir / f"C{res}_oro_data.tile7.nc"
        grid_symlink.symlink_to(tmp_ic_dir / f"C{res}_grid.tile7.halo{halop1}.nc")
        oro_symlink.symlink_to(tmp_ic_dir / f"C{res}_oro_data.tile7.halo{halop1}.nc")

        run_sfc_climo_gen(
            res=res,
            input_sfc_climo_dir=fix_dir / "sfc_climo",
            exec_dir=exe_dir,
            tmp_dir=tmp / "fix_sfc",
            out_dir=tmp_ic_dir / "fix_sfc",
            fix_dir=fix_dir,
            mosaic_dir=tmp_ic_dir,
            orog_dir=tmp_ic_dir,
            grid_type="regional",
            n_nests=0,
        )

        grid_symlink.unlink(missing_ok=True)
        oro_symlink.unlink(missing_ok=True)

    else:
        raise ValueError(f"Unsupported grid type: {gtype}")

    # set flag indicating grid generation complete
    save_state()  # Save configuration state
