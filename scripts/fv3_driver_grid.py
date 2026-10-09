import sys
from pathlib import Path

from fv3_filter_topo import run_filter_topo
from fv3_make_grid import run_make_grid
from fv3_make_lake import run_add_lakefrac
from fv3_make_mosaic import run_make_mosaic
from fv3_make_orog import run_make_orog
from fv3_make_orog_gsl import run_make_orog_gsl
from fv3_nesting import get_nest_indices
from fv3_runtime import GRID_KEYS, OROGRAPHY_KEYS, get_newres, log, stage_for_reuse
from fv3_shave import run_shave
from fv3_state import save_fv3_state, state
from fv3_utils import cp
from sfc_climo_gen import run_sfc_climo_gen

# Minimum number of regional grid points gained in each direction by widening
# the bracket of a regional_gfdl domain (UFS_UTILS fv3gfs_driver_grid.sh).
REGIONAL_GFDL_EXTRA = 10


def widen_regional_bracket(
    istart: int, iend: int, jstart: int, jend: int, refine: int
) -> tuple[int, int, int, int]:
    """Widen a regional_gfdl parent bracket by `add` supergrid points per side
    until the grid gains at least REGIONAL_GFDL_EXTRA points per direction.

    add grows in steps of two, which keeps istart odd and iend even for any
    refine_ratio (make_hgrid requires both); the reference shell grows it by
    one and fails for refine ratios 2, 4 and >= 10.
    """
    add = 0
    # Gain per direction: add/2 parent cells on each of two sides, refine points each.
    while add * refine < REGIONAL_GFDL_EXTRA:
        add += 2
    return istart - add, iend + add, jstart - add, jend + add


def _stage_and_exit(src: Path, kind: str) -> None:
    """End a preprocess_grid_only or preprocess_orog_only run: keep the files
    in IC/<kind> for the next run (fv3_runtime.staged_files) and exit."""
    dest = state.ic_data / kind
    stage_for_reuse(src, dest, OROGRAPHY_KEYS if kind == "orography" else GRID_KEYS)
    path = str(dest).replace(str(state.work_dir), str(state.case_dir))
    state.preprocess_only = False
    state.preprocess_grid_only = False
    state.preprocess_orog_only = False
    save_fv3_state()
    log.info(f"{kind.capitalize()} files staged in {path}")
    sys.exit(0)


def run_driver(
    c_res: int | None = None,
    gtype: str | None = None,
    add_lake: bool | None = None,
    lake_cutoff: float | None = None,
    make_gsl_orog: bool | None = None,
    stretch_factor: float | None = None,
    target_lon: float | None = None,
    target_lat: float | None = None,
    refine_ratio: float | None = None,
    istart_nest: float | None = None,
    jstart_nest: float | None = None,
    iend_nest: float | None = None,
    jend_nest: float | None = None,
    parent_tile: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    n_nests: int | None = None,
    halo: int | None = None,
    idim: int | None = None,
    jdim: int | None = None,
    delx: float | None = None,
    dely: float | None = None,
    # Paths
    tmp: Path | None = None,
    exe_dir: Path | None = None,
    orog_dir: Path | None = None,
    fix_dir: Path | None = None,
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
            _stage_and_exit(tmp / "grid", "grid")

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
            _stage_and_exit(tmp / "orog", "orography")

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

        if gtype in ["uniform", "stretch"] or gtype == "nest":
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
            # Only the GSL products; tmp/orog also holds the unfiltered oro.*
            # files, which must not replace the filtered global orography.
            for f in (tmp / "orog").glob(f"C{c_res}_oro_data_*.nc"):
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
        halop1 = halo + 1

        # --- Expand halo region for regional_gfdl ---
        if gtype == "regional_gfdl":
            # The GFDL regional grid is carved from parent tile 6 like a single
            # nest; the bounding box was coerced to length-1 lists in the init
            # driver. The ESG grid is defined by idim/jdim/delx/dely alone.
            refine = refine_ratio[0] if isinstance(refine_ratio, list) else refine_ratio
            get_nest_indices(
                c_res=c_res,
                tile_idx=0,  # single regional domain
                grid_dir=None,
                parent_tile=6,
                i_refine_ratio=refine,
            )

            istart_nest = state.istart_nest[0]
            iend_nest = state.iend_nest[0]
            jstart_nest = state.jstart_nest[0]
            jend_nest = state.jend_nest[0]

            # The GFDL regional grid is carved from the parent tile, so the
            # bracket is widened until it gains at least the blend halo width.
            nptsx = int(iend_nest - istart_nest + 1)
            nptsy = int(jend_nest - jstart_nest + 1)
            idim = int(nptsx * refine / 2)
            jdim = int(nptsy * refine / 2)

            istart_nest, iend_nest, jstart_nest, jend_nest = widen_regional_bracket(
                istart_nest, iend_nest, jstart_nest, jend_nest, refine
            )
            # make_hgrid needs `halo` parent cells around the widened bracket.
            for s_idx, e_idx in ((istart_nest, iend_nest), (jstart_nest, jend_nest)):
                if (s_idx + 1) // 2 - halo < 1 or e_idx // 2 + halo > c_res:
                    raise ValueError(
                        "regional_gfdl box with its blending margin reaches the edge of "
                        + "tile 6; shrink the box or move it away from the tile edge"
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
                refine_ratio=refine,
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
                mod_dir=state.ic_data / "grid",
            )

            run_make_mosaic(
                c_res=c_res,
                gtype=gtype,
                exec_dir=exe_dir,
                out_dir=tmp / "grid",
                mod_dir=state.ic_data / "grid",
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
                halo=halo,
                idim=idim,
                jdim=jdim,
                delx=delx,
                dely=dely,
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
            _stage_and_exit(tmp / "grid", "grid")

        # --- Replace c_res with derived resolution ---
        # global_equiv_resol wrote the equivalent global resolution into the
        # tile-7 grid, which make_mosaic already used to name the grid file. The
        # ESG file is named with that derived resolution rather than the input
        # c_res, so discover it by pattern instead of assuming the input value.
        old_res = c_res
        grid_tile7 = next(iter((tmp / "grid").glob("C*_grid.tile7.nc")))
        c_res = get_newres(grid_tile7)

        # Downstream chgres_cube and staging key off state.c_res; keep it aligned
        # with the derived regional resolution.
        state.c_res = c_res

        # Name the files after the derived resolution. Only the C<res>_ prefix
        # is replaced (the digits of old_res may occur inside the new value).
        if old_res != c_res:
            for f in list((tmp / "grid").glob(f"C{old_res}_*")):
                f.rename(
                    f.with_name(f"C{c_res}_" + f.name.removeprefix(f"C{old_res}_"))
                )

        # --- Make orography ---
        run_make_orog(
            c_res=c_res,
            tiles=[tile],
            grid_dir=tmp / "grid",
            out_dir=tmp / "orog",
            orog_dir=orog_dir,
            exec_dir=exe_dir,
            tmp=tmp,
            mod_dir=state.ic_data / "orography",
        )

        if state.preprocess_orog_only:
            _stage_and_exit(tmp / "orog", "orography")

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
            grid_dir=tmp_ic_dir,  # the halo0 grid written by run_shave
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
