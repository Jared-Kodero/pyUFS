import shutil
from pathlib import Path

import f90nml
from fv3_nesting import gen_global_nest_parent, get_nest_indices
from fv3_runtime import log, staged_files, tmp_cwd, to_list
from fv3_state import save_fv3_state, state
from fv3_utils import cp, run_cmd


def nest_parents(parent_tile: int | list, n_nests: int, nest_type: str) -> list[int]:
    """Parent tile of each nest, as make_hgrid --parent_tile takes it.

    Same-level nests sit on global tiles (one value applies to every nest).
    In a telescoping chain the outermost nest sits on a global tile and nest
    n on nest n-1 (tile 6 + n - 1).
    """
    parents = to_list(parent_tile)
    if len(parents) != n_nests:
        parents = [parents[0]] * n_nests
    if nest_type == "telescoping":
        return [parents[0]] + [7 + i for i in range(n_nests - 1)]
    return [int(p) for p in parents]


def _hgrid_nests_cmd(
    make_hgrid: str,
    nlon: int,
    c_res: int,
    stretch_factor: float,
    parents: list[int],
    refine_ratio: list[int],
    halo: int,
) -> list[str]:
    """make_hgrid call for the global cube plus the first len(parents) nests,
    using the nest indices recorded in state."""
    n = len(parents)

    def join(values: list) -> str:
        return ",".join(str(v) for v in values[:n])

    return [
        f"{make_hgrid}",
        "--grid_type",
        "gnomonic_ed",
        "--nlon",
        f"{nlon}",
        "--grid_name",
        f"C{c_res}_grid",
        "--do_schmidt",
        "--stretch_factor",
        f"{stretch_factor}",
        "--target_lon",
        f"{state.target_lon}",
        "--target_lat",
        f"{state.target_lat}",
        "--nest_grids",
        f"{n}",
        "--parent_tile",
        join(parents),
        "--refine_ratio",
        join(refine_ratio),
        "--istart_nest",
        join(state.istart_nest),
        "--jstart_nest",
        join(state.jstart_nest),
        "--iend_nest",
        join(state.iend_nest),
        "--jend_nest",
        join(state.jend_nest),
        "--halo",
        f"{halo}",
        "--great_circle_algorithm",
    ]


def make_nested_grid(
    make_hgrid: str,
    nlon: int,
    c_res: int,
    stretch_factor: float,
    parent_tile: list,
    out_dir: Path,
    halo: int,
    gtype: str,
):
    """Global cube and all nests from one make_hgrid --nest_grids call.

    One call names the nest tiles tile7, tile8, ... inside the grid files,
    which sfc_climo_gen and chgres_cube use to name their per-tile files;
    copies of separate single-nest runs would all carry "tile7". Each nest is
    bracketed on the final grid of its parent: a global tile, or for a
    telescoping chain the previous nest, which is first generated with the
    nests before it.
    """
    log_file = state.logs / "make_nested_grid.log"
    n_nests = state.n_nests
    refine_ratio = [int(r) for r in to_list(state.refine_ratio)]
    parents = nest_parents(parent_tile, n_nests, state.nest_type)

    if state.nest_type == "telescoping":
        log.info("Generating telescoped nested grids")
    else:
        log.info("Generating same-level nested grids")

    work = out_dir / "tmp_nested"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    # Parent grids for the bracketing: the global cube, then for a nest whose
    # parent is a nest, the grid with all nests before it.
    (work / "level0").mkdir()
    parent_dir = gen_global_nest_parent(c_res, work / "level0")
    for i, parent in enumerate(parents):
        if parent > 6:
            parent_dir = work / f"level{i}"
            parent_dir.mkdir()
            cmd = _hgrid_nests_cmd(
                make_hgrid, nlon, c_res, stretch_factor, parents[:i], refine_ratio, halo
            )
            result, msgs = run_cmd(
                cmd, cwd=parent_dir, stdout=log_file, stderr=log_file
            )
            if result != 0:
                log.error(msgs)
                raise RuntimeError(
                    f"Failed to generate the parent grid of nest {i + 2:02d}"
                )
        get_nest_indices(
            c_res=c_res, tile_idx=i, grid_dir=parent_dir, parent_tile=parent
        )

    final = work / "final"
    final.mkdir()
    cmd = _hgrid_nests_cmd(
        make_hgrid, nlon, c_res, stretch_factor, parents, refine_ratio, halo
    )
    result, msgs = run_cmd(cmd, cwd=final, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate the nested grids")

    for f in out_dir.glob(f"C{c_res}_grid.tile*.nc"):
        f.unlink()
    for f in final.glob(f"C{c_res}_grid.tile*.nc"):
        shutil.move(str(f), str(out_dir / f.name))
    shutil.rmtree(work)
    save_fv3_state()


def record_nest_indices(c_res: int, grid_dir: Path) -> None:
    """Rebuild the nest indices of fv_nest_nml from existing grid files.

    Grids reused from IC/grid skip make_nested_grid, which otherwise records
    them. Each nest is bracketed on its parent tile in grid_dir, the same
    final parent grid it was generated from.
    """
    parents = nest_parents(state.parent_tile, state.n_nests, state.nest_type)
    for i, parent in enumerate(parents):
        get_nest_indices(c_res=c_res, tile_idx=i, grid_dir=grid_dir, parent_tile=parent)


def make_uniform_grid(make_hgrid: str, nlon: int, c_res: int):
    log_file = state.logs / "make_uniform_grid.log"
    cmd = [
        f"{make_hgrid}",
        "--grid_type",
        "gnomonic_ed",
        "--nlon",
        f"{nlon}",
        "--grid_name",
        f"C{c_res}_grid",
        "--do_schmidt",
        "--stretch_factor",
        f"{state.stretch_factor}",
        "--target_lon",
        f"{state.target_lon}",
        "--target_lat",
        f"{state.target_lat}",
        "--great_circle_algorithm",
    ]

    log.info(f"Generating uniform grid: C{c_res}")
    result, msgs = run_cmd(cmd, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate uniform grid")


def make_stretched_grid(
    make_hgrid: str,
    nlon: int,
    c_res: int,
    stretch_factor: float,
    target_lon: float,
    target_lat: float,
):
    log_file = state.logs / "make_stretched_grid.log"

    if stretch_factor == 1:
        raise ValueError("Stretch factor must be greater than 1 for stretched grid.")

    cmd = [
        f"{make_hgrid}",
        "--grid_type",
        "gnomonic_ed",
        "--nlon",
        f"{nlon}",
        "--grid_name",
        f"C{c_res}_grid",
        "--do_schmidt",
        "--stretch_factor",
        f"{stretch_factor}",
        "--target_lon",
        f"{target_lon}",
        "--target_lat",
        f"{target_lat}",
        "--great_circle_algorithm",
    ]

    result, msgs = run_cmd(cmd, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate stretched grid")


def make_regional_gfdl_grid(
    make_hgrid: str,
    nlon: int,
    c_res: int,
    stretch_factor: float,
    target_lon: float,
    target_lat: float,
    parent_tile: list,
    refine_ratio: list,
    istart_nest: list,
    jstart_nest: list,
    iend_nest: list,
    jend_nest: list,
    halo: int,
    out_dir: Path,
    global_equiv_resol: str,
):

    log_file = state.logs / "make_regional_gfdl_grid.log"

    cmd = [
        f"{make_hgrid}",
        "--grid_type",
        "gnomonic_ed",
        "--nlon",
        f"{nlon}",
        "--grid_name",
        f"C{c_res}_grid",
        "--do_schmidt",
        "--stretch_factor",
        f"{stretch_factor}",
        "--target_lon",
        f"{target_lon}",
        "--target_lat",
        f"{target_lat}",
        "--nest_grid",
        "--parent_tile",
        f"{parent_tile}",
        "--refine_ratio",
        f"{refine_ratio}",
        "--istart_nest",
        f"{istart_nest}",
        "--jstart_nest",
        f"{jstart_nest}",
        "--iend_nest",
        f"{iend_nest}",
        "--jend_nest",
        f"{jend_nest}",
        "--halo",
        f"{halo}",
        "--great_circle_algorithm",
    ]

    result, msgs = run_cmd(cmd, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate regional GFDL grid")

    grid_file = out_dir / f"C{c_res}_grid.tile7.nc"

    cmd = [f"{global_equiv_resol}", f"{grid_file}"]

    result, msgs = run_cmd(cmd, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to run global equiv resol")


def make_regional_esg_grid(
    regional_esg_grid: str,
    target_lon: float,
    target_lat: float,
    idim: int,
    jdim: int,
    delx: float,
    dely: float,
    halo: int,
    out_dir: Path,
    global_equiv_resol: str,
):
    log_file = state.logs / "make_regional_esg_grid.log"

    required = [target_lon, target_lat, idim, jdim, delx, dely, halo]
    if any(v is None for v in required):
        raise ValueError("Missing required parameters for regional_esg grid.")

    halop2 = halo + 2
    lx = -(idim + halop2 * 2)
    ly = -(jdim + halop2 * 2)

    # Create namelist file
    nml_file = out_dir / "regional_grid.nml"
    regional_grid_nml = {
        "regional_grid_nml": {
            "plon": target_lon,
            "plat": target_lat,
            "delx": delx,
            "dely": dely,
            "lx": lx,
            "ly": ly,
        }
    }
    with nml_file.open("w") as f:
        f90nml.write(regional_grid_nml, f)

    result, msgs = run_cmd([regional_esg_grid])
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate regional ESG grid")

    grid_file = out_dir / "regional_grid.nc"
    cmd = [f"{global_equiv_resol}", f"{grid_file}"]
    result, msgs = run_cmd(cmd, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to run global equiv resol")


def run_make_grid(
    c_res: int,
    gtype: str,
    exec_dir: Path,
    out_dir: Path,
    stretch_factor: float | None = None,
    target_lon: float | None = None,
    target_lat: float | None = None,
    refine_ratio: int | list[int] | None = None,
    istart_nest: int | list[int] | None = None,
    jstart_nest: int | list[int] | None = None,
    iend_nest: int | list[int] | None = None,
    jend_nest: int | list[int] | None = None,
    parent_tile: int | list[int] = 6,
    halo: int | None = None,
    idim: int | None = None,
    jdim: int | None = None,
    delx: float | None = None,
    dely: float | None = None,
    mod_dir: Path | None = None,
):
    """
    Generate FV3 grid NetCDF files and a mosaic using fv3 grid tools.

    This function serves as a Python wrapper around the `fv3_make_grid.sh`
    workflow, enabling the creation of cubed-sphere grids for the FV3 dynamical
    core. It supports global uniform, stretched, nested, and regional grid
    configurations. The generated grids and mosaics are compatible with FV3
    and UFS workflows.

    Parameters
    ----------
    c_res : int
        Base cubed-sphere resolution (e.g., 96 for a C96 grid).
    gtype : str
        Grid type. Must be one of:
        - `'uniform'` : Global uniform cubed-sphere grid.
        - `'stretch'` : Stretched global grid centered at a target point.
        - `'nest'` : One or more refined nests embedded within the global grid.
        - `'regional_gfdl'` : GFDL-style regional grid.
        - `'regional_esg'` : ESG-style regional grid.
    exec_dir : Path
        Path to the directory containing FV3 grid generation executables:
        `make_hgrid`, `make_solo_mosaic`, etc.
    out_dir : Path
        Directory in which to write generated grid and mosaic NetCDF files.
    stretch_factor : float, optional
        Stretching factor for stretched or nested grids.
    target_lon, target_lat : float, optional
        Longitude and latitude (degrees) of the stretching target or nest center.
    refine_ratio : int or list of int, optional
        Refinement ratio(s) for one or more nests.
    istart_nest, jstart_nest, iend_nest, jend_nest : int or list of int, optional
        Starting and ending i/j indices defining each nest within the parent grid.
    parent_tile : int or list of int, default=6
        Parent tile number(s) for each nest. Typically tile 6 is used for the
        North American region.
    n_nests : int, default=0
        Number of nested grids (0 for global-only grid).
    halo : int, optional
        Halo width (in grid points) used for regional or nested grids.
    idim, jdim : int, optional
        Domain dimensions (number of grid points) for ESG-style regional grids.
    delx, dely : float, optional
        Grid spacing (meters) for ESG regional grids in x and y directions.
    lon_min, lon_max, lat_min, lat_max : list of float, optional
        Lists of longitude/latitude bounds (degrees) for each telescopingor
        regional nest.
    nest_type : str, optional
        Nesting type. One of:
        - `'normal'` : normal independent nests on the same grid.
        - `'telescoping'` : Successive nested domains with increasing resolution.
    nest_resolutions : list of int, optional
        List of grid resolutions for the telescoping hierarchy, including the
        global resolution as the first element.
    mod_dir : Path, optional

    """

    reused = staged_files(mod_dir)
    if reused:
        src = str(mod_dir).replace(str(state.work_dir), str(state.case_dir))
        log.info(f"Using existing grid files from {src}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for file in reused:
            cp(file, out_dir / file.name)
        if gtype == "nest":
            record_nest_indices(c_res, out_dir)
        save_fv3_state()
        return

    regional_esg_grid = exec_dir / "regional_esg_grid"
    make_hgrid = exec_dir / "make_hgrid"
    global_equiv_resol = exec_dir / "global_equiv_resol"
    nlon = c_res * 2
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Grid generation
    # -------------------------------

    with tmp_cwd(out_dir):
        if gtype == "uniform":
            make_uniform_grid(make_hgrid, nlon, c_res)

        elif gtype == "stretch":
            make_stretched_grid(
                make_hgrid, nlon, c_res, stretch_factor, target_lon, target_lat
            )

        elif gtype == "nest":
            make_nested_grid(
                make_hgrid,
                nlon,
                c_res,
                stretch_factor,
                parent_tile,
                out_dir,
                halo,
                gtype,
            )

        elif gtype == "regional_gfdl":
            make_regional_gfdl_grid(
                make_hgrid,
                nlon,
                c_res,
                stretch_factor,
                target_lon,
                target_lat,
                parent_tile,
                refine_ratio,
                istart_nest,
                jstart_nest,
                iend_nest,
                jend_nest,
                halo,
                out_dir,
                global_equiv_resol,
            )

        elif gtype == "regional_esg":
            make_regional_esg_grid(
                regional_esg_grid,
                state.target_lon,
                state.target_lat,
                idim,
                jdim,
                delx,
                dely,
                halo,
                out_dir,
                global_equiv_resol,
            )

        else:
            raise ValueError(f"Unsupported gtype {gtype}")

        save_fv3_state()
