import os
import shutil
from pathlib import Path

import f90nml
import numpy as np
from fv3gfs_cpu_config import calc_cpu_alloc
from fv3gfs_nesting import (
    gen_global_nest_parent,
    get_nest_indices,
    get_nest_tele_indices,
)
from fv3gfs_runtime import log, to_list
from fv3gfs_state import save_state, state
from fv3gfs_utils import cp, rename, run_cmd


def make_nested_grid(
    make_hgrid,
    nlon,
    res,
    stretch_factor,
    parent_tile,
    out_dir,
    halo,
    gtype,
):
    log_file = state.logs / "make_nested_grid.log"

    n_nests = state.n_nests
    nest_type = state.nest_type
    refine_ratio = to_list(state.refine_ratio)
    parent_tile = to_list(parent_tile)
    nk = "nesting"

    if nest_type == "telescoping":
        log.info("Generating telescoped nested grids")
    else:
        log.info("Generating same-level nested grids")

    out_dir_tmp = out_dir / "tmp_nested"
    out_dir_tmp.mkdir(parents=True, exist_ok=True)

    # Generate global grid first (needed for offsets)
    _ = gen_global_nest_parent(res, out_dir)

    nest_tiles = [7 + i for i in range(n_nests)]

    if len(parent_tile) != n_nests:
        parent_tile = [parent_tile[0]] * n_nests

    for i, tile in enumerate(nest_tiles):
        i_parent_tile = parent_tile[i]
        i_refine_ratio = refine_ratio[i]

        if nest_type == "telescoping":
            i_refine_ratio = np.prod(refine_ratio[: i + 1])

        get_nest_indices(
            res=res,
            tile_idx=i,
            grid_dir=out_dir,
            parent_tile=parent_tile,
            i_refine_ratio=i_refine_ratio,
        )

        istart_nest = ",".join(map(str, state[nk]["istart_nest"]))
        iend_nest = ",".join(map(str, state[nk]["iend_nest"]))
        jstart_nest = ",".join(map(str, state[nk]["jstart_nest"]))
        jend_nest = ",".join(map(str, state[nk]["jend_nest"]))

        cmd = [
            f"{make_hgrid}",
            "--grid_type",
            "gnomonic_ed",
            "--nlon",
            f"{nlon}",
            "--grid_name",
            f"C{res}_grid",
            "--do_schmidt",
            "--stretch_factor",
            f"{stretch_factor}",
            "--target_lon",
            f"{state.target_lon}",
            "--target_lat",
            f"{state.target_lat}",
            "--nest_grid",
            "--parent_tile",
            f"{i_parent_tile}",
            "--refine_ratio",
            f"{i_refine_ratio}",
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

        result, msgs = run_cmd(cmd, log_file=log_file, cwd=out_dir_tmp)

        if result != 0:
            log.error(msgs)
            raise RuntimeError(f"Failed to generate nested grid for tile: {tile}")

        nest_tile = out_dir_tmp / f"C{res}_grid.tile7.nc"

        if tile == 7:
            cp(nest_tile, out_dir / f"C{res}_grid.tile7.nc")
        elif tile > 7:
            cp(nest_tile, out_dir / f"C{res}_grid.tile{tile}.nc")

    shutil.rmtree(out_dir_tmp)

    files = list(out_dir.glob("C*_grid.tile*.nc"))
    for f in files:
        parts = f.name.split("_")
        res_part = str(parts[0])  # e.g., 'C96'
        if res_part != f"C{res}":
            new_name = f.name.replace(res_part, f"C{res}")
            new_path = out_dir / new_name
            rename(f, new_path)

    if nest_type == "telescoping" and gtype == "nest":
        get_nest_tele_indices(res, state.n_nests, refine_ratio, out_dir)

        refine_ratios = ",".join(map(str, refine_ratio))
        istart_nest = ",".join(map(str, state[nk]["istart_nest"]))
        iend_nest = ",".join(map(str, state[nk]["iend_nest"]))
        jstart_nest = ",".join(map(str, state[nk]["jstart_nest"]))
        jend_nest = ",".join(map(str, state[nk]["jend_nest"]))
        parent_tile = ",".join(map(str, state[nk]["parent_tile"]))

        cmd = [
            f"{make_hgrid}",
            "--grid_type",
            "gnomonic_ed",
            "--nlon",
            f"{nlon}",
            "--grid_name",
            f"C{res}_grid",
            "--do_schmidt",
            "--stretch_factor",
            f"{stretch_factor}",
            "--target_lon",
            f"{state.target_lon}",
            "--target_lat",
            f"{state.target_lat}",
            "--nest_grids",
            f"{n_nests}",
            "--parent_tile",
            parent_tile,
            "--refine_ratio",
            refine_ratios,
            "--istart_nest",
            istart_nest,
            "--iend_nest",
            iend_nest,
            "--jstart_nest",
            jstart_nest,
            "--jend_nest",
            jend_nest,
            "--halo",
            f"{halo}",
            "--great_circle_algorithm",
        ]

        telescope_dir = state.tmp / "telescoping"
        telescope_dir.mkdir(parents=True, exist_ok=True)

        result, msgs = run_cmd(cmd, cwd=telescope_dir, log_file=log_file)
        if result != 0:
            log.error(msgs)
            raise RuntimeError("Failed to generate telescoped nested grids")

        shutil.rmtree(out_dir, ignore_errors=True)
        telescope_dir.rename(out_dir)


def make_uniform_grid(make_hgrid, nlon, res):
    log_file = state.logs / "make_uniform_grid.log"
    cmd = [
        f"{make_hgrid}",
        "--grid_type",
        "gnomonic_ed",
        "--nlon",
        f"{nlon}",
        "--grid_name",
        f"C{res}_grid",
        "--do_schmidt",
        "--stretch_factor",
        f"{state.stretch_factor}",
        "--target_lon",
        f"{state.target_lon}",
        "--target_lat",
        f"{state.target_lat}",
        "--great_circle_algorithm",
    ]

    log.info(f"Generating uniform grid: C{res}")
    result, msgs = run_cmd(cmd, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate uniform grid")


def make_stretched_grid(make_hgrid, nlon, res, stretch_factor, target_lon, target_lat):
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
        f"C{res}_grid",
        "--do_schmidt",
        "--stretch_factor",
        f"{stretch_factor}",
        "--target_lon",
        f"{target_lon}",
        "--target_lat",
        f"{target_lat}",
        "--great_circle_algorithm",
    ]

    result, msgs = run_cmd(cmd, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate stretched grid")


def make_regional_gfdl_grid(
    make_hgrid,
    nlon,
    res,
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
):
    log_file = state.logs / "make_regional_gfdl_grid.log"

    cmd = [
        f"{make_hgrid}",
        "--grid_type",
        "gnomonic_ed",
        "--nlon",
        f"{nlon}",
        "--grid_name",
        f"C{res}_grid",
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

    result, msgs = run_cmd(cmd, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate regional GFDL grid")

    grid_file = out_dir / f"C{res}_grid.tile7.nc"

    cmd = [f"{global_equiv_resol}", f"{grid_file}"]

    result, msgs = run_cmd(cmd, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to run global equiv resol")


def make_regional_esg_grid(
    regional_esg_grid,
    target_lon,
    target_lat,
    idim,
    jdim,
    delx,
    dely,
    halo,
    out_dir,
    global_equiv_resol,
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
    result, msgs = run_cmd(cmd, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to run global equiv resol")


def run_make_grid(
    res: int,
    gtype: str,
    exec_dir: Path,
    out_dir: Path,
    stretch_factor: float = None,
    target_lon: float = None,
    target_lat: float = None,
    refine_ratio: int | list[int] = None,
    istart_nest: int | list[int] = None,
    jstart_nest: int | list[int] = None,
    iend_nest: int | list[int] = None,
    jend_nest: int | list[int] = None,
    parent_tile: int | list[int] = 6,
    halo: int = None,
    idim: int = None,
    jdim: int = None,
    delx: float = None,
    dely: float = None,
):
    """
    Generate FV3 grid NetCDF files and a mosaic using FV3GFS grid tools.

    This function serves as a Python wrapper around the `fv3gfs_make_grid.sh`
    workflow, enabling the creation of cubed-sphere grids for the FV3 dynamical
    core. It supports global uniform, stretched, nested, and regional grid
    configurations. The generated grids and mosaics are compatible with FV3
    and UFS workflows.

    Parameters
    ----------
    res : int
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
    """

    regional_esg_grid = exec_dir / "regional_esg_grid"
    make_hgrid = exec_dir / "make_hgrid"
    global_equiv_resol = exec_dir / "global_equiv_resol"
    nlon = res * 2

    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)

    # -------------------------------
    # Grid generation
    # -------------------------------

    if gtype == "uniform":
        make_uniform_grid(make_hgrid, nlon, res)

    elif gtype == "stretch":
        make_stretched_grid(
            make_hgrid, nlon, res, stretch_factor, target_lon, target_lat
        )

    elif gtype == "nest":
        make_nested_grid(
            make_hgrid,
            nlon,
            res,
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
            res,
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

    calc_cpu_alloc(Path(state.tmp / "grid"))
    save_state()
