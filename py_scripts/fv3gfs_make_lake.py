import os
from multiprocessing import Pool
from pathlib import Path

from fv3gfs_runtime import log
from fv3gfs_state import state
from fv3gfs_utils import run_cmd


def _run_add_lakefrac(
    workdir, res, tile, gtype, orog_dir, grid_dir, topo, lake_cutoff, exec_dir, log_file
):

    lakefrac = Path(exec_dir) / "lakefrac"
    inland = Path(exec_dir) / "inland"

    oro_file = Path(orog_dir) / f"oro.C{res}.tile{tile}.nc"
    grid_file = Path(grid_dir) / f"C{res}_grid.tile{tile}.nc"
    oro_symlink = Path(workdir / oro_file.name)
    grid_symlink = Path(workdir / grid_file.name)
    oro_symlink.symlink_to(oro_file)
    grid_symlink.symlink_to(grid_file)

    # 1. Create inland mask
    cutoff = 0.99
    rd = 7
    mode = "g" if gtype == "uniform" else "r"
    cmd1 = [str(inland), str(res), str(cutoff), str(rd), mode]

    result, msgs = run_cmd(cmd1, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"Failed to generate inland mask for tile: [{tile}]")

    # 2. Add lake fraction to orography files

    oro_file = f"oro.C{res}.tile{tile}.nc"
    cmd2 = [
        f"{lakefrac}",
        f"{tile}",
        f"{res}",
        f"{topo}",
        f"{lake_cutoff}",
    ]

    result, msgs = run_cmd(cmd2, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(
            f"Failed to add lake fraction to orography for tile: [{tile}]"
        )


def run_add_lakefrac(
    add_lake: bool,
    res: int,
    gtype: str,
    exec_dir: Path,
    orog_dir: Path,
    grid_dir: Path,
    topo: Path,
    lake_cutoff: float,
    tmp: Path | None = None,
):
    """
    Python wrapper for fv3gfs_lakefrac.sh.
    Adds inland mask, lake_status, and lake_depth to FV3 orography NetCDFs.

    Parameters
    ----------
    add_lake : bool
        Whether to add lake fraction to orography files.
    res : int
        Cubed-sphere resolution (e.g., 96 for C96).
    gtype : str
        Grid type: 'uniform' or 'regional_gfdl'.
    exec_dir : Path
        Directory containing `inland` and `lakefrac` executables.
    orog_dir : Path
        Directory containing orography NetCDF files (oro.C${res}.tile*.nc).
    grid_dir : Path
        Directory containing grid NetCDF files (C${res}_grid.tile*.nc).
    topo : Path
        Directory containing topographic data inputs.
    lake_cutoff : float
        Threshold for lake fraction processing.
    tmp : Path or None
        Temporary working directory (default: $tmp or /tmp).
    """
    if not add_lake:
        return None

    if gtype not in ["uniform", "regional_gfdl"]:
        raise NotImplementedError(
            f"lakefrac only implemented for uniform and regional_gfdl, not {gtype}."
        )

    workdir = tmp / f"C{res}" / "orog" / "tiles"
    workdir.mkdir(parents=True, exist_ok=True)

    os.chdir(workdir)
    log.debug(f"workdir = {workdir}\noutdir = {orog_dir}\nindir = {topo}")

    # Link required orog + grid files
    if gtype == "uniform":
        tile_beg, tile_end = 1, 6
    else:  # regional_gfdl
        tile_beg = tile_end = 7

    args = [
        (
            workdir,
            res,
            tile,
            gtype,
            orog_dir,
            grid_dir,
            topo,
            lake_cutoff,
            exec_dir,
            state.logs / f"add_lakefrac_tile{tile}.log",
        )
        for tile in range(tile_beg, tile_end + 1)
    ]

    with Pool(processes=len(args)) as pool:
        pool.starmap(_run_add_lakefrac, args)
