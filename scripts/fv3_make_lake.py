from pathlib import Path

import xarray as xr
from fv3_runtime import log
from fv3_state import state
from fv3_utils import run_cmd, run_parallel


def _require_var(path: Path, var: str, step: str) -> None:
    """inland and lakefrac stop with exit status 0 on netCDF errors, so their
    success is checked from the variable they add."""
    with xr.open_dataset(path) as ds:
        if var not in ds:
            raise RuntimeError(f"{step} did not add {var} to {path.name}")


def _run_lakefrac(
    workdir: Path,
    c_res: int,
    tile: int,
    topo: Path,
    lake_cutoff: float,
    exec_dir: Path,
    log_file: Path,
) -> None:
    """Add lake_frac and lake_depth to one tile's orography (lakefrac)."""
    cmd = [
        str(Path(exec_dir) / "lakefrac"),
        f"{tile}",
        f"{c_res}",
        f"{topo}",
        f"{lake_cutoff}",
    ]
    result, msgs = run_cmd(cmd, cwd=workdir, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(
            f"Failed to add lake fraction to orography for tile: [{tile}]"
        )
    _require_var(workdir / f"oro.C{c_res}.tile{tile}.nc", "lake_frac", "lakefrac")


def run_add_lakefrac(
    add_lake: bool,
    c_res: int,
    gtype: str,
    exec_dir: Path,
    orog_dir: Path,
    grid_dir: Path,
    topo: Path,
    lake_cutoff: float,
    tmp: Path | None = None,
):
    """
    Python wrapper for fv3_lakefrac.sh.
    Adds inland mask, lake_status, and lake_depth to FV3 orography NetCDFs.

    Parameters
    ----------
    add_lake : bool
        Whether to add lake fraction to orography files.
    c_res : int
        Cubed-sphere resolution (e.g., 96 for C96).
    gtype : str
        Grid type: 'uniform' or 'regional_gfdl'.
    exec_dir : Path
        Directory containing `inland` and `lakefrac` executables.
    orog_dir : Path
        Directory containing orography NetCDF files (oro.C${c_res}.tile*.nc).
    grid_dir : Path
        Directory containing grid NetCDF files (C${c_res}_grid.tile*.nc).
    topo : Path
        Directory containing topographic data inputs.
    lake_cutoff : float
        Threshold for lake fraction processing.
    tmp : Path or None
        Temporary working directory (default: $tmp or /tmp).
    """
    if not add_lake:
        return

    if gtype not in ["uniform", "regional_gfdl"]:
        log.warning(
            f"add_lakefrac is only supported for uniform and regional_gfdl grids, skipping lakefrac generation for gtype: {gtype}"
        )
        return

    workdir = tmp / f"C{c_res}" / "orog" / "tiles"
    workdir.mkdir(parents=True, exist_ok=True)

    # As UFS_UTILS fv3gfs_make_lake.sh: link every tile, build the inland mask
    # once (inland reads and writes all tiles of the mosaic), then add the lake
    # fields tile by tile.
    if gtype == "uniform":
        tiles, mode = list(range(1, 7)), "g"
    else:  # regional_gfdl
        tiles, mode = [7], "r"

    for tile in tiles:
        for src in (
            Path(orog_dir) / f"oro.C{c_res}.tile{tile}.nc",
            Path(grid_dir) / f"C{c_res}_grid.tile{tile}.nc",
        ):
            link = workdir / src.name
            link.unlink(missing_ok=True)
            link.symlink_to(src)

    log_file = state.logs / "add_lakefrac_inland.log"
    cmd = [str(Path(exec_dir) / "inland"), str(c_res), "0.99", "7", mode]
    result, msgs = run_cmd(cmd, cwd=workdir, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate the inland mask")
    for tile in tiles:
        _require_var(workdir / f"oro.C{c_res}.tile{tile}.nc", "inland", "inland")

    args = [
        (
            workdir,
            c_res,
            tile,
            topo,
            lake_cutoff,
            exec_dir,
            state.logs / f"add_lakefrac_tile{tile}.log",
        )
        for tile in tiles
    ]
    # Each lakefrac run writes only its own tile's file.
    run_parallel(_run_lakefrac, args)
