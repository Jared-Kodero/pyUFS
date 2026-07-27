from multiprocessing import Pool
from pathlib import Path

from fv3_runtime import log, tmp_cwd
from fv3_state import state
from fv3_utils import run_cmd


def _run_add_lakefrac(
    workdir: Path,
    c_res: int,
    tile: int,
    gtype: str,
    orog_dir: Path,
    grid_dir: Path,
    topo: Path,
    lake_cutoff: float,
    exec_dir: Path,
    log_file: Path,
):

    lakefrac = Path(exec_dir) / "lakefrac"
    inland = Path(exec_dir) / "inland"

    oro_file = Path(orog_dir) / f"oro.C{c_res}.tile{tile}.nc"
    grid_file = Path(grid_dir) / f"C{c_res}_grid.tile{tile}.nc"
    oro_symlink = Path(workdir / oro_file.name)
    grid_symlink = Path(workdir / grid_file.name)
    oro_symlink.symlink_to(oro_file)
    grid_symlink.symlink_to(grid_file)

    # 1. Create inland mask
    cutoff = 0.99
    rd = 7
    mode = "g" if gtype == "uniform" else "r"
    cmd1 = [str(inland), str(c_res), str(cutoff), str(rd), mode]

    result, msgs = run_cmd(cmd1, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"Failed to generate inland mask for tile: [{tile}]")

    # 2. Add lake fraction to orography files

    oro_file = f"oro.C{c_res}.tile{tile}.nc"
    cmd2 = [
        f"{lakefrac}",
        f"{tile}",
        f"{c_res}",
        f"{topo}",
        f"{lake_cutoff}",
    ]

    result, msgs = run_cmd(cmd2, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(
            f"Failed to add lake fraction to orography for tile: [{tile}]"
        )


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

    with tmp_cwd(workdir):
        # Link required orog + grid files
        if gtype == "uniform":
            tile_beg, tile_end = 1, 6
        else:  # regional_gfdl
            tile_beg = tile_end = 7

        args = [
            (
                workdir,
                c_res,
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
