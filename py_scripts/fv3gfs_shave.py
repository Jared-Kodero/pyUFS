from pathlib import Path

from fv3gfs_runtime import get_launcher
from fv3gfs_state import state
from fv3gfs_utils import cp, run_cmd


def run_single_shave(
    halo: int,
    idim: int,
    jdim: int,
    res: int,
    tile: int,
    cmd: list,
    tmp_dir: Path,
    tmp_ic_dir: Path,
    log_file: str,
):
    """Run shave once for both orog and grid files at given halo."""
    halo_tag = f"halo{halo}"

    # Prepare input control files
    in_orog = tmp_dir / f"oro.C{res}.tile{tile}.nc"
    in_grid = tmp_dir / f"C{res}_grid.tile{tile}.nc"
    out_orog = tmp_dir / f"oro.C{res}.tile{tile}.shave.nc"
    out_grid = tmp_dir / f"C{res}_grid.tile{tile}.shave.nc"

    # Compose input scripts for shave binary
    in_orog_txt = tmp_dir / f"input.shave.orog.{halo_tag}"
    in_grid_txt = tmp_dir / f"input.shave.grid.{halo_tag}"

    with open(in_orog_txt, "w") as f:
        f.write(f"{idim} {jdim} {halo} '{in_orog}' '{out_orog}'\n")

    with open(in_grid_txt, "w") as f:
        f.write(f"{idim} {jdim} {halo} '{in_grid}' '{out_grid}'\n")

    with open(in_orog_txt, "r") as fin:
        run_cmd(cmd, stdin=fin, cwd=tmp_dir, log_file=log_file)

    with open(in_grid_txt, "r") as fin:
        run_cmd(cmd, stdin=fin, cwd=tmp_dir, log_file=log_file)

    # Copy outputs to final filenames
    out_orog_final = tmp_ic_dir / f"C{res}_oro_data.tile{tile}.{halo_tag}.nc"
    out_grid_final = tmp_ic_dir / f"C{res}_grid.tile{tile}.{halo_tag}.nc"

    cp(out_orog, out_orog_final)
    cp(out_grid, out_grid_final)


def run_shave(
    idim: int,
    jdim: int,
    halo: int,
    halop1: int,
    res: int,
    tile: int,
    exec_dir: Path,
    tmp_dir: Path,
    grid_dir: Path,
    tmp_ic_dir: Path,
):
    """
    Crop FV3 regional grid and orography files using the external `shave` utility.

    This function runs the `shave` executable three times—once each for
    `(halo + 1)`, `halo`, and `halo = 0`—to create successively smaller
    versions of regional FV3 input files. These cropped files are used for
    boundary condition generation, model initialization, and runtime
    configurations that require different halo extents.

    Parameters
    ----------
    idim, jdim : int
        Target compute domain dimensions (without halo points).
    halo : int
        Halo width (number of grid cells surrounding the compute domain),
        typically 3 for FV3 regional applications.
    halop1 : int
        One greater than the halo size (`halo + 1`), used to generate boundary
        condition grids.
    res : int
        Base resolution of the grid (e.g., 384 for C384).
    tile : int
        Tile index for regional domain (usually 7).
    exec_dir : Path
        Path to directory containing the `shave` executable.
    tmp_dir : Path
        Directory containing filtered grid/orography files to be cropped.
    grid_dir : Path
        Directory containing the mosaic and grid definition files.
    tmp_ic_dir : Path
        Output directory where the shaved files will be written.
    """

    log_file = state.logs / "shave.log"
    shave = exec_dir / "shave"
    cmd = [get_launcher(1), f"{shave}"]

    # ----------------------------------------------------------------------------
    # Run three shave passes: halo+1, halo, and halo=0
    # ----------------------------------------------------------------------------
    run_single_shave(halop1, idim, jdim, res, tile, cmd, tmp_dir, tmp_dir, log_file)
    run_single_shave(halo, idim, jdim, res, tile, cmd, tmp_dir, tmp_ic_dir, log_file)
    run_single_shave(0, idim, jdim, res, tile, cmd, tmp_dir, tmp_ic_dir, log_file)

    for mosaic in grid_dir.glob(f"C{res}_*mosaic.nc"):
        cp(mosaic, tmp_dir)
