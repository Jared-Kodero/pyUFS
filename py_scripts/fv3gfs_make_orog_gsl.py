import os
from multiprocessing import Pool
from pathlib import Path

from fv3gfs_runtime import log
from fv3gfs_state import FV3State, state
from fv3gfs_utils import cp, run_cmd


def _run_make_orog_gsl(
    make_gsl_orog: bool,
    res: int,
    tile: int,
    halo: int,
    grid_dir: Path,
    out_dir: Path,
    topo_dir: Path,
    exec_dir: Path,
    tmp: Path | None = None,
    local_state: dict = None,
):

    if not make_gsl_orog:
        return

    local_state = FV3State(local_state)

    state.update(local_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = state.logs / f"make_orog_gsl_tile{tile}.log"

    workdir = tmp / f"C{res}" / "orog" / f"tile{tile}"
    workdir.mkdir(parents=True, exist_ok=True)

    # Executable
    orog_gsl = exec_dir / "orog_gsl"

    # OUTGRID name depends on halo
    if halo == -999:
        out_grid = f"C{res}_grid.tile{tile}.nc"
    else:
        out_grid = f"C{res}_grid.tile{tile}.halo{halo}.nc"

    # Work in temporary directory
    os.chdir(workdir)

    # Symlinks to required inputs
    (workdir / out_grid).symlink_to(grid_dir / out_grid)
    (workdir / "HGT.Beljaars_filtered.lat-lon.30s_res.nc").symlink_to(
        topo_dir / "HGT.Beljaars_filtered.lat-lon.30s_res.nc"
    )
    (workdir / "geo_em.d01.lat-lon.2.5m.HGT_M.nc").symlink_to(
        topo_dir / "geo_em.d01.lat-lon.2.5m.HGT_M.nc"
    )

    cp(orog_gsl, ".")

    # Write grid_info.dat
    with open("grid_info.dat", "w") as f:
        f.write(f"{tile}\n{res}\n{halo}\n")

    log.debug("Running orog_gsl with grid_info.dat:")
    log.debug(Path("grid_info.dat").read_text())

    with open("grid_info.dat", "r") as fin:
        cmd = [f"{orog_gsl}"]
        result, msgs = run_cmd(cmd, stdin=fin, log_file=log_file)

    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"Failed to run orog_gsl for tile: [{tile}]")

    # Move outputs
    for nc in workdir.glob("C*oro_data_*.nc"):
        cp(nc, f"{out_dir}/")

    log.info(f"ORO_DATA FILES CREATED IN: {out_dir}")
    return list(out_dir.glob("C*oro_data_*.nc"))


def run_make_orog_gsl(
    make_gsl_orog: bool,
    res: int,
    tiles: list[int],
    halo: int,
    grid_dir: Path,
    out_dir: Path,
    topo_dir: Path,
    exec_dir: Path,
    tmp: Path | None = None,
):
    """
    Python wrapper for fv3gfs_orog_gsl.sh functionality.
    Runs `orog_gsl` to generate oro_data static topographic files
    for the GSL orographic drag suite.

    Parameters
    ----------
    make_gsl_orog : bool
        Whether to make GSL orography files.
    res : int
        Cubed-sphere resolution (e.g., 96 for C96).
    tile : list[int]
        Tile number (1-6 for global cube-sphere, 7 for nest).
    halo : int
        Lateral boundary halo size. Use -999 if no halo file.
    grid_dir : Path
        Directory containing grid NetCDF files.
    out_dir : Path
        Output directory for oro_data NetCDF files.
    topo_dir : Path
        Directory containing topographic datasets
        (HGT.Beljaars_filtered..., geo_em...).
    exec_dir : Path
        Directory containing the `orog_gsl` executable.
    tmp : Path or None
        Temporary working directory (default: $tmp or /tmp).
    """

    args = [
        (
            make_gsl_orog,
            res,
            tile,
            halo,
            grid_dir,
            out_dir,
            topo_dir,
            exec_dir,
            tmp,
            dict(state),
        )
        for tile in tiles
    ]

    with Pool(processes=len(args)) as pool:
        return pool.starmap(_run_make_orog_gsl, args)
