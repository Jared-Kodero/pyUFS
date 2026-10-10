from pathlib import Path

import dask
from fv3_runtime import log, require_fix_files, staged_files, tmp_cwd
from fv3_state import state
from fv3_utils import cp, run_cmd


def _run_make_orog_gsl(
    make_gsl_orog: bool,
    c_res: int,
    tile: int,
    halo: int,
    grid_dir: Path,
    out_dir: Path,
    topo_dir: Path,
    exec_dir: Path,
    tmp: Path | None = None,
    local_state: dict | None = None,
):

    if not make_gsl_orog:
        return

    state.update(local_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = state.logs / f"make_orog_gsl_tile{tile}.log"

    workdir = tmp / f"C{c_res}" / "orog" / f"tile{tile}"
    workdir.mkdir(parents=True, exist_ok=True)

    # Executable
    orog_gsl = exec_dir / "orog_gsl"

    # OUTGRID name depends on halo
    if halo == -999:
        out_grid = f"C{c_res}_grid.tile{tile}.nc"
    else:
        out_grid = f"C{c_res}_grid.tile{tile}.halo{halo}.nc"

    # Inputs linked into the work directory: {link name: source}.
    links = {
        out_grid: grid_dir / out_grid,
        "HGT.Beljaars_filtered.lat-lon.30s_res.nc": topo_dir
        / "HGT.Beljaars_filtered.lat-lon.30s_res.nc",
        "geo_em.d01.lat-lon.2.5m.HGT_M.nc": topo_dir
        / "geo_em.d01.lat-lon.2.5m.HGT_M.nc",
    }

    require_fix_files([*links.values(), orog_gsl])

    # Work in temporary directory
    with tmp_cwd(workdir):
        for name, src in links.items():
            link = workdir / name
            link.unlink(missing_ok=True)
            link.symlink_to(src)

        cp(orog_gsl, ".")

        # Write grid_info.dat
        with open("grid_info.dat", "w") as f:
            f.write(f"{tile}\n{c_res}\n{halo}\n")

        with open("grid_info.dat", "r") as fin:
            cmd = [f"{orog_gsl}"]
            result, msgs = run_cmd(cmd, stdin=fin, stdout=log_file, stderr=log_file)

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
    c_res: int,
    tiles: list[int],
    halo: int,
    grid_dir: Path,
    out_dir: Path,
    topo_dir: Path,
    exec_dir: Path,
    tmp: Path | None = None,
    mod_dir: Path | None = None,
):
    """
    Python wrapper for fv3_orog_gsl.sh functionality.
    Runs `orog_gsl` to generate oro_data static topographic files
    for the GSL orographic drag suite.

    Parameters
    ----------
    make_gsl_orog : bool
        Whether to make GSL orography files.
    c_res : int
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
    mod_dir : Path or None
        Directory containing pre-existing orography files. If provided,
        the function will use these files instead of generating new ones.
    """

    if staged_files(mod_dir):
        return  # copied with the staged orography by run_make_orog

    args = [
        (
            make_gsl_orog,
            c_res,
            tile,
            halo,
            grid_dir,
            out_dir,
            topo_dir,
            exec_dir,
            tmp,
            state.to_dict(),
        )
        for tile in tiles
    ]

    tasks = [dask.delayed(_run_make_orog_gsl)(*a) for a in args]
    return list(
        dask.compute(
            *tasks,
            scheduler=state.preprocess_dask_scheduler,
            num_workers=len(args),
            chunksize=1,
        )
    )
