import os
from pathlib import Path

import f90nml

from fv3gfs_runtime import log
from fv3gfs_state import state
from fv3gfs_utils import cp, run_cmd


def run_filter_topo(
    res: int,
    gtype: str,
    exec_dir: Path,
    grid_dir: Path,
    orog_dir: Path,
    tmp_dir: Path,
    stretch_factor: float | None = None,
):
    """
    Apply topographic filtering to FV3 orography fields.

    This function wraps the `fv3gfs_filter_topo.sh` workflow, which invokes
    the `filter_topo` executable to smooth high-resolution orography fields
    produced by `make_orog`. The filtering step reduces steep gradients
    and small-scale noise in surface topography, improving model stability
    and consistency across cubed-sphere tile boundaries.

    Parameters
    ----------
    res : int
        Cubed-sphere resolution (e.g., 96 for a C96 grid). Used for file naming
        and in filter control parameters.
    gtype : str
        Grid type. Must be one of:
        - `'uniform'` : Global uniform cubed-sphere grid.
        - `'stretch'` : Global stretched grid centered at a target.
        - `'nest'` : Nested cubed-sphere grid.
        - `'regional_gfdl'` : GFDL-style regional configuration.
        - `'regional_esg'` : ESG-style regional configuration.
    exec_dir : Path
        Directory containing the `filter_topo` executable.
    grid_dir : Path
        Directory containing the grid definition NetCDF files
        (e.g., `C{res}_grid.tile*.nc` and `C{res}_mosaic.nc`).
    orog_dir : Path
        Directory containing the unfiltered orography files
        (e.g., `oro.C{res}.tile*.nc`).
    tmp_dir : Path
        Output directory for filtered topography files. Typically used as the
        intermediate or final “IC” directory for FV3 input.
    stretch_factor : float or None, optional
        Stretching factor applied to the grid (required if `gtype='stretch'`).


    """
    log.info("Filtering topography for global tiles")
    # Create unique log file for each nest tile to avoid overwriting

    log_file = state.logs / "filter_topo.log"

    filter_topo = Path(exec_dir) / "filter_topo"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(tmp_dir)

    # Processing all tiles (uniform/stretch) or coarse tiles only
    mosaic_grid = f"C{res}_mosaic.nc"
    # grid_files = f"C{res}_grid.tile[1-6].nc"
    # topo_files = f"oro.C{res}.tile[1-6].nc"
    grid_files = [f"C{res}_grid.tile{t}.nc" for t in range(1, 7)]
    topo_files = [f"oro.C{res}.tile{t}.nc" for t in range(1, 7)]
    topo_file = f"oro.C{res}"

    # Copy mosaic file
    mosaic_src = grid_dir / mosaic_grid
    if mosaic_src.exists():
        cp(mosaic_src, ".")
    else:
        cp(grid_dir / f"C{res}_mosaic.nc", ".")

    # Copy grid and orography files
    for f in grid_files:
        cp(grid_dir / f, ".")
    for f in topo_files:
        cp(orog_dir / f, ".")

    cp(filter_topo, ".")

    # Decide stretch factor
    if gtype in ["stretch", "regional_gfdl"]:
        stretch = stretch_factor
    else:
        stretch = 1.0

    # Regional flag
    regional = gtype in ["regional_gfdl", "regional_esg"]

    # Write namelist - use appropriate mosaic file for namelist
    nml_mosaic = mosaic_grid if mosaic_grid.endswith(".nc") else f"C{res}_mosaic.nc"
    nml = {
        "filter_topo_nml": {
            "grid_file": nml_mosaic,
            "topo_file": topo_file,
            "mask_field": "land_frac",
            "regional": regional,
            "stretch_fac": stretch,
            "res": res,
        }
    }
    with open("input.nml", "w") as f:
        f90nml.write(nml, f)

    cmd = [f"{filter_topo}"]

    result, msgs = run_cmd(cmd, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Filtering topography failed")
