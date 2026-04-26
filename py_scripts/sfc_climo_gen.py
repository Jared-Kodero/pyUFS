import os
from pathlib import Path

import f90nml
from fv3gfs_runtime import get_launcher, log
from fv3gfs_state import state
from fv3gfs_utils import cp, run_cmd


def _run_single_sfc_climo(
    name: str,
    res: int,
    sfc_climo_gen: Path,
    input_sfc_climo_dir: Path,
    orog_dir_mdl: Path,
    mosaic_file_mdl: Path,
    tmp_dir: Path,
    out_dir: Path,
    halo: int,
    vegsoilt_frac: bool,
    veg_type_src: str,
    soil_type_src: str,
    n_cpus: int,
    orog_files: list[str],
    grid_type: str,
    log_file: Path | None = None,
):
    """
    Internal helper to run a single instance of sfc_climo_gen.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "config": {
            "input_facsf_file": f"{input_sfc_climo_dir}/facsf.1.0.nc",
            "input_substrate_temperature_file": f"{input_sfc_climo_dir}/substrate_temperature.gfs.0.5.nc",
            "input_maximum_snow_albedo_file": f"{input_sfc_climo_dir}/maximum_snow_albedo.0.05.nc",
            "input_snowfree_albedo_file": f"{input_sfc_climo_dir}/snowfree_albedo.4comp.0.05.nc",
            "input_slope_type_file": f"{input_sfc_climo_dir}/slope_type.1.0.nc",
            "input_soil_type_file": f"{input_sfc_climo_dir}/soil_type.{soil_type_src}.nc",
            "input_soil_color_file": f"{input_sfc_climo_dir}/soil_color.clm.0.05.nc",
            "input_vegetation_type_file": f"{input_sfc_climo_dir}/vegetation_type.{veg_type_src}.nc",
            "input_vegetation_greenness_file": f"{input_sfc_climo_dir}/vegetation_greenness.0.144.nc",
            "mosaic_file_mdl": f"{mosaic_file_mdl}",
            "orog_dir_mdl": f"{orog_dir_mdl}",
            "orog_files_mdl": orog_files,
            "halo": halo,
            "maximum_snow_albedo_method": "bilinear",
            "snowfree_albedo_method": "bilinear",
            "vegetation_greenness_method": "bilinear",
            "fract_vegsoil_type": vegsoilt_frac,
        }
    }

    # Verify all required input files exist
    required_files = [
        "input_facsf_file",
        "input_substrate_temperature_file",
        "input_maximum_snow_albedo_file",
        "input_snowfree_albedo_file",
        "input_slope_type_file",
        "input_soil_type_file",
        "input_soil_color_file",
        "input_vegetation_type_file",
        "input_vegetation_greenness_file",
        "mosaic_file_mdl",
    ]
    for key in required_files:
        f = Path(config_dict["config"][key])
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    for f in [orog_dir_mdl / fn for fn in orog_files] + [mosaic_file_mdl]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    # Write Fortran namelist
    fort41 = tmp_dir / "fort.41"
    with open(fort41, "w") as f:
        f90nml.write(config_dict, f)

    cmd = [*get_launcher(n_cpus), f"{sfc_climo_gen}"]
    result, msgs = run_cmd(cmd, cwd=tmp_dir, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate sfc climatology")

    for f in tmp_dir.glob("*.nc"):
        if grid_type == "regional":
            if f.name.endswith(".halo.nc"):
                stem = f.stem.replace(".halo", "")
                dest = out_dir / f"C{res}.{stem}.halo{halo}.nc"

            else:
                dest = out_dir / f"C{res}.{f.stem}.halo0.nc"
        else:
            dest = out_dir / f"C{res}.{f.name}"

        cp(f, dest)


def run_sfc_climo_gen(
    res: int,
    input_sfc_climo_dir: Path,
    exec_dir: Path,
    tmp_dir: Path,
    out_dir: Path,
    fix_dir: Path,
    grid_type: str,
    mosaic_dir: Path,
    orog_dir: Path,
    halo: int = 0,
    vegsoilt_frac: bool = False,
    veg_type_src: str = "modis.igbp.0.05",
    soil_type_src: str = "statsgo.0.05",
    n_cpus: int = 6,
    n_nests: int = 0,
):
    """
    Generate surface climatology fields for FV3 global, nested, or regional grids.

    This function wraps the `sfc_climo_gen` executable, which interpolates
    high-resolution surface datasets (e.g., vegetation type, soil type, and
    albedo) onto the FV3 grid. These fields are used in land surface model
    initialization and serve as fixed input to FV3-based model configurations
    such as the UFS Weather Model or GEFS.

    Parameters
    ----------
    res : int
        Cubed-sphere grid resolution (e.g., 96 for C96). Used to identify input
        and output filenames.
    input_sfc_climo_dir : Path
        Directory containing source climatology datasets (e.g., MODIS, STATSGO).
        This typically includes global 0.05° files for vegetation and soil type.
    exec_dir : Path, optional
        Directory containing the `sfc_climo_gen` executable. If not provided,
        assumes it is available in the system `PATH`.
    tmp_dir : Path, optional
        Temporary working directory for intermediate files.
    out_dir : Path, optional
        Directory where the generated surface climatology NetCDF files will be
        written. If not provided, defaults to the current working directory.
    fix_dir : Path, optional
        Path to a “fix” directory containing FV3 static resources
        (e.g., `sfc_climo_gen.nml`, mosaic templates).
    grid_type : str, default="NULL"
        Grid type identifier. Must be one of:
        - `'uniform'`
        - `'stretch'`
        - `'nest'`
        - `'regional_gfdl'`
        - `'regional_esg'`
        If `"NULL"`, the program infers the grid type automatically.
    mosaic_dir : Path, optional
        Directory containing the grid mosaic file (`C{res}_mosaic.nc`).
    orog_dir : Path, optional
        Directory containing orography files (`oro.C{res}.tile*.nc`).
    halo : int, default=0
        Halo width used when generating extended surface climatology fields.
    vegsoilt_frac : bool, default=False
        If True, generates fractional vegetation and soil type cover fields.
    veg_type_src : str, default="modis.igbp.0.05"
        Vegetation type dataset name (e.g., `"modis.igbp.0.05"` or `"modis.umd.0.05"`).
    soil_type_src : str, default="statsgo.0.05"
        Soil type dataset name (e.g., `"statsgo.0.05"` or `"faosoil.0.05"`).
    n_cpus : int, default=6
        Number of MPI processes or threads to use during execution.
    n_nests : int, default=0
        Number of nested domains. If greater than zero, the routine will run
        `sfc_climo_gen` separately for each nest tile (7, 8, …).
    """

    log.info("Generating surface climatology")

    sfc_climo_gen = exec_dir / "sfc_climo_gen"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = state.logs / "sfc_climo_gen.log"

    if not orog_dir:
        orog_dir = fix_dir / "fix_fv3gfs_gmted2010" / f"C{res}"

    local_cpus = len(os.sched_getaffinity(0))

    norm_cpu = (local_cpus // 6) * 6
    n_cpus = min(60, norm_cpu)

    # --- Determine orography and mosaic files ---
    if grid_type in ["regional_gfdl", "regional_esg", "regional"]:
        orog_files = [f"oro.C{res}.tile7.nc"]
        mosaic_file = mosaic_dir / f"C{res}_mosaic.nc"

        _run_single_sfc_climo(
            "regional",
            res,
            sfc_climo_gen,
            input_sfc_climo_dir,
            orog_dir,
            mosaic_file,
            tmp_dir,
            out_dir,
            halo,
            vegsoilt_frac,
            veg_type_src,
            soil_type_src,
            n_cpus,
            orog_files,
            "regional",
            log_file,
        )

    elif grid_type == "nest":
        # Run for global tiles 1–6 first
        orog_files = [f"oro.C{res}.tile{i}.nc" for i in range(1, 7)]
        mosaic_file = mosaic_dir / f"C{res}_coarse_mosaic.nc"
        _run_single_sfc_climo(
            "global",
            res,
            sfc_climo_gen,
            input_sfc_climo_dir,
            orog_dir,
            mosaic_file,
            tmp_dir / "global",
            out_dir,
            halo,
            vegsoilt_frac,
            veg_type_src,
            soil_type_src,
            n_cpus,
            orog_files,
            "uniform",
            log_file,
        )

        # Now run for nested tile(s)
        nested_tiles = [7 + i for i in range(n_nests or 1)]
        nest_indices = [f"{i:02d}" for i in range(2, state.n_nests + 2)]

        args = [
            (
                f"tile_{tile}",
                res,
                sfc_climo_gen,
                input_sfc_climo_dir,
                orog_dir,
                mosaic_dir / f"C{res}_nested{nest_id}_mosaic.nc",
                tmp_dir / f"tile{tile}",
                out_dir,
                halo,
                vegsoilt_frac,
                veg_type_src,
                soil_type_src,
                n_cpus,
                [f"oro.C{res}.tile{tile}.nc"],
                "nest",
                log_file,
            )
            for (
                tile,
                nest_id,
            ) in zip(nested_tiles, nest_indices)
        ]

        for arg in args:
            _run_single_sfc_climo(*arg)

    else:
        # Uniform global (tiles 1–6)
        orog_files = [f"oro.C{res}.tile{i}.nc" for i in range(1, 7)]
        mosaic_file = mosaic_dir / f"C{res}_mosaic.nc"
        _run_single_sfc_climo(
            "global",
            res,
            sfc_climo_gen,
            input_sfc_climo_dir,
            orog_dir,
            mosaic_file,
            tmp_dir,
            out_dir,
            halo,
            vegsoilt_frac,
            veg_type_src,
            soil_type_src,
            n_cpus,
            orog_files,
            "uniform",
            log_file,
        )
