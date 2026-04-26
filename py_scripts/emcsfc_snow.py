import subprocess
from pathlib import Path

import f90nml

from fv3gfs_runtime import log
from fv3gfs_state import state
from fv3gfs_utils import run_cmd


def run_emcsfc_snow(
    ims_file: Path,
    exec_dir: Path,
    fix: Path,
    tmp_ic_dir: Path,
    afwa_nh_file: str = "NPR.SNWN.SP.S1200.MESH16",
    afwa_sh_file: str = "NPR.SNWS.SP.S1200.MESH16",
    afwa_global_file: str = "",
    model_slmask_file: str = "global_slmask.t1534.3072.1536.grb",
    model_lat_file: str = "global_latitudes.t1534.3072.1536.grb",
    model_lon_file: str = "global_longitudes.t1534.3072.1536.grb",
    gfs_lpl_file: str = "global_lonsperlat.t1534.3072.1536.txt",
    climo_qc: str | None = None,
    model_snow_file: str = "snogrb_model",
    output_grib2: bool = False,
    wgrib2: str = "/wgrib2",
    wgrib: str = "wgrib",
    sendcom: bool = False,
    n_nests: int = 0,  # NEW: number of nests for multinest support
    nest_idx: int | None = None,  # NEW: current nest index being processed
):
    """
    Python wrapper for emcsfc_snow.sh.

    Generates model snow analysis from IMS and AFWA input data
    by running `emcsfc_snow2mdl`.

    Parameters
    ----------
    ims_file : Path
        IMS snow cover file (grib2).
    exec_dir : Path
        Path to executables (must contain emcsfc_snow2mdl).
    fix : Path
        Path to fixed files (contains snow climatology).
    outdir : Path
        Directory for work and outputs.
    afwa_* : str
        AFWA snow data file names (default: ops standard).
    model_* : str
        Model grid definition files (lat/lon/lsmask).
    gfs_lpl_file : str
        Reduced grid definition (optional).
    climo_qc : str
        Snow cover climatology file. Default: fix_am/emcsfc_snow_cover_climo.grib2.
    model_snow_file : str
        Output file name.
    output_grib2 : bool
        Write output in grib2 (default False = grib1).
    wgrib2, wgrib : str
        Paths to grib fv3gfs_utilities.
    sendcom : bool
        If True, copy outputs to comout.
    comout : Path
        Directory for COMOUT copy.
    """

    # Create unique log file for each nest to avoid overwriting
    if nest_idx is not None and nest_idx > 0:
        log_file = state.logs / f"emcsfc_snow_nest{nest_idx:02d}.log"
        # Also create nest-specific output file name to avoid overwriting
        model_snow_file = f"snogrb_model_nest{nest_idx:02d}"
    else:
        log_file = state.logs / "emcsfc_snow.log"

    snow_exec = exec_dir / "emcsfc_snow2mdl"

    climo_qc = climo_qc or str(fix / "am" / "emcsfc_snow_cover_climo.grib2")

    # 1. IMS quick check
    try:
        subprocess.run([wgrib2, str(ims_file)], check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"IMS file {ims_file} appears corrupt; aborting.")

    # 2. Extract IMS valid time
    # Prefer wgrib2 -t (grib2), fall back to wgrib -v (grib1)
    try:
        out = subprocess.check_output([wgrib2, "-t", str(ims_file)], text=True)
        tempdate = out.splitlines()[0]
        imsd = tempdate.split("d=")[-1]
    except Exception:
        out = subprocess.check_output(
            [wgrib, "-v", str(ims_file)], text=True, stderr=log_file
        )
        tempdate = out.splitlines()[0]
        imsd = tempdate.split("D=")[-1]

    imsyear = int(imsd[0:4])
    imsmonth = int(imsd[4:6])
    imsday = int(imsd[6:8])
    imshour = 0  # convention

    # 3. Write fort.41 namelist
    fort41 = {
        "source_data": {
            "autosnow_file": "",
            "nesdis_snow_file": str(ims_file),
            "nesdis_lsmask_file": "",
            "afwa_snow_global_file": afwa_global_file,
            "afwa_snow_nh_file": afwa_nh_file,
            "afwa_snow_sh_file": afwa_sh_file,
            "afwa_lsmask_nh_file": "",
            "afwa_lsmask_sh_file": "",
        },
        "qc": {
            "climo_qc_file": climo_qc,
        },
        "model_specs": {
            "model_lat_file": model_lat_file,
            "model_lon_file": model_lon_file,
            "model_lsmask_file": model_slmask_file,
            "gfs_lpl_file": gfs_lpl_file,
        },
        "output_data": {
            "model_snow_file": f"./{model_snow_file}",
            "output_grib2": output_grib2,
        },
        "output_grib_time": {
            "grib_year": imsyear,
            "grib_month": imsmonth,
            "grib_day": imsday,
            "grib_hour": imshour,
        },
        "parameters": {
            "lat_threshold": 55.0,
            "min_snow_depth": 0.05,
            "snow_cvr_threshold": 50.0,
        },
    }

    nml_file = tmp_ic_dir / "fort.41"
    with open(nml_file, "w") as f:
        f90nml.write(fort41, f, force=True)

    # 4. Run program

    comout = tmp_ic_dir / "emcsfc_snow"
    comout.mkdir(parents=True, exist_ok=True)

    result, msgs = run_cmd([str(snow_exec)], cwd=comout, log_file=log_file)

    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to run emcsfc_snow2mdl")

    log.info(f"emcsfc_snow2mdl completed output staged in: {comout}")

    # ims_file = state.get("ims_file", None)
    # if ims_file:
    #     emcsfc_inputs = get_func_signature(run_emcsfc_snow)

    #     emcsfc_inputs = {
    #         k: v for k, v in state.items() if k in emcsfc_inputs and v is not None
    #     }
    #     run_emcsfc_snow(
    #         **emcsfc_inputs,
    #     )
