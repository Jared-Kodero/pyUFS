#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

from fv3gfs_runtime import log
from fv3gfs_state import state
from fv3gfs_utils import cp


def run_ice_blend(
    ims_file: Path,
    five_min_file: Path,
    five_min_mask: Path,
    blend_exec: Path,
    blended_file: Path,
    wgrib2: str = "wgrib2",
    cnvgrib: str = "cnvgrib",
    copygb: str = "copygb",
    copygb2: str = "copygb2",
    f: Path | None = None,
    sendcom: bool = False,
    tmp_ic_dir: Path | None = None,
    verbose: bool = True,
    n_nests: int = 0,  # NEW: number of nests for multinest support
    nest_idx: int | None = None,  # NEW: current nest index being processed
):
    """
    Wrapper for the emcsfc_ice_blend program.

    Parameters
    ----------
    ims_file : Path
        IMS ice cover data (grib1 or grib2).
    five_min_file : Path
        5-minute global ice concentration (grib2).
    five_min_mask : Path
        Land/sea mask of the 5-minute file (grib2).
    blend_exec : Path
        Path to `emcsfc_ice_blend` executable.
    blended_file : Path
        Output blended ice concentration file (grib1 for GFS).
    wgrib2, cnvgrib, copygb, copygb2 : str
        External fv3gfs_utilities (must be available in PATH or given as full paths).
    workdir : Path
        Working directory (temporary). Defaults to CWD.
    sendcom : bool
        If True, copy blended file to `comout`.

    verbose : bool
        If True, print diagnostic commands.
    """

    f = Path(f)

    # Create unique log file for each nest to avoid overwriting
    if nest_idx is not None and nest_idx > 0:
        log_file = state.logs / f"ice_blend_nest{nest_idx:02d}.log"
        # Create nest-specific output file name to avoid overwriting
        blended_file = Path(
            str(blended_file).replace(".grb", f"_nest{nest_idx:02d}.grb")
        )
    else:
        log_file = state.logs / "ice_blend.log"
    f.mkdir(parents=True, exist_ok=True)
    os.chdir(f)

    # -------------------------------------------------------------------------
    # Step 1: IMS input check + convert to grib2 if needed
    # -------------------------------------------------------------------------
    if not Path(ims_file).exists():
        raise FileNotFoundError(f"IMS ice file missing: {ims_file}")

    # check if grib1
    result = subprocess.run(
        [wgrib2, "-Sec0", str(ims_file)],
        capture_output=True,
        text=True,
        stderr=log_file,
    )
    if "grib1 message" in result.stdout:
        subprocess.run(
            [cnvgrib, "-g12", "-p40", str(ims_file), "ims.grib2"],
            check=True,
            stdout=log_file,
            stderr=log_file,
        )
    else:
        cp(ims_file, "ims.grib2")

        subprocess.run(
            [wgrib2, "ims.grib2", "-match", "ICEC", "-grib", "ims.icec.grib2"],
            check=True,
            stdout=log_file,
            stderr=log_file,
        )

    grid173 = "0 0 0 0 0 0 0 0 4320 2160 0 0 89958000 42000 48 -89958000 359958000 83000 83000 0"

    subprocess.run(
        [
            copygb2,
            "-x",
            "-i3",
            "-g",
            grid173,
            "ims.icec.grib2",
            "ims.icec.5min.grib2",
        ],
        check=True,
        stdout=log_file,
        stderr=log_file,
    )

    # -------------------------------------------------------------------------
    # Step 2: check EMC/MMAB 5-min file
    # -------------------------------------------------------------------------
    if not Path(five_min_file).exists():
        raise FileNotFoundError(f"MMAB 5-min ice data missing: {five_min_file}")

    # -------------------------------------------------------------------------
    # Step 3: run blend program
    # -------------------------------------------------------------------------
    env = os.environ.copy()
    env["FORT17"] = str(five_min_mask)
    env["FORT11"] = "ims.icec.5min.grib2"
    env["FORT15"] = str(five_min_file)
    env["FORT51"] = str(blended_file)

    if verbose:
        log.debug(f"Running {blend_exec} with FORT51={blended_file}")

    subprocess.run(
        [str(blend_exec)], check=True, env=env, stdout=log_file, stderr=log_file
    )

    # -------------------------------------------------------------------------
    # Step 4: postprocess (convert to grib1, fix bitmap)
    # -------------------------------------------------------------------------

    subprocess.run(
        [
            wgrib2,
            "-set_int",
            "3",
            "51",
            "42000",
            f"{blended_file}",
            "-grib",
            f"{blended_file}.corner",
        ],
        check=True,
        stdout=log_file,
        stderr=log_file,
    )

    subprocess.run(
        [cnvgrib, "-g21", f"{blended_file}.corner", f"{blended_file}.bitmap"],
        check=True,
        stdout=log_file,
        stderr=log_file,
    )
    os.remove(blended_file)

    subprocess.run(
        [copygb, "-M", "#1.57", "-x", f"{blended_file}.bitmap", str(blended_file)],
        check=True,
        stdout=log_file,
        stderr=log_file,
    )

    comout = tmp_ic_dir / "ice_blend"

    if sendcom and comout:
        comout.mkdir(parents=True, exist_ok=True)
        cp(blended_file, comout)

    # cleanup
    for f in [f"{blended_file}.corner", f"{blended_file}.bitmap"]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    if verbose:
        log.info(f"Ice blend completed successfully: {blended_file}")

    # ice_file = state.get("ice_file", None)
    # if ice_file:
    #     emcsfc_ice_inputs = get_func_signature(run_ice_blend)
    #     emcsfc_ice_inputs = {
    #         k: v for k, v in state.items() if k in emcsfc_ice_inputs and v is not None
    #     }
    #     run_ice_blend(
    #         **emcsfc_ice_inputs,
    #     )
