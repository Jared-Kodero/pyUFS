from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from fv3_pes_config import calc_cpu_alloc
from fv3_runtime import log
from fv3_state import load_fv3_state, save_fv3_state, state
from fv3_utils import run_cmd
from pyproj import Proj


def _wget(url: str, output_path: Path) -> bool:
    """
    Attempt to download a single URL. Returns True on success, False on failure.
    Does not raise; caller is responsible for fallback logic.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["/wget", "-q", "--no-check-certificate", url, "-O", str(output_path)]

    result = 1
    for _ in range(10):
        result, _ = run_cmd(cmd, warn_on_error=False)

        if result == 0:
            if output_path.exists() and output_path.stat().st_size > 0:
                return True

        if output_path.exists():
            output_path.unlink()

        time.sleep(np.random.uniform(0, 5))

    return False


def _download_data(
    urls: list[str],
    output_path: Path,
    external_model: str,
    datetime: str,
    sources: list[str],
) -> None:
    for url, source in zip(urls, sources):
        if _wget(url, output_path):
            log.info(f"Successfully retrieved {external_model} IC data from {source}")
            return external_model
        log.warning(f"Failed to retrieve {external_model} IC data from {source}")

    raise RuntimeError(
        f"All download sources failed for {external_model} at {datetime}.\nAttempted URLs:\n{urls}"
    )


def get_IC(external_model: Literal["GFS", "HRRR"]) -> tuple[str, str]:
    """
    Get initialization data for the specified external model

    URL Sources:
        - GFS: NOAA AWS S3, NCAR GDEX
        - HRRR: NOAA AWS S3, Google Cloud Storage

    Example URL formats:

    https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF => NOAA AWS S3 GFS

    https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfnatfFF.grib2 => NOAA AWS S3 HRRR

    https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d084001/YYYYMMDD/gfs.0p25.YYYYMMDDHH.f000.grib2 => NCAR GDEX

    https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf00.grib2 => Google Cloud Storage HRRR

    """
    datetime = state.init_datetime
    forecast_hour = state.forecast_hour

    date = datetime.strftime("%Y%m%d")
    year = datetime.strftime("%Y")
    hour = datetime.strftime("%H")
    root_dir = state.ic_data / external_model
    root_dir.mkdir(parents=True, exist_ok=True)

    if external_model == "GFS":
        fh_str = str(forecast_hour).zfill(3)
        fh = f"f{fh_str}"

        # Primary: NOAA AWS S3
        noaa_base = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
        noaa_url = f"{noaa_base}/gfs.{date}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.{fh}"

        # Fallback: NCAR GDEX
        # Format: gfs.0p25.{YYYYMMDD}{HH}.f{FFF}.grib2
        ncar_base = "https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d084001"
        ncar_url = f"{ncar_base}/{year}/{date}/gfs.0p25.{date}{hour}.{fh}.grib2"

        local_file = f"GFS.{date}{hour}Z.{fh}.0p25deg.grib2"
        output_path = root_dir / local_file

        if not output_path.exists():
            _download_data(
                urls=[noaa_url, ncar_url],
                sources=["NOAA AWS S3", "NCAR GDEX"],
                output_path=output_path,
                external_model=external_model,
                datetime=datetime,
            )

        return str(root_dir), local_file

    elif external_model == "HRRR":
        fh_str = str(forecast_hour).zfill(2)
        fh = f"f{fh_str}"
        product = f"wrfnat{fh}"

        # Primary: NOAA AWS S3
        noaa_base = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
        noaa_url = f"{noaa_base}/hrrr.{date}/conus/hrrr.t{hour}z.{product}.grib2"

        # Fallback: Google Cloud Storage
        gcs_base = "https://storage.googleapis.com/high-resolution-rapid-refresh"
        gcs_url = f"{gcs_base}/hrrr.{date}/conus/hrrr.t{hour}z.{product}.grib2"

        local_file = f"HRRR.{date}{hour}Z.{fh}.3km.grib2"
        output_path = root_dir / local_file

        if not output_path.exists():
            _download_data(
                urls=[noaa_url, gcs_url],
                sources=["NOAA AWS S3", "Google Cloud Storage"],
                output_path=output_path,
                external_model=external_model,
                datetime=datetime,
            )

        return str(root_dir), local_file

    else:
        raise ValueError(f"Unsupported external model: {external_model}")


def validate_hrrr_bounds(tile: int) -> str:

    geo_hrrr = xr.open_dataset(state.fixed_am / "geo_em.d01.nc_HRRRX")

    # HRRR uses a sphere with radius 6370km usually in WRF/HRRR setups
    proj_hrrr = Proj(
        proj="lcc",
        lat_1=float(geo_hrrr.TRUELAT1),
        lat_2=float(geo_hrrr.TRUELAT2),
        lat_0=float(geo_hrrr.MOAD_CEN_LAT),
        lon_0=float(geo_hrrr.STAND_LON),
        a=6370000.0,
        b=6370000.0,
    )

    # Calculate HRRR domain limits in meters (centered at MOAD_CEN_LAT/LON)
    dx = float(geo_hrrr.DX)
    dy = float(geo_hrrr.DY)
    nx = int(geo_hrrr.sizes["west_east"])
    ny = int(geo_hrrr.sizes["south_north"])

    # HRRR coordinates are typically 0-indexed at center or relative to center
    # In WRF geo_em files, the center of the grid is (0,0) in projection space
    hrrr_x_min = -0.5 * dx * (nx - 1)
    hrrr_x_max = 0.5 * dx * (nx - 1)
    hrrr_y_min = -0.5 * dy * (ny - 1)
    hrrr_y_max = 0.5 * dy * (ny - 1)

    grid = xr.open_dataset(state.tmp / "grid" / f"C{state.res}_grid.tile{tile}.nc")

    grid_lon = grid["x"].values
    grid_lat = grid["y"].values

    grid_lon = ((grid_lon + 180) % 360) - 180

    shield_x, shield_y = proj_hrrr(grid_lon, grid_lat)

    is_contained = (
        (shield_x.min() >= hrrr_x_min)
        and (shield_x.max() <= hrrr_x_max)
        and (shield_y.min() >= hrrr_y_min)
        and (shield_y.max() <= hrrr_y_max)
    )

    geo_hrrr.close()
    grid.close()

    if is_contained:
        return "HRRR"
    return "GFS"


def preprocess_only():
    files_to_rm = []
    for pattern in ["*run.id", "*.out", "shield.native", "*table*"]:
        files_to_rm.extend(state.case_home.glob(pattern))
    subprocess.run(["rm", "-rf", *map(str, files_to_rm)], check=True)
    Path(state.case_home / "ic.only").touch()
    save_fv3_state()


def init_external_ic() -> bool:

    ic_dir = state.external_ic_dir or state.case_home
    ic_at_home = not state.external_ic_dir

    ic_dir = Path(ic_dir)
    required = ("FIXED", "GRID", "IC", "INPUT")
    missing = [
        d
        for d in required
        if not (ic_dir / d).is_dir() or not any((ic_dir / d).iterdir())
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete IC staging in {state.case_home}: {', '.join(missing)} missing or empty"
        )

    if not ic_at_home:
        os.system(f"cp -rf {ic_dir}/* {state.case_home}/")
        log.info(
            f"Copied external IC data from {state.external_ic_dir} to {state.case_home}"
        )
    else:
        log.info(f"IC data was found directly in {state.case_home}")

    load_fv3_state(merge=True)
    calc_cpu_alloc(state.input)

    return True
