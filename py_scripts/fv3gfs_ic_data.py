from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Literal

import xarray as xr
from fv3gfs_cpu_config import calc_cpu_alloc
from fv3gfs_runtime import log
from fv3gfs_stage_data import update_table_files
from fv3gfs_state import FV3State, load_state, prev_state, save_state, state
from fv3gfs_utils import run_cmd
from pyproj import Proj

# Matches ens / ENS / ensemble / ENSEMBLE / mem / MEM / member / MEMBER
# followed by exactly two digits. Group 1 captures the digits.
_ENS_DIR_PATTERN = re.compile(
    r"(?:ens(?:emble)?|mem(?:ber)?)(\d{2})",
    flags=re.IGNORECASE,
)
_ENS_PREFIXES = ("ens", "ENS", "ensemble", "ENSEMBLE", "mem", "MEM", "member", "MEMBER")


def merge_states():
    """Merge current and previos states"""

    load_state()
    new_state = FV3State({**prev_state, **state})
    state.update(new_state)
    save_state()


def ic_only():
    files_to_rm = []
    for pattern in ["*run.id", "*.out", "shield.native", "*table*"]:
        files_to_rm.extend(state.home.glob(pattern))
    subprocess.run(["rm", "-rf", *map(str, files_to_rm)], check=True)
    Path(state.home / "ic.only").touch()
    save_state()


def _unpack_case_tarball(src: Path) -> Path:

    unpack_name = src.parent.name
    unpack_root = state.home / "case.tar.gz"
    unpack_dir = unpack_root / unpack_name

    shutil.rmtree(unpack_dir, ignore_errors=True)
    unpack_dir.mkdir(parents=True, exist_ok=True)

    _tarball = unpack_root / src.name
    shutil.copy2(src, _tarball)

    log.info(f"IC source tarball: {src}")
    log.info(f"Unpacking IC source tarball to: {unpack_dir}")

    try:
        with tarfile.open(_tarball, mode="r:*") as tf:
            tf.extractall(unpack_dir)
    finally:
        _tarball.unlink(missing_ok=True)

    return unpack_dir


def _is_unpacked_case(path: Path) -> bool:
    src_ic = path / "IC" / "INPUT"
    return path.is_dir() and src_ic.is_dir() and any(src_ic.iterdir())


def _validate_ic_case(path: Path) -> None:
    """
    Validate that a resolved IC source is an unpacked case containing IC files.
    """

    src_ic = path / "IC" / "INPUT"

    if not path.exists():
        raise FileNotFoundError(
            f"Specified IC source case path does not exist:\n{path}"
        )

    if not path.is_dir():
        raise ValueError(f"Specified IC source case path is not a directory:\n{path}")

    if not src_ic.exists():
        raise FileNotFoundError(
            f"Specified IC source case is not a valid unpacked case.\nExpected IC files under:\n\t{src_ic}"
        )

    if not src_ic.is_dir() or not any(src_ic.iterdir()):
        raise ValueError(
            f"Specified IC source case has no initial condition files.\nExpected files under:\n\t{src_ic}"
        )


def _resolve_case_dir_or_tarball(src: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Specified IC source case path does not exist:\n{src}")

    if src.is_file():
        if not tarfile.is_tarfile(str(src)):
            raise ValueError(f"Specified IC source file is not a tarball:\n{src}")

        return _unpack_case_tarball(src)

    if not src.is_dir():
        raise ValueError(f"Specified IC source case path is not a directory:\n{src}")

    if _is_unpacked_case(src):
        return src

    case_tarball = src / "case.tar.gz"

    if case_tarball.is_file():
        if not tarfile.is_tarfile(str(case_tarball)):
            raise ValueError(
                f"Specified IC source file is not a tarball:\n{case_tarball}"
            )

        return _unpack_case_tarball(case_tarball)

    raise FileNotFoundError(
        f"Specified IC source is not a valid unpacked case and does not contain case.tar.gz:\n{src}"
    )


def _resolve_ic_source_path(src: Path) -> tuple[Path, bool]:

    ens_id_str = f"{state.ensemble_id:02d}"

    if not src.exists():
        raise FileNotFoundError(f"Specified IC source case path does not exist:\n{src}")

    if not state.ensemble_run:
        resolved = _resolve_case_dir_or_tarball(src)
        _validate_ic_case(resolved)
        return resolved

    if src.is_file():
        match = _ENS_DIR_PATTERN.fullmatch(src.parent.name)
        if match is None:
            msg = "Expected IC source tarball for ensemble run to be under a directory named "
            msg = msg + f"{{{'/'.join(_ENS_PREFIXES)}}} followed by 2 digits, "
            msg = msg + f"got {src.parent.name}"
            raise ValueError(msg)

        if state.paired_ensembles and match.group(1) != ens_id_str:
            raise ValueError(
                f"Expected IC source tarball under ensemble id {ens_id_str}, got {src.parent.name}"
            )

        resolved = _resolve_case_dir_or_tarball(src)
        _validate_ic_case(resolved)
        return resolved

    if not src.is_dir():
        raise ValueError(f"Specified IC source case path is not a directory:\n{src}")

    src_match = _ENS_DIR_PATTERN.fullmatch(src.name)

    if src_match is not None:
        if state.paired_ensembles and src_match.group(1) != ens_id_str:
            raise ValueError(
                f"Expected IC source directory ensemble id {ens_id_str}, got {src.name}"
            )
        candidate_paths = [src]
    else:
        candidate_paths = [src]
        for child in sorted(src.iterdir()):
            if not child.is_dir():
                continue
            m = _ENS_DIR_PATTERN.fullmatch(child.name)
            if m is not None and m.group(1) == ens_id_str:
                candidate_paths.append(child)

    errors: list[str] = []
    for candidate in candidate_paths:
        try:
            resolved = _resolve_case_dir_or_tarball(candidate)
            _validate_ic_case(resolved)
            return resolved
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{candidate}: {exc}")

    raise FileNotFoundError(
        "No valid IC source case found. Tried:\n" + "\n".join(errors)
    )


def init_external_ic() -> None:
    """
    Copy IC data from an existing case into the working directory,
    preserving symlinks and mimicking `cp -rf` semantics.
    """

    src = state.get("ic_source_path", None)
    src = Path(src) if src is not None else None
    ic_src_type = state.get("ic_source_type", "case").lower()
    if ic_src_type not in ["case", "external"]:
        raise ValueError(
            f"Invalid IC source type: {ic_src_type}. Expected 'case' or 'external'."
        )

    if not src:
        merge_states()
        return

    log.info("Skipping Grid and IC generation; using existing files")
    log.info(f"Using IC data from case: {state.get('ic_source_path')}")

    if ic_src_type == "case":
        src = _resolve_ic_source_path(src)
        src_ic = src / "IC" / "INPUT"
    else:
        src_ic = Path(src)

    log.info(f"Resolved IC source directory: {src_ic}")

    if not src.exists() or not any(src.iterdir()):
        msg = f"Directory is empty: {src.resolve()}\n"
        msg = msg + "No initial condition files detected"
        msg = msg + f"Ensure files are placed in:\n\t{src_ic}"
        raise ValueError(msg)

    subprocess.run(["cp", "-rf", f"{src}/.", f"{state.home}/"], check=True)

    for d in ["INPUT", "HIST", "RESTART", "LOGS"]:
        p = state.home / d
        subprocess.run(["rm", "-rf", str(p)], check=True)
        p.mkdir(parents=True, exist_ok=True)

    subprocess.run(["cp", "-rf", f"{src_ic}/.", f"{state.home}/INPUT/"], check=True)

    dirs_to_rm = list((state.home / "IC").glob("R*"))
    dirs_to_rm = dirs_to_rm + [state.home / "IC" / "INPUT"]

    files_to_rm = []
    for pattern in ["*run.id", "*.out", "shield.native", "*table*"]:
        files_to_rm.extend(state.home.glob(pattern))

    subprocess.run(["rm", "-rf", *map(str, dirs_to_rm)], check=True)
    subprocess.run(["rm", "-rf", *map(str, files_to_rm)], check=True)
    subprocess.run(["rm", "-rf", str(state.home / "case.tar.gz")], check=True)

    run_id = os.environ.get("CURR_RUN_ID", "0")
    (state.home / "run.id").write_text(str(run_id))

    merge_states()

    update_table_files()
    calc_cpu_alloc(Path(state.input))


def _wget(url: str, output_path: Path) -> bool:
    """
    Attempt to download a single URL. Returns True on success, False on failure.
    Does not raise; caller is responsible for fallback logic.
    """
    cmd = ["/wget", "-q", "--no-check-certificate", url, "-O", str(output_path)]
    result, _ = run_cmd(cmd)
    if result != 0:
        if output_path.exists():
            output_path.unlink()
        return False
    return True


def _download_data(
    urls: list[str],
    output_path: Path,
    external_model: str,
    datetime,
) -> None:
    for url in urls:
        if _wget(url, output_path):
            return

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
    root_dir = state.home / "IC" / external_model
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
                output_path=output_path,
                external_model=external_model,
                datetime=datetime,
            )

        return str(root_dir), local_file

    else:
        raise ValueError(f"Unsupported external model: {external_model}")


def validate_hrrr_bounds(tile: int) -> str:

    geo_hrrr = xr.open_dataset(state.fix_am / "geo_em.d01.nc_HRRRX")

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
