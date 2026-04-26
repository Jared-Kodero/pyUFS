import logging
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pandas as pd
from fv3gfs_paths import paths

log = logging.getLogger("UFS_UTILS")


def run_cmd(
    cmd: list, *, stdin=None, cwd=None, env=None, log_file=None, msgs=None
) -> tuple[int, str]:

    out_file = open(log_file, "a") if log_file else None
    if not msgs:
        msgs = f"See full log at {log_file}" if log_file else ""

    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdin=stdin,
            cwd=cwd,
            env=env,
            stdout=out_file,
            stderr=out_file,
        )

        if result.returncode != 0:
            error = f"Error running command:\n\t{' '.join(cmd)}\n"
            error = f"{error}\n\t{msgs}"
        else:
            error = ""

        return result.returncode, error
    finally:
        if out_file:
            out_file.close()


def rename(src, dest):
    log_file = paths["logs"] / "rename_files.log"

    src = Path(src).resolve()
    dest = Path(dest).resolve()

    cmd = ["mv", "-v", str(src), str(dest)]
    result, msgs = run_cmd(cmd, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"Failed to rename file: {src} to {dest}")


def cp(src, dest):
    if isinstance(src, list):
        raise TypeError("src must be a single path, not a list.")

    log_file = paths["logs"] / "copy_files.log"

    src = Path(src).resolve()
    dest = Path(dest).resolve()

    cmd = ["cp", "-v", "-rf", str(src), str(dest)]
    result, msgs = run_cmd(cmd, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"Failed to copy file: {src} to {dest}")


def env_setup():
    """
    Set up environment variables for UFS_UTILS execution.
    """
    python_path = str(Path(sys.executable).resolve().parent)
    openmpi_bin = "/opt/openmpi/bin"
    bin_paths = "/usr/local/bin:/usr/bin:/bin"
    sys_path = os.environ.get("PATH")
    os.environ["PATH"] = f"{openmpi_bin}:{python_path}:{bin_paths}:{sys_path}"


def parse_datetime(input_args):
    datetime_str = input_args["init_datetime"]

    try:
        dt = pd.to_datetime(datetime_str, format="%Y%m%d%HZ")
    except ValueError as exc:
        try:
            dt = pd.to_datetime(datetime_str)
        except ValueError:
            raise ValueError('Invalid cdate format. Expected "%Y%m%d%HZ".') from exc

    valid_hours = [0, 6, 12, 18]
    if dt.hour not in valid_hours:
        raise ValueError(
            f"Invalid GFS cycle hour: {dt.hour:02d}Z. Valid GFS cycle times are 00Z, 06Z, 12Z, and 18Z."
        )
    input_args["init_datetime"] = dt
    return input_args


def cres_to_deg(C):
    """Convert C-resolution to grid spacing in km and degrees."""
    deg_mapping = {
        96: 1.0,
        192: 0.5,
        384: 0.25,
        768: 0.12,
        1152: 0.08,
        3072: 0.03,
    }
    earth_circumference = 40075.0
    face_length_km = earth_circumference / 4.0  # ≈ 10018.75 km
    dx_km = face_length_km / C
    km_per_deg = 111.2
    if C in deg_mapping:
        ddeg = deg_mapping[C]
    else:
        ddeg = dx_km / km_per_deg
    Resolution = namedtuple("Resolution", ["C", "km", "deg"])
    return Resolution(C, round(dx_km, 2), round(ddeg, 2))


def km_to_cres(dx_km):
    """Convert grid spacing in km to nearest UFS-recommended C-resolution."""
    earth_circumference = 40075.0
    face_length_km = earth_circumference / 4.0
    C_exact = int(face_length_km / dx_km)
    C = int(96 * round(C_exact / 96))
    return C


def deg_to_cres(ddeg):
    """Convert grid spacing in degrees to nearest UFS-recommended C-resolution."""

    km_per_deg = 111.2
    dx_km = ddeg * km_per_deg
    C = km_to_cres(dx_km)

    return C


def parse_resolution(in_str):

    if in_str is None:
        return None

    in_str = str(in_str).strip().upper()
    in_str = "".join(in_str.split())

    if not in_str.startswith("C"):
        raise ValueError(
            f"Invalid resolution format: {in_str}. Expected one of (C48, C96, C192, C384, C768, C1152, C3072)"
        )

    num = in_str.replace("C", "")
    try:
        c_res = int(num)
    except ValueError:
        raise ValueError(f"Invalid C-resolution format: {in_str}")

    valid_cres = (48, 96, 192, 384, 768, 1152, 3072)

    if c_res not in valid_cres:
        raise ValueError(
            f"Unsupported C-resolution: {c_res}. Supported values are: {valid_cres}"
        )

    return c_res
