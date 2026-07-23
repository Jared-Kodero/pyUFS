# utils.py

import logging
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from fv3_paths import paths

log = logging.getLogger("PREPROCESS")


def exit_code(code: int = 0) -> None:
    """Write the exit code to a file in the work directory"""
    (paths["work_dir"] / "exit_code").write_text(str(code))


@contextmanager
def redirect_streams(
    stdout: Path | None = None,
    stderr: Path | None = None,
):
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    out_file = None
    err_file = None

    try:
        if stdout == stderr and stdout is not None:
            out_file = open(stdout, "a")
            err_file = out_file

        else:
            if stdout:
                out_file = open(stdout, "a")

            if stderr:
                err_file = open(stderr, "a")

        if out_file:
            sys.stdout = out_file

        if err_file:
            sys.stderr = err_file

        yield out_file, err_file

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        if out_file:
            out_file.close()

        if err_file and err_file is not out_file:
            err_file.close()


def run_cmd(
    cmd: list[str],
    *,
    stdin: object = None,
    cwd: Path | None = None,
    env: dict | None = None,
    stdout: Path | None = None,
    stderr: Path | None = None,
    msgs: str | None = None,
    **kwargs,
) -> tuple[int, str]:

    log_path = stdout or stderr

    if not msgs:
        msgs = f"See full log at {log_path}" if log_path else ""

    try:
        with redirect_streams(stdout, stderr) as (_stdout, _stderr):
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                stdin=stdin,
                cwd=cwd,
                env=env,
                stdout=_stdout,
                stderr=_stderr,
            )

        return result.returncode, ""

    except subprocess.CalledProcessError as exc:
        if kwargs.get("warn_on_error", True):
            log.warning("Command failed: %s", " ".join(cmd))

        return exc.returncode, f"{type(exc).__name__}: {exc}\n{msgs}"

    except Exception as exc:
        log.warning("Exception running command: %s", " ".join(cmd))
        return 1, f"{type(exc).__name__}: {exc}\n{msgs}"


def rename(src: str | Path, dest: str | Path):
    log_file = paths["logs"] / "rename_files.log"

    src = Path(src).resolve()
    dest = Path(dest).resolve()

    if src == dest:
        return

    if dest.exists():
        dest.unlink()

    cmd = ["mv", "-v", str(src), str(dest)]
    result, msgs = run_cmd(cmd, stdout=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"Failed to rename file: {src} to {dest}")


def cp(src: str | Path, dest: str | Path):
    if isinstance(src, list):
        raise TypeError("src must be a single path, not a list.")

    log_file = paths["logs"] / "copy_files.log"

    src = Path(src).resolve()
    dest = Path(dest).resolve()

    cmd = ["cp", "-v", "-rf", str(src), str(dest)]
    result, msgs = run_cmd(cmd, stdout=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"Failed to copy file: {src} to {dest}")


def clear_dir(directory: str | Path) -> None:
    """
    Remove all files and directories in the specified directory.
    """
    path = Path(directory)

    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def env_setup():
    """
    Set up environment variables for UFS_UTILS_DIR execution.
    """
    python_path = str(Path(sys.executable).resolve().parent)
    openmpi_bin = "/opt/openmpi/bin"
    bin_paths = "/usr/local/bin:/usr/bin:/bin"
    sys_path = os.environ.get("PATH")
    os.environ["PATH"] = f"{openmpi_bin}:{python_path}:{bin_paths}:{sys_path}"


def parse_datetime(dt):
    dt = pd.to_datetime(dt, format="%Y%m%d%HZ")
    valid_hours = [0, 6, 12, 18]
    if dt.hour not in valid_hours:
        log.error(
            f"Invalid GFS cycle hour: {dt.hour:02d}Z. Valid GFS cycle times are 00Z, 06Z, 12Z, and 18Z."
        )
        exit_code(1)
        sys.exit(1)

    return dt


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


def format_forecast_length(nhours: int) -> str:
    """Convert forecast length in hours to a readable string.

    Months are approximated as 30 days.
    """
    hours_per_day = 24
    hours_per_month = 30 * hours_per_day

    months, remainder = divmod(nhours, hours_per_month)
    days, hours = divmod(remainder, hours_per_day)

    parts = []

    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours or not parts:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")

    return " ".join(parts)


def _read_required_env_int(name: str) -> int:
    value = os.getenv(name)

    if value is None:
        raise RuntimeError(f"Required environment variable {name} is not set.")

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name} must be an integer, got {value!r}."
        ) from exc


def runtime_env_vars() -> dict[str, object]:
    """Read scheduler and case metadata from the runtime environment."""
    case_name = os.getenv("CASE_NAME")

    values: dict[str, object] = {
        "case_name": case_name,
        "n_cpus": _read_required_env_int("CASE_NTASKS"),
        "n_nodes": int(os.getenv("CASE_NNODES", "1")),
        "ensemble_id": int(os.getenv("CASE_ENSEMBLE_ID", "0")),
        "n_ensembles": int(os.getenv("CASE_ENSEMBLES", "0")),
        "n_cpus_per_node": _read_required_env_int("CASE_NTASKS_PER_NODE"),
        "multi_node": bool(int(os.getenv("CASE_MULTI_NODE_FLAG", "0"))),
        "resubmit": int(os.getenv("CASE_RESUBMIT_MAX", "0")),
        "resubmit_idx": int(os.getenv("CASE_RESUBMIT_INDEX", "0")),
    }

    return {key: value for key, value in values.items() if value is not None}


def require_minimum_cpus(minimum: int = 32) -> int:
    """Validate the number of CPUs visible to the current process."""
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1

    if available < minimum:
        raise RuntimeError(
            f"Insufficient CPUs for this run. Detected {available} available, but at least {minimum} per node is required."
        )

    return available
