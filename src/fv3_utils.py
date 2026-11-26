# utils.py

import logging
import os
import re
import shutil
import subprocess
import sys
from collections import namedtuple
from contextlib import ExitStack, contextmanager
from itertools import pairwise
from pathlib import Path

import pandas as pd

from fv3_paths import paths

log = logging.getLogger("PREPROCESS")


def find_tool(name: str) -> str:
    """An external program: on PATH, in the bin directory of the running
    Python environment (the preprocess image's conda environment), or linked
    at / as the preprocess image links wgrib2."""
    found = shutil.which(name)
    if found:
        return found
    for path in (
        Path(os.path.realpath(sys.executable)).parent / name,
        Path("/") / name,
    ):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise FileNotFoundError(
        f"{name} not found on PATH, in {Path(sys.executable).parent} or at /"
    )


def exit_code(code: int = 0) -> None:
    """Write the exit code to a file in the case directory"""
    (paths["case_dir"] / "exit_code").write_text(str(code))


@contextmanager
def redirect_streams(
    stdout: Path | None = None,
    stderr: Path | None = None,
):
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with ExitStack() as stack:
        out_file = stack.enter_context(open(stdout, "a")) if stdout else None
        if stderr is not None and stderr == stdout:
            err_file = out_file
        else:
            err_file = stack.enter_context(open(stderr, "a")) if stderr else None

        try:
            if out_file:
                sys.stdout = out_file
            if err_file:
                sys.stderr = err_file
            yield out_file, err_file
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_cmd(
    cmd: list[str],
    *,
    stdin: object = None,
    cwd: Path | None = None,
    env: dict | None = None,
    stdout: Path | None = None,
    stderr: Path | None = None,
    msgs: str | None = None,
    warn_on_error: bool = True,
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
        if warn_on_error:
            log.warning("Command failed: %s", " ".join(map(str, cmd)))

        return exc.returncode, f"{type(exc).__name__}: {exc}\n{msgs}"

    except OSError as exc:  # executable missing or not runnable
        log.warning("Exception running command: %s", " ".join(map(str, cmd)))
        return 1, f"{type(exc).__name__}: {exc}\n{msgs}"


def remove_path(*targets: str | Path) -> None:
    """Remove files, symlinks and directory trees; a missing path is ignored (rm -rf)."""
    for target in map(Path, targets):
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)


def move_path(src: str | Path, dest: str | Path):
    """Move src to dest, replacing an existing dest (mv -f); works across filesystems."""
    src = Path(src).resolve()
    dest = Path(dest).resolve()

    if src == dest:
        return

    try:
        remove_path(dest)
        shutil.move(src, dest)
    except OSError as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        raise RuntimeError(f"Failed to move file: {src} to {dest}") from exc
    log.debug("renamed %s -> %s", src, dest)


def _copy_force(src: str | Path, dest: str | Path) -> None:
    """Copy a file, replacing a read-only destination (cp -f)."""
    try:
        shutil.copy(src, dest, follow_symlinks=False)
    except PermissionError:
        if not os.access(src, os.R_OK) or not Path(dest).exists():
            raise
        Path(dest).unlink()
        shutil.copy(src, dest, follow_symlinks=False)


def copy_path(src: str | Path, dest: str | Path):
    """Copy a file or directory tree (cp -rf).

    Into an existing directory the copy is made inside it, under the name of
    src. Symlinks inside a tree stay symlinks; modes are kept, modification
    times are not.
    """
    if isinstance(src, list):
        raise TypeError("src must be a single path, not a list.")

    src = Path(src).resolve()
    dest = Path(dest).resolve()
    if dest.is_dir():
        dest = dest / src.name

    try:
        if src.is_dir():
            shutil.copytree(
                src, dest, symlinks=True, dirs_exist_ok=True, copy_function=_copy_force
            )
        else:
            _copy_force(src, dest)
    except OSError as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        raise RuntimeError(f"Failed to copy file: {src} to {dest}") from exc
    log.debug("copied %s -> %s", src, dest)


def parse_datetime(dt, cycle: bool = True):
    """Parse YYYYMMDDHHZ; with cycle, require a 00, 06, 12 or 18 UTC cycle."""
    dt = pd.to_datetime(dt, format="%Y%m%d%HZ")
    valid_hours = [0, 6, 12, 18]
    if cycle and dt.hour not in valid_hours:
        log.error(
            f"Invalidcycle hour: {dt.hour:02d}Z. Valid GFS cycle times are 00Z, 06Z, 12Z, and 18Z."
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


def parse_resolution(c_res: str | int | None) -> int | None:
    """Cubed-sphere face resolution as an integer, from a label such as "C96".

    run_config.yaml gives c_res as a string that starts with C (C48, C96, C192,
    C384, C768, C1152, C3072). A plain integer such as 96 is still accepted.
    The rest of the workflow uses the integer.
    """

    if c_res is None:
        return None

    valid_cres = (48, 96, 192, 384, 768, 1152, 3072)
    labels = ", ".join(f"C{v}" for v in valid_cres)

    if isinstance(c_res, bool):
        raise TypeError(f"c_res must be a string such as C96; got {c_res!r}")
    if isinstance(c_res, str):
        match = re.fullmatch(r"\s*[Cc](\d+)\s*", c_res)
        if not match:
            raise ValueError(f"c_res must start with C, for example C96; got {c_res!r}")
        value = int(match.group(1))
    elif isinstance(c_res, int):
        value = c_res
    else:
        raise TypeError(f"c_res must be a string such as C96; got {c_res!r}")

    if value not in valid_cres:
        raise ValueError(f"c_res {c_res!r} not supported; use one of {labels}")

    return value


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


def segment_hours(init: pd.Timestamp, length: int, units: str, count: int) -> list[int]:
    """Hours in each of `count` segments of `length` `units` starting at `init`.

    Months follow the calendar as in the SHiELD coupler (pandas Gregorian equals
    the FMS julian calendar for 1901-2099).
    """
    if units not in ("hours", "days", "months", "years"):
        raise ValueError("run_length_units must be hours, days, months or years")
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError(f"run_length must be a positive integer, got {length!r}")
    edges = [init + pd.DateOffset(**{units: length * i}) for i in range(count + 1)]
    return [(b - a) // pd.Timedelta(hours=1) for a, b in pairwise(edges)]


def get_allocated_nodes() -> list[str]:
    """Return the hostnames allocated to the job."""
    return os.getenv("CASE_NODE_LIST", "").split()


def runtime_env_vars() -> dict[str, object]:
    """Read scheduler and case metadata from the runtime environment."""
    case_name = os.getenv("CASE_NAME")

    values: dict[str, object] = {
        "case_name": case_name,
        "n_cpus": int(os.getenv("CASE_NTASKS", "0")),
        "n_nodes": int(os.getenv("CASE_NNODES", "1")),
        "ensemble_id": int(os.getenv("CASE_ENSEMBLE_ID", "0")),
        "n_ensembles": int(os.getenv("CASE_ENSEMBLES", "0")),
        "n_cpus_per_node": int(os.getenv("CASE_NTASKS_PER_NODE", "0")),
        "multi_node": bool(int(os.getenv("CASE_MULTI_NODE_FLAG", "0"))),
        "resubmit": int(os.getenv("CASE_RESUBMIT_MAX", "0")),
        "resubmit_idx": int(os.getenv("CASE_RESUBMIT_INDEX", "0")),
        "resume_run": bool(int(os.getenv("CASE_CONTINUE_RUN", "0"))),
        "python": os.getenv("PYUFS_PYTHON") or None,
    }

    return {key: value for key, value in values.items() if value is not None}


def require_minimum_cpus() -> int:
    """Validate the number of CPUs visible to the current process."""

    minimum_node_cpus = 6
    available = len(os.sched_getaffinity(0))

    if available < minimum_node_cpus:
        raise RuntimeError(
            f"Insufficient CPUs for this run. Detected {available} available, but at least {minimum_node_cpus} per node is required."
        )

    return available
