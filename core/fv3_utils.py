# utils.py

"""Helpers that need no case state.

The launchers load this module on the login node, before any workflow
environment exists, so only the standard library and PyYAML are imported at
the top level; pandas, f90nml and xarray are imported where they are used.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from collections import namedtuple
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import pandas as pd


log = logging.getLogger("PREPROCESS")


# External commands


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
    stdin: IO[str] | None = None,
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


def ufs_utils_version(ufs_bin_dir: object) -> str:
    """UFS_UTILS release recorded by tools/build_ufs.py in <ufs_bin_dir>/manifest.yaml."""
    manifest = Path(os.path.expandvars(str(ufs_bin_dir or ""))) / "manifest.yaml"
    try:
        release = open_yaml(manifest).get("ufs_utils")
    except (OSError, ValueError):
        return f"unknown (no readable {manifest})"
    if not release:
        return f"unknown (no release recorded in {manifest})"
    return f"{release} ({manifest.parent})"


# Files and paths


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


def sort_paths(f: str | Path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", Path(f).name)]


def ic_filename(kind: str, tile: int, gtype: str) -> Path:
    """INPUT file of a tile, ``kind`` gfs, sfc or oro: global <kind>_data.tileN.nc,
    nest <kind>_data.nestNN.tileN.nc, regional <kind>_data.nc (single tile, no suffix)."""
    if gtype in ("regional_gfdl", "regional_esg"):
        return Path(f"{kind}_data.nc")
    nest_idx = f"nest{(tile - 5):02d}." if tile > 6 else ""
    return Path(f"{kind}_data.{nest_idx}tile{tile}.nc")


@contextmanager
def tmp_cwd(path: Path | str):
    cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


# Configuration files: YAML and Fortran namelists


def open_yaml(path: Path | str) -> Any:
    """Parse a YAML file with ``yaml.safe_load``; an empty file gives ``{}``.

    Raises ValueError for a syntax error, with the line and column, and OSError
    when the file cannot be read.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Bad YAML file: {path}\n{error}") from error
    return {} if data is None else data


def resolve_includes(
    conf: dict,
    conf_path: Path | str,
    *,
    stack: frozenset[Path] = frozenset(),
    includes: list[str] | None = None,
) -> dict:
    """Merge the files named by the ``include`` key of ``conf`` into it.

    ``include`` is a path or a list of paths, expanded for ``$VARS`` and ``~``
    and taken relative to the directory of ``conf_path``; it may be nested.
    Keys of ``conf`` win over included keys. Two includes that set the same
    key to different values raise ValueError, as does a circular include. An
    include that does not exist is skipped with a warning. ``conf`` itself is
    not modified. The resolved path of every included file is appended to
    ``includes`` when it is given.
    """
    conf = dict(conf)
    conf_path = Path(os.path.expandvars(str(conf_path))).expanduser().resolve()
    stack |= {conf_path}
    include_paths = conf.pop("include", None)
    if isinstance(include_paths, str):
        include_paths = [include_paths]

    base: dict = {}
    for path in include_paths or []:
        include_path = Path(os.path.expandvars(str(path))).expanduser()
        include_path = (conf_path.parent / include_path).resolve()

        if include_path in stack:
            raise ValueError(f"Circular include of {include_path} in {conf_path}")
        if not include_path.exists():
            log.warning(
                "Include %s of %s does not exist; skipped", include_path, conf_path
            )
            continue
        if includes is not None:
            includes.append(str(include_path))
        included = resolve_includes(
            open_yaml(include_path) or {}, include_path, stack=stack, includes=includes
        )
        conflicts = [k for k in base.keys() & included.keys() if base[k] != included[k]]
        if conflicts:
            raise ValueError(
                f"Key(s) {conflicts} in {include_path} are already set by an earlier include of {conf_path}"
            )
        base.update(included)
    return base | conf


def load_config(
    path: Path | str,
    includes: list[str] | None = None,
    defaults: Path | str | None = None,
) -> dict:
    """Read a configuration file and resolve its includes; keys that are unset are
    then filled from the file ``defaults`` when it is given.

    The resolved paths of the included files are appended to ``includes`` when it
    is given. Raises TypeError when the file is not a mapping at the top level.
    """
    path = Path(path)
    conf = open_yaml(path) or {}
    if not isinstance(conf, dict):
        raise TypeError(
            f"Bad YAML file: {path}\nthe top level must be a mapping of keys to values"
        )
    conf = resolve_includes(conf, path, includes=includes)
    if defaults is not None:
        for key, value in load_config(defaults).items():
            if conf.get(key) is None and value is not None:
                conf[key] = value
    return conf


def dump_yaml(
    data: Any, path: Path | str, *, atomic: bool = True, **dump_kwargs: Any
) -> None:
    """Write ``data`` as YAML; keys keep their order unless ``sort_keys`` is given.

    With ``atomic`` the text goes to a temporary file in the same directory
    that then replaces ``path``, so a reader never sees a half-written file and
    an interrupted write leaves the old file. ``dump_kwargs`` go to
    ``yaml.safe_dump`` (``default_flow_style``, ``allow_unicode``, ...).
    """
    path = Path(path)
    dump_kwargs.setdefault("sort_keys", False)
    target = path.with_name(f".{path.name}.tmp") if atomic else path
    try:
        with target.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, **dump_kwargs)
        if atomic:
            target.replace(path)
    except BaseException:
        if atomic:
            target.unlink(missing_ok=True)
        raise


def to_builtin(obj: object) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj


def to_list(x: object) -> list[Any]:
    return [x] if not isinstance(x, list) else x


def open_nml(path: Path) -> dict:
    import f90nml

    return to_builtin(f90nml.read(path))


def write_nml(nml: dict, path: Path) -> None:
    import f90nml

    with open(path, "w") as f:
        f90nml.write(nml, f)


# Dates and forecast length


def parse_datetime(dt, cycle: bool = True):
    """Parse YYYYMMDDHHZ; with cycle, require a 00, 06, 12 or 18 UTC cycle."""
    import pandas as pd

    dt = pd.to_datetime(dt, format="%Y%m%d%HZ")
    valid_hours = [0, 6, 12, 18]
    if cycle and dt.hour not in valid_hours:
        raise ValueError(
            f"Invalid cycle hour: {dt.hour:02d}Z. Valid GFS cycle times are 00Z, 06Z, 12Z, and 18Z."
        )

    return dt


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

    from itertools import pairwise

    import pandas as pd

    step: dict[str, Any] = {units: length}
    edges = [init + pd.DateOffset(**step) * i for i in range(count + 1)]
    return [int((b - a).total_seconds()) // 3600 for a, b in pairwise(edges)]


# Grid resolution


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


def get_newres(gridfile: Path) -> int:
    """Return the global-equivalent cubed-sphere resolution of a regional grid.

    global_equiv_resol writes this value to the grid file as the global
    attribute RES_equiv (UFS_UTILS, global_equiv_resol.f90). The supergrid
    dimension nx is twice the zonal cell count of the regional domain and bears
    no relation to the equivalent resolution, so it cannot be used in its place.
    """
    import xarray as xr

    with xr.open_dataset(gridfile) as ds:
        res_equiv = ds.attrs.get("RES_equiv", None)

    if res_equiv is None:
        raise ValueError(
            f"{gridfile} has no RES_equiv attribute; run global_equiv_resol first"
        )

    return int(res_equiv)


# Allocated nodes and runtime environment


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
