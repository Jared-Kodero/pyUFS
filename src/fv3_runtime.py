# runtime.py
import logging
import os
import re
import shutil
import sys
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import f90nml
import xarray as xr

from fv3_paths import paths
from fv3_state import state
from fv3_update_fix import ensure_fix_file, missing_message
from fv3_utils import parse_resolution
from fv3_yaml import dump_yaml, load_config, open_yaml, resolve_includes
from release import resolve_release

log = logging.getLogger("PREPROCESS")


def ufs_utils_version(ufs_bin_dir: object) -> str:
    """UFS_UTILS release recorded by tools/build_ufs.py in <ufs_bin_dir>/manifest.yaml."""
    manifest = Path(os.path.expandvars(str(ufs_bin_dir or ""))) / "manifest.yaml"
    try:
        release = read_namelist(manifest).get("ufs_utils")
    except (OSError, ValueError):
        return f"unknown (no readable {manifest})"
    if not release:
        return f"unknown (no release recorded in {manifest})"
    return f"{release} ({manifest.parent})"


def log_release() -> None:
    """Log the pyUFS branch, the UFS_UTILS release and the SHiELD build of this case.

    Called before any other driver output of the initial run (not on restarts,
    which would repeat it in every segment). The branch is read from .git/HEAD of
    the pyUFS checkout (no git binary is needed inside the containers) and the
    shield_exe with the shield_release from the run configuration, or the container
    image when shield_use_container is true.
    """
    branch = "unknown"
    try:
        head = (Path(paths["pyufs"]) / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            branch = head.removeprefix("ref: refs/heads/")
        else:
            branch = f"detached at {head[:12]}"
    except OSError:
        pass

    config = load_run_config()
    if config.get("preprocess_use_container"):
        image = config.get("preprocess_image")
        name = Path(str(image)).name if image else "(preprocessing)"
        ufs_utils = f"container image {name}"
    else:
        ufs_utils = ufs_utils_version(config.get("ufs_bin_dir"))

    if config.get("shield_use_container"):
        image = config.get("shield_image")
        name = Path(str(image)).name if image else "(SHiELD)"
        version = f"container image {name}"
    else:
        shield_exe = os.path.expandvars(str(config.get("shield_exe") or "")).strip()
        version = f"{Path(shield_exe).name} (release: {config['shield_release']})"

    log.info("pyUFS branch: %s", branch)
    log.info("UFS_UTILS version: %s", ufs_utils)
    log.info("SHiELD version: %s", version)


def get_newres(gridfile: Path) -> int:
    """Return the global-equivalent cubed-sphere resolution of a regional grid.

    global_equiv_resol writes this value to the grid file as the global
    attribute RES_equiv (UFS_UTILS, global_equiv_resol.f90). The supergrid
    dimension nx is twice the zonal cell count of the regional domain and bears
    no relation to the equivalent resolution, so it cannot be used in its place.
    """
    with xr.open_dataset(gridfile) as ds:
        res_equiv = ds.attrs.get("RES_equiv", None)

    if res_equiv is None:
        raise ValueError(
            f"{gridfile} has no RES_equiv attribute; run global_equiv_resol first"
        )

    return int(res_equiv)


def load_run_config() -> dict:
    conf_path = Path(paths["submit_dir"]) / "run_config.yaml"
    conf = read_namelist(conf_path) or {}
    release = resolve_release(conf.get("shield_release"))
    release_defaults = load_config(Path(paths["configs"]) / "run_config.yaml")

    conf = resolve_includes(conf, conf_path)

    for key, value in release_defaults.items():
        if conf.get(key) is None and value is not None:
            conf[key] = value
    conf["shield_release"] = release
    return conf


# run_config.yaml keys that determine the grid and its orography. Files staged
# in IC/grid or IC/orography are reused only while these are unchanged.
GRID_KEYS = (
    "c_res",
    "gtype",
    "stretch_factor",
    "target_lon",
    "target_lat",
    "refine_ratio",
    "parent_tile",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
    "halo",
    "idim",
    "jdim",
    "delx",
    "dely",
)
# The staged orography also depends on whether the GSL files were made.
OROGRAPHY_KEYS = (*GRID_KEYS, "make_gsl_orog")
STAGED_RECORD = ".grid_settings.yaml"


def _grid_settings(keys: tuple = OROGRAPHY_KEYS) -> dict:
    cfg = load_run_config()
    settings = {k: to_builtin(cfg.get(k)) for k in keys}
    if "c_res" in settings:
        # "C96" and 96 name the same grid; staged records store the integer.
        try:
            settings["c_res"] = parse_resolution(settings["c_res"])
        except (TypeError, ValueError):
            pass
    return settings


def staged_files(mod_dir: Path | None) -> list[Path]:
    """Files to reuse from IC/grid or IC/orography, or [] to generate them.

    The directories hold the output of a preprocess_grid_only or
    preprocess_orog_only run, possibly edited, or files supplied by the user.
    Raises when they were staged for different grid settings, since a grid
    of another resolution, box or grid type would be reused silently.
    """
    if mod_dir is None or not Path(mod_dir).is_dir():
        return []
    files = sorted(f for f in Path(mod_dir).iterdir() if f.name != STAGED_RECORD)
    if not files:
        return []

    record = Path(mod_dir) / STAGED_RECORD
    if record.exists():
        saved = open_yaml(record)
        current = _grid_settings(tuple(saved))
        changed = sorted(k for k in saved if saved[k] != current[k])
        if changed:
            raise ValueError(
                f"{mod_dir} holds files staged for other grid settings "
                + f"({', '.join(changed)} changed). Remove the directory to "
                + "regenerate them, or restore the settings."
            )
    else:
        log.warning(
            f"Reusing {mod_dir} without a record of its grid settings; "
            + "the files must match run_config.yaml"
        )
    return files


def stage_for_reuse(src_dir: Path, mod_dir: Path, keys: tuple = GRID_KEYS) -> None:
    """Copy src_dir to mod_dir and record the settings (`keys`) it was made for."""
    mod_dir = Path(mod_dir)
    mod_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(src_dir).iterdir():
        dest = mod_dir / f.name
        if f.is_dir():
            shutil.copytree(f, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(f, dest)
    dump_yaml(_grid_settings(keys), mod_dir / STAGED_RECORD)


# Flags of every srun launch of an MPI program; the rank count follows as `-n N`.
SRUN = [
    "srun",
    "--mpi=pmix",
    "--distribution=block:block",
    "--cpu-bind=cores",
    "--ntasks-per-core=1",
    "--kill-on-bad-exit=1",
]


def get_launcher(n_procs: int, serial: bool = False) -> list:
    """Command prefix for the preprocessing executables.

    Native preprocessing (preprocess_use_container false) runs MPI steps as srun steps of
    the Slurm allocation, so Slurm places them on any node of the job. In the container
    the MPI steps use mpirun on this node: the rank count follows the CPUs visible to
    the process, which can exceed the Slurm task slots (n_cpus_per_task > 1), so
    oversubscription is allowed; Open MPI then does not bind ranks to cores.

    ``serial`` marks a program that is not MPI (orog, lakefrac, make_solo_mosaic).
    It runs bare, with no prefix, in both modes, so it can use the memory of the
    whole job (a one-task srun step is limited to its own share and can be OOM killed).
    """
    if serial:
        return []
    if state.preprocess_use_container:
        return [
            "mpirun",
            "--oversubscribe",
            "-np",
            str(n_procs),
            "--host",
            f"localhost:{n_procs}",
        ]
    return [*SRUN, "-n", str(n_procs)]


def to_builtin(obj: object) -> object:
    if isinstance(obj, Mapping):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj


def open_nml(path: Path) -> dict:
    return to_builtin(f90nml.read(path))


def write_nml(nml: dict, path: Path) -> None:
    with open(path, "w") as f:
        f90nml.write(nml, f)


def read_namelist(path: Path) -> dict:
    if re.search(r"\.(?:nml|\d+)$", str(path)):
        data = open_nml(path)
    elif str(path).endswith((".yaml", ".yml")):
        data = dict(open_yaml(path))
    else:
        raise ValueError(
            "Unsupported Namelist file format. Use .nml, fotran file fds i.e .41 or .yaml/.yml"
        )
    return data


def sort_paths(f: str | Path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", Path(f).name)]


def to_list(x: object) -> list:
    return [x] if not isinstance(x, list) else x


def fix_file(rel: str, required: bool = True) -> Path | None:
    """Path of a fix_src file, fetched from the NOAA fix bucket when missing.

    `rel` is relative to fix_src, for example "am/global_glacier.2x2.grb".
    Raises FileNotFoundError when a required file is unavailable locally and
    remotely; returns None for an optional one.
    """
    return ensure_fix_file(paths["fix_src"], rel, required=required)


def require_fix_files(files: list[Path]) -> None:
    """Ensure `files` exist, fetching any missing from fix_src's remote source.

    Raises FileNotFoundError naming every file that cannot be found. Paths
    outside fix_src (for example executables) cannot be fetched and are
    reported directly.
    """
    fix_src = Path(paths["fix_src"])
    unresolved = []
    for f in files:
        if Path(f).exists():
            continue
        try:
            rel = Path(f).relative_to(fix_src).as_posix()
        except ValueError:
            unresolved.append(str(f))
            continue
        if fix_file(rel, required=False) is None:
            unresolved.append(rel)

    if unresolved:
        raise FileNotFoundError(missing_message(fix_src, unresolved))


@contextmanager
def tmp_cwd(path: Path | str):
    cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def handle_errors(exc_type, value, tb):
    log = logging.getLogger("ERROR.HANDLER")

    def _norm_path(p: str) -> str:
        try:
            return str(Path(p).resolve())
        except (OSError, RuntimeError):
            return p

    user_frames = [
        f
        for f in traceback.extract_tb(tb)
        if Path(_norm_path(f.filename)).is_relative_to(paths["pyufs"] / "src")
        and f.filename.endswith(".py")
    ]
    sys_frames = [f for f in traceback.extract_tb(tb)]

    if not user_frames:
        log.error(f"{exc_type.__qualname__}: {value}")
        return

    frame = user_frames[-1]

    file_name = Path(frame.filename).name
    lineno = f"{frame.lineno}"

    log.warning(f"An error has been detected in file: {file_name},  line no: {lineno}")
    log.error(f"{exc_type.__qualname__}: {value}")

    # now print frames
    print("\nTraceback")
    for f in sys_frames:
        print(f)


sys.excepthook = handle_errors
