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
import yaml
from fv3_paths import paths
from fv3_update_fix import ensure_fix_file, missing_message

log = logging.getLogger("PREPROCESS")


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


def merged_run_config() -> dict:
    """The case run_config.yaml, with unset or null keys from configs/run_config.yaml.

    The same merge is applied by drivers/case_submit.py at submission.
    """
    default = read_namelist(Path(paths["configs"]) / "run_config.yaml")
    case = read_namelist(Path(paths["run_dir"]) / "run_config.yaml") or {}
    merged = dict(case)
    for key, value in default.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
    return merged


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
    cfg = merged_run_config()
    return {k: to_builtin(cfg.get(k)) for k in keys}


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
        saved = yaml.safe_load(record.read_text()) or {}
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
    with open(mod_dir / STAGED_RECORD, "w") as fh:
        yaml.safe_dump(_grid_settings(keys), fh, sort_keys=False)


def get_launcher(n_procs: int | None = None) -> list:
    """mpirun for the preprocessing executables on this node.

    The rank count follows the CPUs visible to the process, which can exceed
    the Slurm task slots (n_cpus_per_task > 1), so oversubscription is
    allowed; Open MPI then does not bind ranks to cores.
    """
    return [
        "mpirun",
        "--oversubscribe",
        "-np",
        str(n_procs),
        "--host",
        f"localhost:{n_procs}",
    ]


def open_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = dict(yaml.safe_load(f) or {})
    return data


def to_builtin(obj: object) -> object:
    if isinstance(obj, Mapping):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj


def nml_to_dict(nml: dict) -> dict:
    return to_builtin(nml)


def open_nml(path: Path) -> dict:
    return nml_to_dict(f90nml.read(path))


def read_namelist(path: Path) -> dict:
    if re.search(r"\.(?:nml|\d+)$", str(path)):
        data = open_nml(path)
    elif str(path).endswith((".yaml", ".yml")):
        data = open_yaml(path)
    else:
        raise ValueError(
            "Unsupported Namelist file format. Use .nml, fotran file fds i.e .41 or .yaml/.yml"
        )
    return data


def sort_paths(f: str | Path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", Path(f).name)]


def to_list(x: object) -> list:
    return [x] if not isinstance(x, list) else x


def get_stream_handles() -> list[str]:
    """Return unique file names from diag_table.yaml (modern) or diag_table."""
    yaml_path = Path(paths["work_dir"]) / "diag_table.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            table = yaml.safe_load(f)
        return list(dict.fromkeys(d["file_name"] for d in table["diag_files"]))

    path = Path(paths["work_dir"]) / "diag_table"
    stream_files: list[str] = []

    file_section_keys = [
        "file_name",
        "freq",
        "freq_units",
        "time_units",
        "unlimdim",
        "new_file_freq",
        "new_file_freq_units",
        "start_time",
        "file_duration",
        "file_duration_units",
        "filename_time_bounds",
    ]

    file_section_fvalues = {
        "file_name": str,
        "freq": int,
        "freq_units": str,
        "time_units": str,
        "unlimdim": str,
        "new_file_freq": int,
        "new_file_freq_units": str,
        "start_time": str,
        "file_duration": int,
        "file_duration_units": str,
        "filename_time_bounds": str,
    }

    global_lines_read = 0

    with open(path) as f:
        for raw_line in f:
            stripped = raw_line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            # Match parse_diag_table(): skip title and base_date.
            if global_lines_read < 2:
                global_lines_read += 1
                continue

            line = stripped.strip(",")
            parts = line.split("#", 1)[0].split(",")

            try:
                # Match the parser's file-section conversion logic.
                for i, part in enumerate(parts):
                    if i == 3:
                        continue  # file_format

                    key_index = i if i < 3 else i - 1
                    key = file_section_keys[key_index]
                    value = file_section_fvalues[key](
                        part.strip().strip('"').strip("'")
                    )

                    # These conditions do not affect identification of a file line,
                    # but are retained to mirror the parser.
                    if i == 9 and value <= 0:
                        continue
                    if i == 10 and value == "":
                        continue

                stream_files.append(parts[0].strip().strip('"').strip("'"))

            except (IndexError, KeyError, ValueError):
                # The source parser treats this as a field-section line.
                continue

    return list(dict.fromkeys(stream_files))


def fix_file(rel: str, required: bool = True) -> Path | None:
    """Path of a fix_src file, fetched from the NOAA fix bucket when missing.

    `rel` is relative to fix_src, for example "am/global_glacier.2x2.grb".
    Raises FileNotFoundError when a required file is unavailable locally and
    remotely; returns None for an optional one.
    """
    return ensure_fix_file(paths["fix_src"], rel, required=required)


def require_fix_files(files: list[Path], sub_dir: str = "") -> None:
    """Ensure `files` exist, fetching any missing from fix_src's remote source.

    Raises FileNotFoundError naming every file that cannot be found. Paths
    outside fix_src (for example executables) cannot be fetched and are
    reported directly. `sub_dir` is accepted for older call sites.
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
    cwd = paths["work_dir"]
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
        if "py_scripts" in _norm_path(f.filename) and f.filename.endswith(".py")
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
