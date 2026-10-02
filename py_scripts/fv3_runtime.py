# runtime.py
import logging
import os
import re
import sys
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import f90nml
import xarray as xr
import yaml

from fv3_paths import paths

log = logging.getLogger("PREPROCESS")


def get_newres(gridfile: Path) -> int:
    """Return the global-equivalent cubed-sphere resolution of a regional grid.

    global_equiv_resol writes this value to the grid file as the global
    attribute RES_equiv (UFS_UTILS, global_equiv_resol.f90). The supergrid
    dimension nx is twice the zonal cell count of the regional domain and bears
    no relation to the equivalent resolution, so it cannot be used in its place.
    """
    with xr.open_dataset(gridfile) as ds:
        nx = ds.nx.shape[0]
        res_equiv = ds.attrs.get("RES_equiv", None)

    if res_equiv is None:
        res_equiv = int(nx / 2)

    return int(res_equiv)


def get_launcher(n_procs: int | None = None) -> list:
    return ["mpirun", "-np", str(n_procs), "--host", f"localhost:{n_procs}"]


def open_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = dict(yaml.safe_load(f))
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
    """Return unique file-section names from a legacy diag_table."""
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

            except Exception:
                # The source parser treats this as a field-section line.
                continue

    return list(dict.fromkeys(stream_files))


def report_missing_fixed_files(missing_files: list[Path], sub_dir: str = "am") -> None:
    url = f"https://noaa-nws-global-pds.s3.amazonaws.com/index.html#fix/{sub_dir}"
    print("Missing required file(s):")
    for f in missing_files:
        print(f"  - {f}")
    print(
        f"Please download them from\n\t{url}\nand place them in\n\t{paths['fix_src'] / sub_dir}."
    )
    raise FileNotFoundError("Missing required fixed files. See above for details.")


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
        except Exception:
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
