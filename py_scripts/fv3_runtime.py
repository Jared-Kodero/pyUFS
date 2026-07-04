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
    with xr.open_dataset(gridfile) as ds:
        nx = ds.nx.shape[0]

    return int(nx / 2)


def get_launcher(n_procs: int = None) -> list:
    return ["mpirun", "-np", str(n_procs), "--host", "localhost"]


def exit_code(code: int) -> None:
    (paths["work_dir"] / "exit_code").write_text(str(code))


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
        override_nml = open_nml(path)
    elif str(path).endswith((".yaml", ".yml")):
        override_nml = open_yaml(path)
    else:
        raise ValueError(
            "Unsupported Namelist file format. Use .nml, fotran file fds i.e .41 or .yaml/.yml"
        )
    return override_nml


def sort_paths(f: str | Path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", Path(f).name)]


def to_list(x: object) -> list:
    return [x] if not isinstance(x, list) else x


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
    code_line = frame.line.strip() if frame.line else ""

    log.warning(f"An error has been detected in file: {file_name},  line no: {lineno}")
    log.error(f"{exc_type.__qualname__}: {value}")

    # now print frames
    print("\nTraceback")
    for f in sys_frames:
        print(f)


sys.excepthook = handle_errors
