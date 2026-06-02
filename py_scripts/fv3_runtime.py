# runtime.py
import logging
import re
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

import f90nml
import xarray as xr
import yaml
from fv3_paths import paths

log = logging.getLogger("PREPROCESSING")


def get_newres(gridfile: Path) -> int:
    with xr.open_dataset(gridfile) as ds:
        nx = ds.nx.shape[0]

    return int(nx / 2)


def get_launcher(n_procs: int = None) -> list:
    return ["mpirun", "-np", str(n_procs), "--host", "localhost"]


def exit_code(code: int) -> None:
    (paths["home"] / "exit_code").write_text(str(code))


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
    if str(path).endswith(".nml"):
        override_nml = open_nml(path)
    elif str(path).endswith((".yaml", ".yml")):
        override_nml = open_yaml(path)
    else:
        raise ValueError("Unsupported Namelist file format. Use .nml or .yaml/.yml")
    return override_nml


def sort_paths(f: str | Path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", Path(f).name)]


def to_list(x: object) -> list:
    return [x] if not isinstance(x, list) else x


def handle_errors(type, value, tb):
    log = logging.getLogger("ERROR_HANDLER")

    def _norm_path(p: str) -> str:
        try:
            return str(Path(p).resolve())
        except Exception:
            return p

    frames = [
        f
        for f in traceback.extract_tb(tb)
        if "py_scripts" in _norm_path(f.filename) and f.filename.endswith(".py")
    ]

    if not frames:
        log.error(f"{type.__qualname__}: {value}")
        return

    frame = frames[-1]

    file_name = Path(frame.filename).name
    lineno = f"{frame.lineno}"
    code_line = frame.line.strip() if frame.line else ""

    log.warning(f"An error has been detected in: {file_name}: {lineno}: {code_line}")
    log.error(f"{type.__qualname__}: {value}")


sys.excepthook = handle_errors
