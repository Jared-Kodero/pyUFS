# runtime.py

import logging
import re
from pathlib import Path

import xarray as xr
from fv3gfs_paths import paths

log = logging.getLogger("PREPROCESSING")


def get_newres(gridfile):
    with xr.open_dataset(gridfile) as ds:
        nx = ds.nx.shape[0]

    return int(nx / 2)


def get_launcher(n_procs: int = None) -> list:
    return ["mpirun", "-np", str(n_procs), "--host", "localhost"]


def exit_code(code: int) -> None:
    (paths["home"] / "exit_code").write_text(str(code))


def nml_to_dict(nml):
    nml = dict(nml)
    return {k: dict(v) for k, v in nml.items()}


def sort_paths(f):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", Path(f).name)]


def to_list(x):
    return [x] if not isinstance(x, list) else x
