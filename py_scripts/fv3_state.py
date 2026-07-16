import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fv3_paths import case_paths, paths
from fv3_utils import parse_datetime

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    format=log_format,
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    force=True,
)


class FV3State(dict):
    __slots__ = ()

    # Case identity, descriptions, and directories
    case_name: str
    case_description: str
    description: str
    checksum: str

    case_dir: Path
    work_dir: Path
    run_dir: Path
    run_config: Path
    configs: Path
    input: Path
    output: Path
    logs: Path
    tmp: Path

    # Archiving and history
    archive_data: bool
    archive_dir: Path
    hist: Path
    restarts: Path

    # Execution control and simulation timeline
    init_datetime: pd.Timestamp
    run_nhours: int
    total_run_hours: int

    forecast_hour: int
    forecast_length: str

    continue_run: bool
    warm_start: bool
    preprocess_only: bool
    restart_no: int
    total_restarts: int
    resubmit: int
    resubmit_idx: int

    # Model time stepping and resolution
    dt_atmos: int
    dt_ocean: int
    c_res: int
    res_km: list[float]
    delx: float
    dely: float
    levels: int

    # Computational resources and decomposition
    n_nodes: int
    n_cpus: int
    n_cpus_per_node: int
    total_pes: int
    multi_node: bool

    layout: list[list[int]]
    io_layout: list[list[int]]
    blocksize: list[int]
    grid_pes: list[int]
    n_split: list[int]
    k_split: list[int]

    # Grid geometry and domain configuration
    gtype: str
    grid: Path
    idim: int
    jdim: int
    npx: list[int]
    npy: list[int]
    ntiles: list[int]
    ngrid_cells: list[int]
    halo: int

    stretch_factor: float
    target_lat: float
    target_lon: float

    # Nested-domain configuration
    n_nests: int
    nest_type: str
    refine_ratio: list[int]
    merge_freq: int

    # Note: the original declaration listed parent_tile twice with
    # incompatible types. Retain the appropriate form for your workflow.
    parent_tile: list[int]  # or int

    istart_nest: list[int]
    iend_nest: list[int]
    jstart_nest: list[int]
    jend_nest: list[int]

    nest_ioffsets: list[int]
    nest_joffsets: list[int]

    lat_min: list[int]
    lat_max: list[int]
    lon_min: list[int]
    lon_max: list[int]

    # Initial-condition generation and external data sources
    generate_ic_data: bool
    external_ic_dir: Path | None
    ic_data: Path

    global_ic_source: dict[str, str]
    nest02_ic_source: dict[str, str]
    nest03_ic_source: dict[str, str]
    nest04_ic_source: dict[str, str]

    # Surface and terrain preprocessing
    add_lake: bool
    lake_cutoff: float
    make_gsl_orog: bool
    do_deep: bool
    sm_perturbations: dict

    # Fixed files, preprocessing, and executables
    fix: Path
    fix_src: Path

    ufs_exe: Path
    ufs_utils: Path
    shield_exe: str

    shield_image: Path
    fregrid_image: Path
    preprocess_image: Path
    python_modules: list[Path]

    # Runtime environment and diagnostics
    container_bindpath: list[str]
    modules: list[str]
    fv3_debug: bool

    # Ensemble configuration
    ensemble_run: bool
    ensemble_id: int
    n_ensembles: int

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name) -> Any:
        if name in self:
            return self[name]
        return None


state = FV3State({})
prev_state = FV3State({})

env_vars = {
    "case_name": os.getenv("CASE_NAME"),
    "n_cpus": int(os.environ.get("CASE_NTASKS")),
    "n_nodes": int(os.environ.get("CASE_NNODES", 1)),
    "ensemble_id": int(os.environ.get("CASE_ENSEMBLE_ID", 0)),
    "n_ensembles": int(os.environ.get("CASE_ENSEMBLES", 0)),
    "n_cpus_per_node": int(os.environ.get("CASE_NTASKS_PER_NODE")),
    "multi_node": bool(int(os.getenv("CASE_MULTI_NODE_FLAG", 0))),
    "resubmit": int(os.getenv("CASE_RESUBMIT_MAX", 0)),
    "resubmit_idx": int(os.getenv("CASE_RESUBMIT_INDEX", 0)),
    "ufs_utils": Path(__file__).resolve().parent.parent,
    "configs": Path(__file__).resolve().parent.parent / "configs",
}

state.update(env_vars)

if env_vars["resubmit_idx"] > 0:
    log = logging.getLogger("RESTART")
else:
    log = logging.getLogger("PREPROCESS")


def compute_checksum(data: dict | FV3State, hash_keys: list = None) -> str:
    if not isinstance(hash_keys, list) and hash_keys is not None:
        raise ValueError("hash_keys must be a list of keys to include in the hash")

    _hash_keys = [
        "c_res",
        "gtype",
        "levels",
        "target_lon",
        "target_lat",
        "stretch_factor",
        "refine_ratio",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "init_datetime",
    ]

    if hash_keys is not None:
        _hash_keys += list(hash_keys)

    def _normalize_for_hash(value):
        if isinstance(value, dict):
            return {
                str(k): _normalize_for_hash(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, list):
            return [_normalize_for_hash(v) for v in value]
        if isinstance(value, pd.Timestamp):
            return str(value)

        if isinstance(value, tuple):
            return [_normalize_for_hash(v) for v in value]
        return value

    payload = {key: _normalize_for_hash(data.get(key, None)) for key in _hash_keys}

    hash_data_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(hash_data_str.encode("utf-8")).hexdigest()


def save_fv3_state(cfg: dict = None, path: Path = None):
    """
    Save the current state to a YAML file
    """

    if cfg is not None:
        _cfg = cfg
    else:
        _cfg = state

    if not _cfg:
        return

    if path is None:
        path = Path(paths["work_dir"]) / "state.yaml"

    data = {}
    if path.exists():
        path.unlink()

    for k, v in _cfg.items():
        if k in case_paths:
            continue
        if isinstance(v, Path):
            v = str(v)

        data[k] = v

    data["init_datetime"] = str(data["init_datetime"])

    with open(path, "w") as f:
        yaml.safe_dump(dict(data), f, default_flow_style=None, sort_keys=False)


def load_fv3_state(merge: bool = False):
    """
    Load the previous state from a YAML file, if it exists
    """
    path = Path(paths["work_dir"]) / "state.yaml"
    if not path.exists():
        log.info(f"No previous state file found at {path}. Starting with empty state.")
        return

    prev_state.clear()
    with open(path, "r") as f:
        data = yaml.safe_load(f)

        data = parse_datetime(data)
        prev_state.update(data)
        prev_state.update(paths)
        state.update(paths)

    if merge:
        new_state = FV3State({**prev_state, **state})
        state.update(new_state)
        save_fv3_state()
