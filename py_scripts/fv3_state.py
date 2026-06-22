import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
import yaml
from fv3_paths import paths
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
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getattr__ = dict.get


state = FV3State({})
prev_state = FV3State({})

env_vars = {
    "case_name": os.getenv("CASE_NAME"),
    "n_cpus": int(os.environ.get("SBATCH_NTASKS")),
    "n_nodes": int(os.environ.get("SBATCH_NNODES", 1)),
    "node_list": os.environ.get("SLURM_NODELIST"),
    "ensemble_id": int(os.environ.get("CASE_ENSEMBLE_ID", 0)),
    "n_ensembles": int(os.environ.get("CASE_ENSEMBLES", 0)),
    "n_cpus_per_node": int(os.environ.get("SBATCH_NTASKS_PER_NODE")),
    "multi_node": bool(int(os.getenv("SBATCH_MULTI_NODE_FLAG", 0))),
    "resubmit": int(os.getenv("CASE_RESUBMIT_MAX", 0)),
    "resubmit_idx": int(os.getenv("CASE_RESUBMIT_INDEX", 0)),
    "ufs_utils": Path(__file__).resolve().parent.parent,
    "configs": Path(__file__).resolve().parent.parent / "configs",
}

state.update(env_vars)
log = logging.getLogger("PREPROCESSING")


def compute_checksum(data: dict | FV3State, hash_keys: list = None) -> str:
    if not isinstance(hash_keys, list) and hash_keys is not None:
        raise ValueError("hash_keys must be a list of keys to include in the hash")

    _hash_keys = [
        "res",
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
        "chgres_config",
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
        path = Path(paths["case_home"]) / "state.yaml"

    data = {}
    if path.exists():
        path.unlink()

    for k, v in _cfg.items():
        if isinstance(v, Path):
            continue
        data[k] = v

    data["init_datetime"] = str(data["init_datetime"])

    with open(path, "w") as f:
        yaml.safe_dump(dict(data), f, default_flow_style=None)


def load_fv3_state(merge: bool = False):
    """
    Load the previous state from a YAML file, if it exists
    """
    path = Path(paths["case_home"]) / "state.yaml"
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
