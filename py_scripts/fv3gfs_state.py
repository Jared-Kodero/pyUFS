import hashlib
import logging
import os
from pathlib import Path

import yaml
from fv3gfs_paths import paths
from fv3gfs_utils import parse_datetime

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    format=log_format,
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    force=True,
)


class FV3State(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


state = FV3State({})
prev_state = FV3State({})


def compute_checksum(data: dict) -> str:
    _hash_keys = (
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
    )

    _hash_data_str = ",".join(str(data.get(k)) for k in _hash_keys)
    checksum = hashlib.sha256(_hash_data_str.encode("utf-8")).hexdigest()

    return checksum


def save_state():
    """
    Save the current state to a YAML file
    """

    if not state:
        return

    path = Path(paths["home"]) / "state.yaml"
    data = {}

    if path.exists():
        path.unlink()

    for k, v in state.items():
        if isinstance(v, Path):
            continue
        data[k] = v

    data["init_datetime"] = str(data["init_datetime"])

    with open(path, "w") as f:
        yaml.safe_dump(dict(data), f, default_flow_style=None)


def load_state():
    """
    Load the previous state from a YAML file, if it exists
    """
    path = Path(paths["home"]) / "state.yaml"
    if not path.exists():
        return FV3State({})

    prev_state.clear()
    with open(path, "r") as f:
        data = yaml.safe_load(f)

        data = parse_datetime(data)
        prev_state.update(data)
        prev_state.update(paths)
        state.update(paths)


env_vars = {
    "case_name": os.getenv("CASE_NAME"),
    "n_cpus": int(os.environ.get("SBATCH_NTASKS")),
    "n_nodes": int(os.environ.get("SBATCH_NNODES", 1)),
    "node_list": os.environ.get("SLURM_NODELIST"),
    "ensemble_id": int(os.environ.get("CASE_ENSEMBLE_ID", 0)),
    "n_ensembles": int(os.environ.get("CASE_ENSEMBLES", 1)),
    "n_cpus_per_node": int(os.environ.get("SBATCH_NTASKS_PER_NODE")),
    "multi_node": bool(int(os.getenv("SBATCH_MULTI_NODE_FLAG", 0))),
    "ufs_utils": Path(__file__).resolve().parent.parent,
    "configs": Path(__file__).resolve().parent.parent / "configs",
}

state.update(env_vars)
log = logging.getLogger("PREPROCESSING")
