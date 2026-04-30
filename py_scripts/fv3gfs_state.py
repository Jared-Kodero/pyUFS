import logging
import os
from pathlib import Path

import yaml
from fv3gfs_paths import paths
from fv3gfs_utils import parse_datetime


class FV3State(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


state = FV3State({})
prev_state = FV3State({})


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


def logger(debug=False):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        format=log_format,
        datefmt="%Y-%m-%d %H:%M",
        level=level,
        handlers=[logging.StreamHandler()],
        force=True,
    )


env_vars = {
    "case_name": os.getenv("CASE_NAME"),
    "n_cpus": int(os.environ.get("SBATCH_NTASKS")),
    "n_nodes": int(os.environ.get("SBATCH_NNODES", 1)),
    "node_list": os.environ.get("SLURM_NODELIST"),
    "ensemble_id": int(os.environ.get("ENSEMBLE_ID", 0)),
    "n_ensembles": int(os.environ.get("N_ENSEMBLES", 1)),
    "n_cpus_per_node": int(os.environ.get("SBATCH_NTASKS_PER_NODE")),
    "multi_node": bool(int(os.getenv("SBATCH_MULTI_NODE", 0))),
    "ufs_utils": Path(__file__).resolve().parent.parent,
    "configs": Path(__file__).resolve().parent.parent / "configs",
}

state.update(env_vars)
logger()
log = logging.getLogger("PREPROCESSING")
