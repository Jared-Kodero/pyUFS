from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeAlias

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


NamelistOverride: TypeAlias = str | Path | dict[str, Any] | None
StateValue: TypeAlias = Any


@dataclass
class FV3State:
    # run_config.yaml

    case_name: str | None = None
    description: str | None = None
    fv3_debug: bool = False
    archive_data: bool = True

    shield_exe: str | Path | None = None
    init_datetime: pd.Timestamp | str | None = None
    run_nhours: int | None = None
    forecast_hour: int = 0
    continue_run: bool = False
    resubmit: int = 0

    ensemble_run: bool = False
    n_ensembles: int = 0
    paired_ensembles: bool = False
    skip_ensembles: list[int] | None = None

    ic_gen: bool = True
    ic_only: bool = False
    chgres_config: NamelistOverride = None

    res: int = 96
    gtype: str = "uniform"
    target_lon: float = -96.0
    target_lat: float = 35.0
    stretch_factor: float = 1.0

    refine_ratio: list[int] = field(default_factory=lambda: [3])
    parent_tile: int = 6
    halo: int = 3
    lon_min: list[float] | None = None
    lon_max: list[float] | None = None
    lat_min: list[float] | None = None
    lat_max: list[float] | None = None

    idim: int = 200
    jdim: int = 200
    delx: float = 0.0585
    dely: float = 0.0585

    levels: int = 64
    do_deep: bool = False

    dt_atmos: float | None = None
    dt_ocean: float | None = None
    k_split: list[int] | None = None
    n_split: list[int] | None = None

    lake_cutoff: float = 0.2
    add_lake: bool = False
    make_gsl_orog: bool = False

    global_input_nml: NamelistOverride = None
    nestXX_input_nml: NamelistOverride = None

    sm_perturbations: dict[str, Any] | None = None

    # Existing fv3_state.py environment fields

    n_cpus: int | None = None
    n_nodes: int = 1
    node_list: str | None = None
    ensemble_id: int = 0
    n_cpus_per_node: int | None = None
    multi_node: bool = False

    ufs_utils: Path | None = None
    configs: Path | None = None

    # Stable workflow and restart fields

    warm_start: bool = False
    restart_no: int = 0
    update_nml_only: bool = False

    # Stable keys supplied by fv3_paths.py

    fix: Path | None = None
    home: Path | None = None
    fix_am: Path | None = None
    ufs_exe: Path | None = None
    rundir: Path | None = None
    case_dir: Path | None = None
    scratch_dir: Path | None = None
    archive_dir: Path | None = None

    tmp: Path | None = None
    hist: Path | None = None
    grid: Path | None = None
    logs: Path | None = None
    fixed: Path | None = None
    input: Path | None = None
    output: Path | None = None
    restarts: Path | None = None
    IC: Path | None = None

    # Generated grid and nest keys remain here.

    _dynamic: dict[str, StateValue] = field(
        default_factory=dict,
        repr=False,
    )

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(item.name for item in fields(cls) if item.name != "_dynamic")

    def __getattr__(self, key: str) -> StateValue:
        try:
            return self._dynamic[key]
        except KeyError as exc:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {key!r}"
            ) from exc

    def __setattr__(self, key: str, value: StateValue) -> None:
        if key.startswith("_") or key in self.field_names():
            object.__setattr__(self, key, value)
        else:
            self._dynamic[key] = value

    def __getitem__(self, key: str) -> StateValue:
        if key in self.field_names():
            return object.__getattribute__(self, key)
        return self._dynamic[key]

    def __setitem__(self, key: str, value: StateValue) -> None:
        setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        yield from self.field_names()
        yield from self._dynamic

    def __len__(self) -> int:
        return len(self.field_names()) + len(self._dynamic)

    def keys(self) -> Iterator[str]:
        return iter(self)

    def items(self) -> Iterator[tuple[str, StateValue]]:
        for key in self:
            yield key, self[key]

    def get(self, key: str, default: StateValue = None) -> StateValue:
        try:
            return self[key]
        except KeyError:
            return default

    def update(
        self,
        values: Mapping[str, StateValue] | None = None,
        **kwargs: StateValue,
    ) -> None:
        if values is not None:
            for key, value in values.items():
                self[key] = value

        for key, value in kwargs.items():
            self[key] = value

    def clear(self) -> None:
        defaults = type(self)()

        for key in self.field_names():
            object.__setattr__(self, key, getattr(defaults, key))

        self._dynamic.clear()


def env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


state = FV3State(
    case_name=os.getenv("CASE_NAME"),
    n_cpus=env_int("SBATCH_NTASKS"),
    n_nodes=env_int("SBATCH_NNODES", 1) or 1,
    node_list=os.getenv("SLURM_NODELIST"),
    ensemble_id=env_int("CASE_ENSEMBLE_ID", 0) or 0,
    n_ensembles=env_int("CASE_ENSEMBLES", 0) or 0,
    n_cpus_per_node=env_int("SBATCH_NTASKS_PER_NODE"),
    multi_node=bool(env_int("SBATCH_MULTI_NODE_FLAG", 0)),
    ufs_utils=Path(__file__).resolve().parent.parent,
    configs=Path(__file__).resolve().parent.parent / "configs",
)

prev_state = FV3State()


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


def merge_saved_state():
    """Merge current and previous states"""

    load_state()
    new_state = FV3State({**prev_state, **state})
    state.update(new_state)
    save_state()


def save_state(cfg: dict = None, path: Path = None):
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
        path = Path(paths["home"]) / "state.yaml"

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
