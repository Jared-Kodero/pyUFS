# state.py

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
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

log = logging.getLogger("UFS_UTILS")

# Fields stored as pathlib.Path. Strings assigned to them, for example from
# state.yaml, are converted on assignment.
PATH_FIELDS = frozenset(
    {
        "work_dir",
        "fix_src",
        "ufs_exe",
        "run_dir",
        "case_dir",
        "archive_dir",
        "ufs_utils",
        "configs",
        "tmp",
        "hist",
        "grid",
        "logs",
        "fix",
        "input",
        "output",
        "restarts",
        "ic_data",
        "bc_data",
        "run_config",
        "external_ic_dir",
    }
)


@dataclass
class FV3State:
    """Run state shared by the preprocessing, model and regridding stages.

    Every key is a declared field. Assigning an undeclared key raises, so a
    misspelled key fails at the point of use instead of reading as None.

    Fields in the configuration group default to None; their default values
    are defined once, in configs/run_config.yaml, and merged by the init
    driver. The remaining fields are derived during the run and carry typed
    defaults. Mapping-style access (state["key"], get, update, items) is kept
    for the existing call sites.
    """

    # --- Configuration (configs/run_config.yaml) --------------------------
    # 0. Job environment
    case_root: str | None = None
    jobtmp: str | None = None
    fix_src: Path | None = None
    ufs_utils: Path | None = None
    shield_image: str | None = None
    fregrid_image: str | None = None
    preprocess_image: str | None = None
    containers_root: str | None = None
    shield_root: str | None = None
    archive_root: str | None = None
    shield_exe: str | None = None
    container_bindpath: list[str] | None = None
    modules: list[str] | None = None
    # 1. Job submission
    constraint_node: str | None = None
    exclusive_node: bool | None = None
    walltime: int | None = None
    n_nodes: int | None = None
    n_cpus: int | None = None
    partition: str | None = None
    n_cpus_per_task: int | None = None
    mem: int | None = None
    sbatch_options: str | None = None
    logfile: str | None = None
    # 2. Case metadata and output
    case_name: str | None = None
    description: str | None = None
    fv3_debug: bool | None = None
    archive_data: bool | None = None
    merge_freq: int | None = None
    use_modern_diag: bool | None = None
    pack_output: bool | None = None
    # 3. Execution control
    init_datetime: pd.Timestamp | None = None  # model start
    run_length: int | None = None
    run_length_units: str | None = None
    forecast_hour: int | None = None
    # Cycle of the source data (run_config init_datetime); the model starts
    # forecast_hour hours later.
    ic_cycle: pd.Timestamp | None = None
    resubmit: int | None = None
    continue_run: bool | None = None
    preprocess_dask_scheduler: str | None = None
    # 4. Ensembles
    ensemble_run: bool | None = None
    n_ensembles: int | None = None
    skip_ensembles: list[int] | int | None = None
    # 5. Initial conditions and preprocessing
    external_ic_dir: Path | None = None
    generate_ic_data: bool | None = None
    preprocess_only: bool | None = None
    preprocess_grid_only: bool | None = None
    preprocess_orog_only: bool | None = None
    # 5a. Surface analysis and coupled grids (UFS_UTILS)
    run_global_cycle: bool | None = None
    global_cycle_sst_file: str | None = None
    global_cycle_ice_file: str | None = None
    global_cycle_snow_file: str | None = None
    global_cycle_vars: dict | None = None
    run_emcsfc_snow: bool | None = None
    ims_snow_file: str | None = None
    afwa_snow_nh_file: str | None = None
    afwa_snow_sh_file: str | None = None
    afwa_snow_global_file: str | None = None
    run_emcsfc_ice_blend: bool | None = None
    ims_ice_file: str | None = None
    five_min_ice_file: str | None = None
    run_cpld_gridgen: bool | None = None
    cpld_gridgen_res: str | int | None = None
    cpld_gridgen_postwgts: bool | None = None
    # 6. Horizontal grid
    c_res: int | None = None
    gtype: str | None = None
    target_lon: float | None = None
    target_lat: float | None = None
    stretch_factor: float | None = None
    refine_ratio: list[int] | int | None = None
    parent_tile: list[int] | int | None = None
    halo: int | None = None
    lon_min: list[float] | float | None = None
    lon_max: list[float] | float | None = None
    lat_min: list[float] | float | None = None
    lat_max: list[float] | float | None = None
    idim: int | None = None
    jdim: int | None = None
    delx: float | None = None
    dely: float | None = None
    # 7. Vertical grid and physics
    levels: int | None = None
    do_deep: bool | None = None
    # 8. Time stepping
    dt_atmos: int | None = None
    dt_ocean: int | None = None
    k_split: list[int] | None = None
    n_split: list[int] | None = None
    # 9. Surface and orography
    lake_cutoff: float | None = None
    add_lake: bool | None = None
    make_gsl_orog: bool | None = None
    # 10-11. Perturbations
    sm_perturbations: dict | None = None
    tgrad_perturbations: dict | None = None

    # --- Runtime environment (fv3_utils.runtime_env_vars) ----------------
    n_cpus_per_node: int | None = None
    multi_node: bool = False
    ensemble_id: int = 0
    resubmit_idx: int = 0

    # --- Derived case identity and timeline ------------------------------
    case_description: str = ""
    checksum: str | None = None
    config_checksum: str | None = None
    run_config: Path | None = None
    warm_start: bool = False
    restart_no: int = 0
    total_restarts: int = 1
    total_run_hours: int = 0
    forecast_length: str = ""
    model_start_date: list[int] = field(default_factory=list)

    # --- Derived grid geometry and nesting --------------------------------
    res_km: list[float] = field(default_factory=list)
    n_nests: int = 0
    nest_type: str | None = None
    istart_nest: list[int] = field(default_factory=list)
    iend_nest: list[int] = field(default_factory=list)
    jstart_nest: list[int] = field(default_factory=list)
    jend_nest: list[int] = field(default_factory=list)
    nest_ioffsets: list[int] = field(default_factory=list)
    nest_joffsets: list[int] = field(default_factory=list)

    # --- Derived decomposition -------------------------------------------
    ngrid_cells: list[int] = field(default_factory=list)
    ntiles: list[int] = field(default_factory=list)
    npx: list[int] = field(default_factory=list)
    npy: list[int] = field(default_factory=list)
    layout: list[list[int]] = field(default_factory=list)
    io_layout: list[list[int]] = field(default_factory=list)
    blocksize: list[int] = field(default_factory=list)
    grid_pes: list[int] = field(default_factory=list)
    total_pes: int = 0

    # Model supplying each field group {"atm", "sfc", "nst"} per domain
    # ("global", "regional", "nest02", ...); None when a group was not
    # converted. Set by chgres_cube.run_chgres when ICs are generated.
    ic_source: dict[str, dict[str, str | None]] = field(default_factory=dict)

    # --- Paths (fv3_paths) -------------------------------------------------
    work_dir: Path | None = None
    ufs_exe: Path | None = None
    run_dir: Path | None = None
    case_dir: Path | None = None
    archive_dir: Path | None = None
    configs: Path | None = None
    tmp: Path | None = None
    hist: Path | None = None
    grid: Path | None = None
    logs: Path | None = None
    fix: Path | None = None
    input: Path | None = None
    output: Path | None = None
    restarts: Path | None = None
    ic_data: Path | None = None
    bc_data: Path | None = None

    # --- Field access ---------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in _FIELD_NAMES:
            raise AttributeError(f"FV3State has no field {name!r}")
        if name in PATH_FIELDS and value is not None and not isinstance(value, Path):
            value = Path(value)
        object.__setattr__(self, name, value)

    def __getitem__(self, name: str) -> Any:
        if name not in _FIELD_NAMES:
            raise KeyError(name)
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if name not in _FIELD_NAMES:
            raise KeyError(f"FV3State has no field {name!r}")
        setattr(self, name, value)

    def __contains__(self, name: object) -> bool:
        return name in _FIELD_NAMES

    def get(self, name: str, default: Any = None) -> Any:
        if name not in _FIELD_NAMES:
            raise KeyError(f"FV3State has no field {name!r}")
        value = getattr(self, name)
        return default if value is None else value

    def keys(self) -> list[str]:
        return list(_FIELD_NAMES)

    def items(self) -> list[tuple[str, Any]]:
        return [(name, getattr(self, name)) for name in _FIELD_NAMES]

    def update(self, other: dict | None = None, **kwargs: Any) -> None:
        for source in (other or {}, kwargs):
            for name, value in source.items():
                self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return dict(self.items())

    def reset(self) -> None:
        """Restore every field to its default."""
        for f in dataclasses.fields(self):
            if f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()
            else:
                value = f.default
            object.__setattr__(self, f.name, value)

    clear = reset


_FIELD_NAMES = tuple(f.name for f in dataclasses.fields(FV3State))

state = FV3State()


def compute_checksum(data: dict | FV3State, hash_keys: list | None = None) -> str:
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

    # Hashed only when set, so checksums of existing cases remain valid.
    if data.get("forecast_hour"):
        _hash_keys.append("forecast_hour")
    if data.get("tgrad_perturbations") is not None:
        _hash_keys.append("tgrad_perturbations")

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


def save_fv3_state(cfg: FV3State | None = None, path: Path | None = None) -> None:
    """Save the state to a YAML file, excluding the per-run case paths."""

    _cfg = state if cfg is None else cfg

    if path is None:
        path = Path(paths["work_dir"]) / "state.yaml"

    data = {}
    for k, v in _cfg.items():
        if k in case_paths:
            continue
        if isinstance(v, Path):
            v = str(v)
        data[k] = v

    for key in ("init_datetime", "ic_cycle"):
        if data.get(key) is not None:
            data[key] = pd.Timestamp(data[key]).strftime("%Y%m%d%HZ")

    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
    tmp.replace(path)


def load_fv3_state() -> FV3State:
    """Load the persisted state, then apply the current environment paths."""
    path = Path(paths["work_dir"]) / "state.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Restart state file not found: {path}")

    with path.open("r") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise TypeError("Invalid state file format")

    # The model start is a cycle plus forecast_hour, so any hour.
    data["init_datetime"] = parse_datetime(data["init_datetime"], cycle=False)
    if data.get("ic_cycle") is not None:
        data["ic_cycle"] = parse_datetime(data["ic_cycle"])

    state.reset()
    state.update(data)
    state.update(paths)

    return state
