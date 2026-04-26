from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import f90nml
import yaml
from fv3gfs_ic_data import get_init_data, validate_hrrr_bounds
from fv3gfs_runtime import get_launcher
from fv3gfs_stage_data import stage_files
from fv3gfs_state import FV3State, log, state
from fv3gfs_utils import cp, env_setup, run_cmd


@dataclass
class ChgresCubeConfig:
    # === Target grid ===
    mosaic_file_target_grid: Path | None = None
    fix_dir_target_grid: Path | None = None
    orog_dir_target_grid: Path | None = None
    orog_files_target_grid: list[str] | None = None
    vcoord_file_target_grid: Path | None = None

    # === Input grid ===
    data_dir_input_grid: Path | None = None
    atm_files_input_grid: list[str] | None = None
    sfc_files_input_grid: list[str] | None = None
    nst_files_input_grid: list[str] | None = None
    atm_core_files_input_grid: list[str] | None = None
    atm_tracer_files_input_grid: list[str] | None = None
    orog_dir_input_grid: Path | None = None
    orog_files_input_grid: list[str] | None = None
    grib2_file_input_grid: Path | None = None
    geogrid_file_input_grid: Path | None = None
    mosaic_file_input_grid: Path | None = None

    # === Physics / mapping ===
    varmap_file: Path | None = None
    thomp_mp_climo_file: Path | None = None
    wam_parm_file: Path | None = None

    # === Cycle ===
    cycle_year: int | None = None
    cycle_mon: int | None = None
    cycle_day: int | None = None
    cycle_hour: int | None = None

    # === Conversion flags ===
    convert_atm: bool = True
    convert_sfc: bool = True
    convert_nst: bool = True

    # === Input type ===
    input_type: Literal[
        "restart",
        "history",
        "gaussian_nemsio",
        "gaussian_netcdf",
        "grib2",
        "gfs_gaussian_nemsio",
        "gfs_sigio",
    ] = "grib2"

    tracers: list[str] = (
        "sphum",
        "liq_wat",
        "rainwat",
        "ice_wat",
        "snowwat",
        "graupel",
        "o3mr",
        "cld_amt",
    )
    tracers_input: list[str] = (
        "spfh",
        "clwmr",
        "rwmr",
        "icmr",
        "snmr",
        "grle",
        "o3mr",
    )

    # === Grid / nesting ===
    regional: int = 0
    halo_bndy: int = 0
    halo_blend: int = 0

    external_model: Literal["GFS", "NAM", "RAP", "HRRR", "RRFS", "FV3", "FV3LAM"] = (
        "GFS"
    )

    # === Land / soil options ===
    nsoill_out: Literal[4, 9] = 4
    sotyp_from_climo: bool = True
    vgtyp_from_climo: bool = True
    vgfrc_from_climo: bool = True
    lai_from_climo: bool = True
    minmax_vgfrc_from_climo: bool = True
    tg3_from_soil: bool = True
    wam_cold_start: bool = False


def load_yml(n_tiles: int, chgres_config: str) -> dict:
    """Load and validate a YAML configuration for CHGRES with normal tiles."""

    if not chgres_config:
        log.info("No chgres_cube configuration provided. Using default settings.")
        chgres_config = state.configs / "chgres_cube_default.yaml"

    with open(chgres_config, "r") as f:
        yc = dict(yaml.safe_load(f))

    # -------------------------
    # Valid keys
    # -------------------------

    valid_keys = ["global", "regional", "tileX"] + [
        f"tile{i}" for i in range(7, 7 + n_tiles)
    ]

    invalid_keys = [k for k in yc if k not in valid_keys]
    if invalid_keys:
        raise KeyError(
            f"{chgres_config}: invalid keys {invalid_keys} Expected one or more of {valid_keys}."
        )

    # -------------------------
    # Handle zero-tile case
    # -------------------------
    if n_tiles == 0:
        return {k: v for k, v in yc.items() if not k.startswith("tile")}

    # -------------------------
    # Expand tileX template
    # -------------------------
    tileX = yc.pop("tileX", None)

    tiles = {}
    for i in range(n_tiles):
        tile = 7 + i
        key = f"tile{tile}"
        if key in yc:
            tiles[key] = yc[key]
        elif tileX is not None:
            tiles[key] = copy.deepcopy(tileX)
        else:
            raise KeyError(f"Missing configuration for {key} and no tileX provided.")

    # -------------------------
    # Assemble output
    # -------------------------
    out = {k: v for k, v in yc.items() if not k.startswith("tile")}
    out.update(tiles)

    return out


def run_chgres_cube() -> None:
    env_setup()

    yml_configs = load_yml(state.n_nests, state.chgres_config)
    state["external_ic_source"] = {}

    # Determine IC directory based on run_chgres_only flag
    ic_dir = state.tmp / "ic"

    # Prepare fort.41 configuration
    f41 = FV3State(asdict(ChgresCubeConfig()))

    # Normalize tuple values for YAML compatibility
    for k, v in f41.items():
        if isinstance(v, tuple):
            f41[k] = list(v)

    if not state.levels:
        raise ValueError("Vertical levels  must be specified in run_config.yaml")

    log.info("Running chgres_cube to generate initial conditions")

    f41.cycle_year = state.init_datetime.year
    f41.cycle_mon = state.init_datetime.month
    f41.cycle_day = state.init_datetime.day
    f41.cycle_hour = state.init_datetime.hour
    f41.orog_dir_target_grid = ic_dir
    f41.fix_dir_target_grid = ic_dir / "fix_sfc"
    f41.vcoord_file_target_grid = state.fix_am / f"global_hyblev.l{state.levels}.txt"
    f41.varmap_file = state.fix / "varmap_tables" / "GFSphys_var_map.txt"

    # Create symlinks for fix files
    link_fix_files(state.res, f41)

    mosaic_dir = state.tmp / "chgres_cube" / "mosaics"
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    mosaic_file = mosaic_dir / f"C{state.res}_mosaic.nc"

    local_cpus = len(os.sched_getaffinity(0))
    norm_cpu = (local_cpus // 6) * 6
    n_cpus = min(60, norm_cpu)

    for domain, yml_cfg in yml_configs.items():
        domain_f41 = copy.deepcopy(f41)
        yml_cfg = FV3State(yml_cfg)
        tile = None

        # --------------------
        # Domain-specific grid setup
        # --------------------
        if domain == "global":
            orog = [f"oro.C{state.res}.tile{i}.nc" for i in range(1, 7)]
            if state.n_nests > 0:
                mosaic = ic_dir / f"C{state.res}_coarse_mosaic.nc"
            else:
                mosaic = ic_dir / f"C{state.res}_mosaic.nc"

        elif domain.startswith("tile"):
            tile = int(domain.replace("tile", ""))
            nest_idx = f"{tile - 5:02d}"
            mosaic = ic_dir / f"C{state.res}_nested{nest_idx}_mosaic.nc"
            orog = [f"oro.C{state.res}.tile{tile}.nc"]

        elif domain == "regional":
            tile = 7
            mosaic = ic_dir / f"C{state.res}_mosaic.nc"
            orog = [f"oro.C{state.res}.tile7.nc"]
        else:
            raise ValueError(f"Unrecognized domain key: {domain}")

        mosaic_file.unlink(missing_ok=True)
        cp(mosaic, mosaic_file)

        domain_f41.orog_files_target_grid = orog
        domain_f41.mosaic_file_target_grid = mosaic_file

        # --------------------
        # External model handling
        # --------------------

        multi_external_models = yml_cfg.get("external_models")
        if multi_external_models:
            for ext_model_cfg in multi_external_models:
                ext_model = ext_model_cfg.get("external_model")
                ext_model = apply_config_settings(
                    domain,
                    tile,
                    n_cpus,
                    ext_model,
                    ext_model_cfg,
                    domain_f41,
                )
                if ext_model == "GFS":
                    break
        else:
            ext_model = yml_cfg.get("external_model") or f41.external_model
            ext_model = apply_config_settings(
                domain,
                tile,
                n_cpus,
                ext_model,
                yml_cfg,
                domain_f41,
            )

    # set flag indicating IC generation complete
    stage_files()
    state["ic_and_grid_generated"] = True


def apply_config_settings(
    domain: str,
    tile: int,
    n_cpus: int,
    ext_model: str,
    yml_cfg: dict,
    domain_f41: dict,
) -> str:
    if ext_model == "HRRR" and domain.startswith(("tile", "regional")):
        ext_model = validate_hrrr_bounds(tile)

        if ext_model == "HRRR":
            varmap_file = state.fix / "varmap_tables" / "GSDphys_var_map.txt"
            domain_f41.geogrid_file_input_grid = state.fix_am / "geo_em.d01.nc_HRRRX"
            domain_f41.varmap_file = varmap_file
        else:
            # Revert to GFS  settings
            yml_cfg["convert_sfc"] = True
            yml_cfg["convert_atm"] = True

    elif ext_model == "GFS":
        pass

    else:
        raise NotImplementedError("Only GFS and HRRR external models are supported")

    data_dir, data_file = get_init_data(ext_model)
    domain_f41.data_dir_input_grid = data_dir
    domain_f41.grib2_file_input_grid = data_file

    # Override with any YAML-specified values
    for k, v in yml_cfg.items():
        if v is not None:
            domain_f41[k] = v

    if domain not in state["external_ic_source"]:
        state["external_ic_source"][domain] = {"atm": None, "sfc": None, "nst": None}
    if domain_f41.convert_atm:
        state["external_ic_source"][domain]["atm"] = ext_model
    if domain_f41.convert_sfc:
        state["external_ic_source"][domain]["sfc"] = ext_model
    if domain_f41.convert_nst:
        state["external_ic_source"][domain]["nst"] = ext_model

    chgres_exe(domain_f41, n_cpus, domain, ext_model)

    return ext_model


def chgres_exe(input_dict: dict, n_cpus: int, id_name: str, ext_model: str) -> None:
    if id_name.startswith("tile"):
        name = f"nest tile {int(id_name[4:])}"
    else:
        name = str(id_name)

    # check if we are converting atm, sfc, nst or any combination
    converts = []
    if input_dict["convert_atm"] is True:
        converts.append("atm")
    if input_dict["convert_sfc"] is True:
        converts.append("sfc")
    if input_dict["convert_nst"] is True:
        converts.append("nst")
    converts = " and ".join(converts)

    log.debug(f"Running chgres_cube with {ext_model} data for {str(name)} {converts}")
    chgres_cube = state.ufs_exe / "chgres_cube"

    tmp_dir = state.tmp / "chgres_cube" / id_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log_file = state.logs / f"chgres_cube_{id_name}.log"

    # Any instances in fort_41 that are PathLike, convert to str
    for key, value in input_dict.items():
        if isinstance(value, (Path)):
            input_dict[key] = str(value)

    # Write fort.41 namelist
    fort_41 = {"config": input_dict}
    with open(tmp_dir / "fort.41", "w") as f:
        f90nml.write(fort_41, f)

    # Run chgres_cube

    cmd = [*get_launcher(n_cpus), f"{chgres_cube}"]
    result, msgs = run_cmd(cmd, cwd=tmp_dir, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(
            f"chgres_cube failed : {ext_model},  {str(name)},  {converts}"
        )


def link_fix_files(res: int, fort_41: dict) -> None:
    files = Path(fort_41.fix_dir_target_grid).glob("*")
    files = [Path(f) for f in files if f.name.startswith(f"C{res}")]
    symlinks = [f.parent / f.name.replace(f"C{res}", "", 1) for f in files]

    if not files:
        raise ValueError(
            f"No fix files found for resolution C{res} in {fort_41.fix_dir_target_grid}"
        )
    #
    # create symlinks in in fix_dir_target_grid
    # C96.name.tile1.nc -> .name.tile1.nc

    for src, dest in zip(files, symlinks):
        src_path = Path(src)
        dest_path = Path(dest)
        if not dest_path.exists():
            dest_path.symlink_to(src_path.resolve())
