from __future__ import annotations

import copy
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import f90nml
from fv3_ic_data import get_ic_data, validate_hrrr_bounds
from fv3_runtime import get_launcher, log, read_namelist
from fv3_stage_data import stage_files
from fv3_state import FV3State, state
from fv3_utils import cp, env_setup, run_cmd

log = logging.getLogger("PREPROCESS")


@dataclass
class ChgresCubeConfig:
    # === Target grid ===
    mosaic_file_target_grid: Path = None
    fix_dir_target_grid: Path = None
    orog_dir_target_grid: Path = None
    orog_files_target_grid: list[str] = None
    vcoord_file_target_grid: Path = None

    # === Input grid ===
    data_dir_input_grid: Path = None
    atm_files_input_grid: list[str] = None
    sfc_files_input_grid: list[str] = None
    nst_files_input_grid: list[str] = None
    atm_core_files_input_grid: list[str] = None
    atm_tracer_files_input_grid: list[str] = None
    orog_dir_input_grid: Path = None
    orog_files_input_grid: list[str] = None
    grib2_file_input_grid: Path = None
    geogrid_file_input_grid: Path = None
    mosaic_file_input_grid: Path = None

    # === Physics / mapping ===
    varmap_file: Path = None
    thomp_mp_climo_file: Path = None
    wam_parm_file: Path = None

    # === Cycle ===
    cycle_year: int = None
    cycle_mon: int = None
    cycle_day: int = None
    cycle_hour: int = None

    # === Conversion flags ===
    convert_atm: bool = True
    convert_sfc: bool = True
    convert_nst: bool = False

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


def load_yml(n_tiles: int) -> dict:
    """Load and validate a YAML configuration for CHGRES with normal tiles."""

    fort_41 = Path(state.run_dir / "fort.41")
    yml_41 = Path(state.run_dir / "chgres_cube.yaml")
    fort_41_exists = fort_41.exists()
    yml_41_exists = yml_41.exists()

    config_path = fort_41 if fort_41_exists else yml_41 if yml_41_exists else None

    if config_path:
        cfg = read_namelist(config_path)

        valid_keys = ["global", "regional", "nestXX"] + [
            f"nest{nest_idx:02d}" for nest_idx in range(2, 2 + n_tiles)
        ]

        invalid_keys = [k for k in cfg if k not in valid_keys]
        if invalid_keys:
            raise KeyError(
                f"{config_path}: invalid keys {invalid_keys} Expected one or more of {valid_keys}."
            )

    else:
        cfg = {
            "global": {
                "external_model": "GFS",
                "convert_atm": True,
                "convert_sfc": True,
                "convert_nst": False,
            },
            "nestXX": {
                "external_models": [
                    {
                        "external_model": "HRRR",
                        "convert_atm": True,
                        "convert_sfc": False,
                        "convert_nst": False,
                    },
                    {
                        "external_model": "GFS",
                        "convert_atm": False,
                        "convert_sfc": True,
                        "convert_nst": False,
                    },
                ]
            },
        }

    # -------------------------
    # Valid keys
    # -------------------------

    # -------------------------
    # Handle zero-tile case
    # -------------------------
    if n_tiles == 0:
        return {k: v for k, v in cfg.items() if not k.startswith("nest")}

    # -------------------------
    # Expand nestXX template
    # -------------------------
    nestXX = cfg.pop("nestXX", None)

    nests = {}
    for nest_idx in range(2, 2 + n_tiles):
        key = f"nest{nest_idx:02d}"

        if key in cfg:
            nests[key] = cfg[key]
        elif nestXX is not None:
            nests[key] = copy.deepcopy(nestXX)
        else:
            raise KeyError(f"Missing configuration for {key} and no nestXX provided.")
    # -------------------------
    # Assemble output
    # -------------------------
    out = {k: v for k, v in cfg.items() if not k.startswith("nest")}
    out.update(nests)

    return out


def run_chgres_cube() -> None:
    env_setup()

    yml_configs = load_yml(state.n_nests)
    state.external_ic_source = {}

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
    f41.vcoord_file_target_grid = (
        state.fix / "am" / f"global_hyblev.l{state.levels}.txt"
    )
    f41.varmap_file = state.fix_src / "varmap_tables" / "GFSphys_var_map.txt"

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

        elif domain.startswith("nest"):
            nest_idx = domain.replace("nest", "")
            tile = int(nest_idx) + 5
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
                requested_model = ext_model_cfg["external_model"]

                # Fresh namelist state for every source-model conversion.
                # This prevents HRRR-specific settings from leaking into GFS.
                model_f41 = copy.deepcopy(domain_f41)

                apply_config_settings(
                    domain=domain,
                    tile=tile,
                    n_cpus=n_cpus,
                    ext_model=requested_model,
                    yml_cfg=ext_model_cfg,
                    domain_f41=model_f41,
                )
        else:
            requested_model = yml_cfg.get("external_model") or f41.external_model
            model_f41 = copy.deepcopy(domain_f41)

            apply_config_settings(
                domain=domain,
                tile=tile,
                n_cpus=n_cpus,
                ext_model=requested_model,
                yml_cfg=yml_cfg,
                domain_f41=model_f41,
            )

    # set flag indicating IC generation complete
    stage_files()
    state.generate_ic_data = False


def apply_config_settings(
    domain: str,
    tile: int | None,
    n_cpus: int,
    ext_model: str,
    yml_cfg: dict,
    domain_f41: dict,
) -> str:
    requested_model = ext_model.upper()

    # Resolve the actual model to use.
    if requested_model == "HRRR":
        if tile is None or tile < 7:
            raise ValueError(
                f"HRRR ICs are only supported for nested/regional domains, got {domain}"
            )

        resolved_model = validate_hrrr_bounds(tile)

    elif requested_model == "GFS":
        resolved_model = "GFS"

    else:
        raise NotImplementedError(
            f"Only GFS and HRRR external models are supported, got {requested_model}"
        )

    # Set model-specific defaults before applying YAML overrides.
    if resolved_model == "HRRR":
        domain_f41.varmap_file = state.fix_src / "varmap_tables" / "GSDphys_var_map.txt"
        domain_f41.geogrid_file_input_grid = state.fix / "am" / "geo_em.d01.nc_HRRRX"

    elif resolved_model == "GFS":
        domain_f41.varmap_file = state.fix_src / "varmap_tables" / "GFSphys_var_map.txt"
        domain_f41.geogrid_file_input_grid = None

    # Apply all YAML values except external_model.
    # external_model must reflect resolved_model, not the requested model.
    for key, value in yml_cfg.items():
        if key == "external_model" or value is None:
            continue
        domain_f41[key] = value

    domain_f41.external_model = resolved_model

    data_dir, data_file = get_ic_data(resolved_model)
    domain_f41.data_dir_input_grid = data_dir
    domain_f41.grib2_file_input_grid = data_file

    # Record provenance only after all settings are finalized.
    if domain not in state.external_ic_source:
        state.external_ic_source[domain] = {
            "atm": None,
            "sfc": None,
            "nst": None,
        }

    if domain_f41.convert_atm:
        state.external_ic_source[domain]["atm"] = resolved_model

    if domain_f41.convert_sfc:
        state.external_ic_source[domain]["sfc"] = resolved_model

    if domain_f41.convert_nst:
        state.external_ic_source[domain]["nst"] = resolved_model

    chgres_exe(domain_f41, n_cpus, domain, resolved_model)

    return resolved_model


def chgres_exe(input_dict: dict, n_cpus: int, domain: str, ext_model: str) -> None:

    # check if we are converting atm, sfc, nst or any combination
    converts = []
    if input_dict["convert_atm"] is True:
        converts.append("atm")
    if input_dict["convert_sfc"] is True:
        converts.append("sfc")
    if input_dict["convert_nst"] is True:
        converts.append("nst")
    converts = " and ".join(converts)

    chgres_cube = state.ufs_exe / "chgres_cube"

    tmp_dir = state.tmp / "chgres_cube" / domain
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log_file = state.logs / f"chgres_cube_{domain}.log"

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
            f"chgres_cube failed : {ext_model},  {str(domain)},  {converts}"
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
