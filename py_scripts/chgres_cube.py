from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

import f90nml

import regional_bc
from fv3_ic_data import get_ic_data, validate_hrrr_bounds
from fv3_runtime import get_launcher, read_namelist
from fv3_stage_data import stage_files
from fv3_state import state
from fv3_utils import cp, env_setup, run_cmd

log = logging.getLogger("PREPROCESS")


@dataclass
class ChgresCubeConfig:
    # === Target grid ===
    mosaic_file_target_grid: Path | None = None
    fix_dir_target_grid: Path | None = None
    orog_dir_target_grid: Path | None = None
    orog_files_target_grid: list[str] = None
    vcoord_file_target_grid: Path | None = None

    # === Input grid ===
    data_dir_input_grid: Path | None = None
    atm_files_input_grid: list[str] = None
    sfc_files_input_grid: list[str] = None
    nst_files_input_grid: list[str] = None
    atm_core_files_input_grid: list[str] = None
    atm_tracer_files_input_grid: list[str] = None
    orog_dir_input_grid: Path | None = None
    orog_files_input_grid: list[str] = None
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


# Valid override keys accepted from a user config block.
CONFIG_FIELDS = {f.name for f in fields(ChgresCubeConfig)}

# Nest atmosphere source when a nest config sets no external_model. A nest
# inside HRRR coverage takes the HRRR atmosphere (GFS surface); a nest outside
# coverage falls back to GFS. Regional and global domains still default to GFS.
# To disable the HRRR default for a nest, set external_model in its per-domain
# config file (chgres_cube_nest<NN>.yaml, or fort_nest<NN>.41).
NEST_DEFAULT_MODEL = "HRRR"


@dataclass(frozen=True)
class RunSpec:
    """One fully populated chgres_cube invocation plus its domain label.

    source_mosaic is the per-domain mosaic on disk; it is copied onto the
    shared canonical path in config.mosaic_file_target_grid at run time.
    """

    domain: str
    source_mosaic: Path
    config: ChgresCubeConfig


def config_candidates(domain: str) -> list[str]:
    """Accepted filenames for a domain, in precedence order (YAML then namelist)."""
    if domain in ("global", "regional"):
        return ["chgres_cube.yaml", "fort.41"]
    nn = domain.removeprefix("nest")  # e.g. "02"
    return [f"chgres_cube_nest{nn}.yaml", f"fort_nest{nn}.41"]


def load_block(domain: str) -> dict:
    """Load a domain's flat config block, or an empty block if no file is present.

    Accepted files (YAML preferred over namelist) are read verbatim. When none
    exist, an empty block is returned so the built-in default applies.
    """
    for name in config_candidates(domain):
        path = state.run_dir / name
        if path.exists():
            block = {k: v for k, v in read_namelist(path).items() if v is not None}
            unknown = set(block) - CONFIG_FIELDS
            if unknown:
                raise KeyError(f"{path}: unknown chgres_cube keys {sorted(unknown)}")
            return block
    return {}


def resolve_model(requested: str, domain: str, tile: int | None) -> str:
    """Resolve a requested external model, downgrading HRRR to GFS off-coverage."""
    requested = requested.upper()
    if requested == "HRRR":
        if tile is None or tile < 7:
            raise ValueError(
                f"HRRR ICs are only supported for nested/regional domains, got {domain}"
            )
        return validate_hrrr_bounds(tile)
    if requested == "GFS":
        return "GFS"
    raise NotImplementedError(
        f"Only GFS and HRRR external models are supported, got {requested}"
    )


def resolve_atm_model(requested: str, domain: str, tile: int, explicit: bool) -> str:
    """Resolve a limited-area atmosphere model (nest or regional).

    GFS is always accepted. HRRR is accepted only inside HRRR coverage. An
    explicit external_model: HRRR outside coverage raises. A defaulted HRRR
    (nest with no user config) outside coverage falls back to GFS silently; the
    fallback is announced once up front by run_chgres_cube.
    """
    requested = requested.upper()
    if requested == "GFS":
        return "GFS"
    if requested == "HRRR":
        if validate_hrrr_bounds(tile) == "HRRR":
            return "HRRR"
        if explicit:
            raise ValueError(
                f"{domain}: external_model HRRR requested but tile {tile} lies outside HRRR coverage"
            )
        return "GFS"
    raise NotImplementedError(
        f"Only GFS and HRRR external models are supported, got {requested}"
    )


def plan_runs(domain: str, tile: int | None, block: dict) -> list[tuple[str, dict]]:
    """Return (external_model, override_dict) for each run of a domain.

    Global takes its model and convert flags from the file, or GFS with default
    flags when the file is absent, as a single conversion.

    Nests and regional domains (limited-area, tile >= 7) may take the atmosphere
    from HRRR but always take the surface from GFS, because HRRR carries no soil
    levels. With no user config a nest defaults to HRRR and a regional domain
    defaults to GFS. A defaulted HRRR nest outside HRRR coverage falls back to
    GFS silently (announced once up front by run_chgres_cube); an explicit
    external_model: HRRR outside coverage raises. An HRRR atmosphere yields two
    conversions (HRRR atmosphere, GFS surface); a GFS atmosphere yields one
    combined conversion. The convert switches follow this split and are not read
    from the file; all other file keys are applied as overrides.
    """
    if domain == "global":
        resolved = resolve_model(block.get("external_model", "GFS"), domain, tile)
        overrides = {k: v for k, v in block.items() if k != "external_model"}
        return [(resolved, overrides)]

    default_model = NEST_DEFAULT_MODEL if domain.startswith("nest") else "GFS"
    atm_model = resolve_atm_model(
        block.get("external_model", default_model),
        domain,
        tile,
        explicit="external_model" in block,
    )
    plans = (
        [("HRRR", True, False), ("GFS", False, True)]
        if atm_model == "HRRR"
        else [("GFS", True, True)]
    )

    plan_controlled = {"external_model", "convert_atm", "convert_sfc", "convert_nst"}
    file_overrides = {k: v for k, v in block.items() if k not in plan_controlled}
    return [
        (
            model,
            {
                **file_overrides,
                "convert_atm": ca,
                "convert_sfc": cs,
                "convert_nst": False,
            },
        )
        for model, ca, cs in plans
    ]


def nests_defaulting_to_hrrr() -> list[str]:
    """Return nest domains that set no external_model and take the HRRR default.

    Empty for non-nested grids or when every nest sets external_model
    explicitly, in which case no HRRR-default notice is emitted.
    """
    if state.gtype != "nest":
        return []
    domains = [f"nest{n:02d}" for n in range(2, 2 + state.n_nests)]
    return [d for d in domains if "external_model" not in load_block(d)]


def build_run_specs() -> list[RunSpec]:
    """Enumerate domains from state.gtype and expand each into full configs.

    uniform / stretch : global grid on tiles 1-6, no nests.
    nest              : global coarse grid plus one nest per additional tile.
    regional_*        : a single limited-area domain on tile 7.
    """
    res = state.c_res
    ic_dir = state.tmp / "input"
    canonical_mosaic = state.tmp / "chgres_cube" / "mosaics" / f"C{res}_mosaic.nc"
    tiles_1_6 = [f"oro.C{res}.tile{i}.nc" for i in range(1, 7)]

    # (domain, tile, source mosaic, orography tiles).
    if state.gtype in ("uniform", "stretch"):
        grids = [("global", None, ic_dir / f"C{res}_mosaic.nc", tiles_1_6)]

    elif state.gtype == "nest":
        grids = [("global", None, ic_dir / f"C{res}_coarse_mosaic.nc", tiles_1_6)]
        for n in range(2, 2 + state.n_nests):
            tile = n + 5
            grids.append(
                (
                    f"nest{n:02d}",
                    tile,
                    ic_dir / f"C{res}_nested{n:02d}_mosaic.nc",
                    [f"oro.C{res}.tile{tile}.nc"],
                )
            )

    elif state.gtype in ("regional_gfdl", "regional_esg"):
        grids = [
            (
                "regional",
                7,
                ic_dir / f"C{res}_mosaic.nc",
                [f"C{res}_oro_data.tile7.halo0.nc"],
            )
        ]

    else:
        raise ValueError(f"Unrecognized grid type: {state.gtype}")

    base = ChgresCubeConfig(
        cycle_year=state.init_datetime.year,
        cycle_mon=state.init_datetime.month,
        cycle_day=state.init_datetime.day,
        cycle_hour=state.init_datetime.hour,
        orog_dir_target_grid=ic_dir,
        fix_dir_target_grid=ic_dir / "fix_sfc",
        vcoord_file_target_grid=state.fix_src
        / "am"
        / f"global_hyblev.l{state.levels}.txt",
    )

    specs: list[RunSpec] = []
    for domain, tile, mosaic, orog in grids:
        block = load_block(domain)  # empty dict falls back to the built-in default
        for model, overrides in plan_runs(domain, tile, block):
            data_dir, data_file = get_ic_data(
                model, state.init_datetime, state.forecast_hour
            )
            varmap_dir = state.fix_src / "varmap_tables"
            geogrid_file_input_grid = None

            if model == "HRRR":
                varmap_file = varmap_dir / "GSDphys_var_map.txt"
                geogrid_file_input_grid = state.fix / "am" / "geo_em.d01.nc_HRRRX"
            else:
                varmap_file = varmap_dir / "GFSphys_var_map.txt"

            settings = {
                "mosaic_file_target_grid": canonical_mosaic,
                "orog_files_target_grid": orog,
                "external_model": model,
                "varmap_file": varmap_file,
                "geogrid_file_input_grid": geogrid_file_input_grid,
                "data_dir_input_grid": data_dir,
                "grib2_file_input_grid": data_file,
            }
            if domain == "regional":
                # regional = 1 produces the interior IC and the hour-0 boundary;
                # halo_bndy is the lateral halo (halo + 1 rows) that the model
                # reads, halo_blend the tendency-blend width. regional_bc reuses
                # this config for the boundary-only passes.
                settings.update(
                    regional=regional_bc.REGIONAL_IC,
                    halo_bndy=state.halo + 1,
                    halo_blend=regional_bc.HALO_BLEND,
                )
            settings.update(overrides)  # file values win over derived defaults
            cfg = replace(base, **settings)
            specs.append(RunSpec(domain=domain, source_mosaic=mosaic, config=cfg))
    return specs


def run_chgres(spec: RunSpec, n_cpus: int) -> Path:
    """Stage the mosaic, record IC provenance, write fort.41, and run chgres_cube.

    Returns the working directory holding the run output, so the caller can
    collect per-domain products such as the regional boundary file.
    """
    cfg = spec.config
    domain = spec.domain
    model = cfg.external_model

    tmp_dir = state.tmp / "chgres_cube" / domain
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fields_on = [
        ("atm", cfg.convert_atm),
        ("sfc", cfg.convert_sfc),
        ("nst", cfg.convert_nst),
    ]
    active = [name for name, on in fields_on if on]
    converts = " and ".join(active)

    # Distinct log per converted-field set so a nest's HRRR (atm) and GFS (sfc)
    # runs write separate files instead of overwriting one another, e.g.
    # chgres_cube_nest02_atm.log and chgres_cube_nest02_sfc.log. The log lives in
    # state.logs, separate from the executable's working directory, so this name
    # never affects how chgres_cube resolves its inputs.
    log_file = state.logs / f"chgres_cube_{domain}_{'_'.join(active) or 'none'}.log"

    # Stage this domain's mosaic onto the shared canonical path.
    canonical = Path(cfg.mosaic_file_target_grid)
    canonical.unlink(missing_ok=True)
    cp(spec.source_mosaic, canonical)

    # Record which model supplied each field group.
    key = f"{domain}_ic_source"
    if key not in state:
        state[key] = {"atm": None, "sfc": None, "nst": None}
    for name, on in fields_on:
        if on:
            state[key][name] = model

    # Serialise the dataclass, coercing Path to str and tuple to list for f90nml.
    # fort.41 stays in tmp_dir as in the original: consumed by the launch below
    # before any later run for the same domain rewrites it, so no unsafe clobber.
    config = {
        k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
        for k, v in asdict(cfg).items()
    }
    with open(tmp_dir / "fort.41", "w") as fh:
        f90nml.write({"config": config}, fh)

    cmd = [*get_launcher(n_cpus), str(state.ufs_exe / "chgres_cube")]
    result, msgs = run_cmd(cmd, cwd=tmp_dir, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"chgres_cube failed: {model}, {domain}, {converts}")

    return tmp_dir


def run_chgres_cube() -> None:
    env_setup()

    if not state.levels:
        raise ValueError("Vertical levels must be specified in run_config.yaml")

    link_fix_files(state.c_res, state.tmp / "input" / "fix_sfc")

    (state.tmp / "chgres_cube" / "mosaics").mkdir(parents=True, exist_ok=True)

    specs = build_run_specs()
    n_cpus = min(60, (len(os.sched_getaffinity(0)) // 6) * 6)

    log.info("Running chgres_cube to generate initial conditions")

    # Announce the HRRR nest default once, only when nests are present and no
    # external_model is configured for them. Nests outside HRRR coverage fall
    # back to GFS silently; this notice covers that case in place of a per-nest
    # warning.
    defaulting = nests_defaulting_to_hrrr()
    if defaulting:
        log.info(
            f"No external_model configured for {', '.join(defaulting)}; will "
            + "try HRRR and default to GFS if nest is not fully within HRRR domain"
        )

    for spec in specs:
        run_chgres(spec, n_cpus)

    if state.gtype in ("regional_gfdl", "regional_esg"):
        regional = next(spec for spec in specs if spec.domain == "regional")

        def run_bc(domain_label: str, source_mosaic, config) -> Path:
            spec = RunSpec(
                domain=domain_label, source_mosaic=source_mosaic, config=config
            )
            return run_chgres(spec, n_cpus)

        regional_bc.generate_boundary_files(
            regional.config, regional.source_mosaic, run_bc
        )

    stage_files()
    state.generate_ic_data = False


def link_fix_files(c_res: int, fix_dir: Path) -> None:
    """Symlink resolution-prefixed fix files to their unprefixed names.

    C96.name.tile1.nc -> .name.tile1.nc
    """
    files = [f for f in Path(fix_dir).glob("*") if f.name.startswith(f"C{c_res}")]
    if not files:
        raise ValueError(f"No fix files found for resolution C{c_res} in {fix_dir}")

    for src in files:
        dest = src.parent / src.name.replace(f"C{c_res}", "", 1)
        if not dest.exists():
            dest.symlink_to(src.resolve())
