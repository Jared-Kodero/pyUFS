import os
import re
from pathlib import Path

import f90nml
import numpy as np
import yaml
from fv3_runtime import (
    get_stream_handles,
    log,
    read_namelist,
    report_missing_fixed_files,
)
from fv3_state import state
from fv3_timings import get_timings
from fv3_utils import cp, cres_to_deg, env_setup
from regional_bc import BC_INTERVAL_HOURS, HALO_BLEND
from tgrad_perturbations import apply_tgrad_perturbations, tgrad_namelist


def restart_config():

    for f in list(state.work_dir.glob("*.nml")):
        nml = read_namelist(f)

        nml["fv_core_nml"]["warm_start"] = True
        nml["fv_core_nml"]["external_ic"] = False
        nml["fv_core_nml"]["nggps_ic"] = False
        nml["fv_core_nml"]["ncep_ic"] = False

        nml["fv_core_nml"]["mountain"] = True
        nml["fv_core_nml"]["n_zs_filter"] = 0
        nml["fv_core_nml"]["na_init"] = 0
        nml["fv_core_nml"]["make_nh"] = False

        nml["fms_io_nml"]["checksum_required"] = False
        nml.setdefault("fms2_io_nml", {})["checksum_required"] = False
        nml["fms_io_nml"]["restart_checksums_required"] = False
        nml["fms2_io_nml"]["restart_checksums_required"] = False

        with open(f, "w") as nml_out:
            f90nml.write(nml, nml_out)


def update_nml_configs():
    env_setup()

    dt = state.init_datetime

    current_date = [dt.year, dt.month, dt.day, dt.hour, 0, 0]
    state.model_start_date = current_date

    # Do nest namelists
    timings = get_timings()

    log.info("Generating namelist files")

    update_global_nml(
        c_res=state.c_res,
        fhmax=state.run_nhours,
        n_nests=state.n_nests,
        current_date=current_date,
        levels=state.levels,
        refine_ratios=state.refine_ratio,
        do_deep=state.do_deep,
        timings=timings,
    )
    update_nest_nml(
        c_res=state.c_res,
        fhmax=state.run_nhours,
        n_nests=state.n_nests,
        current_date=current_date,
        levels=state.levels,
        refine_ratios=state.refine_ratio,
        do_deep=state.do_deep,
        timings=timings,
    )

    update_table_files()
    update_fixed_files()

    # Update state with the calculated timings
    state.dt_atmos = timings["dt_atmos"]
    state.dt_ocean = timings["dt_ocean"]
    state.k_split = timings["k_split"]
    state.n_split = timings["n_split"]


def disable_deep_convection(nml: dict, tile: int, name: str):
    if name.startswith("nest"):
        i = tile - 7  # index for nests
        refine_ratio = state.refine_ratio

        c_res = state.c_res * refine_ratio[i]
        if state.nest_type == "telescoping":
            c_res = state.c_res * int(np.prod(refine_ratio[: i + 1]))
    else:
        c_res = state.c_res

    do_deep = state.do_deep

    res_km = cres_to_deg(c_res).km
    if do_deep or res_km > 4:
        return nml

    nml["gfs_physics_nml"]["do_deep"] = False
    nml["gfs_physics_nml"]["imfdeepcnv"] = -1  # 2
    nml["gfs_physics_nml"]["shal_cnv"] = True
    nml["gfs_physics_nml"]["imfshalcnv"] = -1  # 2

    log.info(f"{name} deep convection disabled ({res_km:.2f} km)")

    return nml


# for all nests
def common_configs(nml: dict):
    nml["fms_nml"]["domains_stack_size"] = 2**30  # 1 GiB
    nml["fv_core_nml"]["npz"] = state.levels - 1
    nml["external_ic_nml"]["levp"] = state.levels
    nml["fv_core_nml"]["warm_start"] = False

    if state.fv3_debug:
        nml["fv_core_nml"]["fv_debug"] = True
        nml["fv_core_nml"]["print_freq"] = -1

    if state.use_modern_diag:
        nml.setdefault("diag_manager_nml", {})["use_modern_diag"] = True
        nml.setdefault("data_override_nml", {})["use_data_table_yaml"] = True

    return nml


def sync_fhzero(nml: dict) -> dict:
    """Set fhzero = fdiag when output_freq is set (applied after overrides).

    Physics diagnostics reach FMS only every fdiag hours, and bucket fields
    (totprcpb_ave, cnvprcpb_ave) are means since the last fhzero reset. With
    fhzero = fdiag each sample is the exact mean of a non-overlapping fdiag
    window, so the FMS time average over any output interval is exact.
    """
    if state.output_freq is None:
        return nml

    fdiag = nml.get("atmos_model_nml", {}).get("fdiag", 0.0)
    if isinstance(fdiag, list) or float(fdiag) <= 0.0:
        log.warning("fdiag is not a single positive interval; fhzero left unchanged.")
        return nml
    fdiag = float(fdiag)

    n, unit = model_output_interval()
    interval = n * (24.0 if unit == "days" else 1.0)
    if interval % fdiag != 0:
        raise ValueError(
            f"Output interval ({interval} h) must be a multiple of fdiag ({fdiag} h)."
        )

    nml["gfs_physics_nml"]["fhzero"] = fdiag
    return nml


OUTPUT_UNITS = ("hours", "days", "months", "years")


def model_output_interval() -> tuple[int, str]:
    """Interval written by FMS: as configured for hours/days, daily means for
    months/years (aggregated to calendar months/years after the final merge,
    because fixed-length segments cannot align with calendar months).
    """
    if state.output_freq_units in ("months", "years"):
        return 1, "days"
    return int(state.output_freq), state.output_freq_units


def validate_output_config() -> None:
    freq = state.output_freq
    units = state.output_freq_units
    if freq is None and units is None:
        return
    if freq is None or units is None:
        raise ValueError("output_freq and output_freq_units must be set together.")
    if units not in OUTPUT_UNITS:
        raise ValueError(f"output_freq_units must be one of {OUTPUT_UNITS}.")
    if int(freq) < 1:
        raise ValueError("output_freq must be a positive integer.")

    # Averaging windows must not straddle a segment boundary.
    n, unit = model_output_interval()
    interval = n * (24 if unit == "days" else 1)
    if state.run_nhours % interval != 0:
        raise ValueError(
            f"run_nhours ({state.run_nhours}) must be a multiple of the "
            f"averaging interval ({interval} h)."
        )
    if units in ("months", "years") and state.get("merge_freq", -1) != -1:
        raise ValueError(f"output_freq_units: {units} requires merge_freq: -1.")


def set_hist_output(lines: list[str], hist: list[str]) -> list[str]:
    """Set interval and time averaging of the history streams in diag_table."""
    if state.output_freq is None:
        return lines

    n, unit = model_output_interval()
    out = []
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        if cells[0].strip("\"'") in hist and len(cells) >= 6:  # file section
            cells[1] = str(n)
            cells[2] = f'"{unit}"'
            line = ", ".join(cells) + "\n"
        elif len(cells) >= 8 and cells[3].strip("\"'") in hist:  # field section
            cells[5] = ".true."  # time mean over each output interval
            line = " " + ", ".join(cells) + "\n"
        out.append(line)
    return out


def update_global_nml(
    c_res: int,
    fhmax: int,
    n_nests: int,
    current_date: str,
    levels: int,
    refine_ratios: list,
    do_deep: bool,
    timings: dict,
):

    nml_template_path = state.configs / "input_nml.yaml"
    parent_save_path = state.work_dir / "input.nml"
    user_nml = state.run_dir / "input"

    nml = read_namelist(nml_template_path)
    nml = common_configs(nml)

    nml["fv_core_nml"]["target_lat"] = state.target_lat
    nml["fv_core_nml"]["target_lon"] = state.target_lon
    nml["fv_core_nml"]["stretch_fac"] = state.stretch_factor
    nml["coupler_nml"]["current_date"] = current_date
    nml["coupler_nml"]["hours"] = fhmax

    # Use first-guess timings unless overridden by user
    nml["coupler_nml"]["dt_atmos"] = timings["dt_atmos"]
    nml["coupler_nml"]["dt_ocean"] = timings["dt_ocean"]

    # FIX: Pull explicitly from the global keys
    nml["fv_core_nml"]["n_split"] = timings["n_split"][0]
    nml["fv_core_nml"]["k_split"] = timings["k_split"][0]
    nml["fv_core_nml"]["npx"] = state.npx[0]
    nml["fv_core_nml"]["npy"] = state.npy[0]
    nml["fv_core_nml"]["ntiles"] = state.ntiles[0]
    nml["fv_core_nml"]["layout"] = state.layout[0]
    nml["fv_core_nml"]["io_layout"] = state.io_layout[0]
    nml["atmos_model_nml"]["blocksize"] = state.blocksize[0]

    if n_nests > 0:
        nml["fv_nest_nml"]["grid_pes"] = state.grid_pes
        nml["fv_nest_nml"]["nest_refine"] = [0] + state.refine_ratio
        nml["fv_nest_nml"]["num_tile_top"] = 6  # use 7 if regional suppergrid is used
        nml["fv_nest_nml"]["tile_coarse"] = [0] + state.parent_tile
        nml["fv_nest_nml"]["nest_ioffsets"] = state.nest_ioffsets
        nml["fv_nest_nml"]["nest_joffsets"] = state.nest_joffsets
        nml["fv_nest_nml"]["p_split"] = 1

    else:
        del nml["fv_nest_nml"]

    if state.gtype in ("regional_gfdl", "regional_esg"):
        # Standalone regional domain (single tile with a prescribed lateral
        # boundary). regional activates the boundary-forcing path in the
        # dynamical core; bc_update_interval must match the cadence of the
        # boundary files written by regional_bc, and nrows_blend must match the
        # halo_blend width passed to chgres_cube. Reference values follow a
        # working UFS regional input.nml (Harris et al., 2021).
        nml["fv_core_nml"]["regional"] = True
        nml["fv_core_nml"]["ntiles"] = 1
        nml["fv_core_nml"]["bc_update_interval"] = BC_INTERVAL_HOURS
        nml["fv_core_nml"]["nrows_blend"] = HALO_BLEND

    nml = disable_deep_convection(nml, 1, "global")
    nml = update_namsfc(nml)

    # check for nml overrides if user provided external nml
    nml = namelist_overrides(user_nml, nml, "global")
    nml = sync_fhzero(nml)

    with open(parent_save_path, "w") as f:
        f90nml.write(nml, f)

    return 0


def update_nest_nml(
    c_res: int,
    fhmax: int,
    n_nests: int,
    current_date: str,
    levels: int,
    refine_ratios: list,
    do_deep: bool,
    timings: dict,
):
    if n_nests == 0:
        return

    nest_nml_template_path = state.configs / "input_nestXX_nml.yaml"
    save_paths = [
        state.work_dir / f"input_nest{i:02d}.nml" for i in range(2, n_nests + 2)
    ]
    user_nmls = [state.run_dir / f"input_nest{i:02d}" for i in range(2, n_nests + 2)]
    tiles = [7 + i for i in range(n_nests)]

    nest_pes = state.grid_pes  # includes parent tile pes
    nest_pes = nest_pes[1:]

    validate = (
        len(save_paths) == len(tiles) == len(refine_ratios) == len(nest_pes) == n_nests
    )

    if not validate:
        raise ValueError(
            "Mismatch between number of nests, nest resolutions, tiles, and refine ratios."
        )

    for i, (out_file, user_nml, tile) in enumerate(
        zip(save_paths, user_nmls, tiles), start=1
    ):
        nml = read_namelist(nest_nml_template_path)
        nml = common_configs(nml)
        nml = disable_deep_convection(nml, tile, f"nest{i + 1:02d}")

        # Use first-guess timings unless overridden by user

        nml["fv_core_nml"]["n_split"] = timings["n_split"][i]
        nml["fv_core_nml"]["k_split"] = timings["k_split"][i]

        # Assign calculated values to namelist, add +1 to skip the global tile
        nml["fv_core_nml"]["npx"] = state.npx[i]
        nml["fv_core_nml"]["npy"] = state.npy[i]
        nml["fv_core_nml"]["ntiles"] = state.ntiles[i]
        nml["fv_core_nml"]["layout"] = state.layout[i]
        nml["fv_core_nml"]["io_layout"] = state.io_layout[i]
        nml["atmos_model_nml"]["blocksize"] = state.blocksize[i]

        nml = update_namsfc(nml)

        nml = namelist_overrides(user_nml, nml, f"nest{i + 1:02d}")
        nml = sync_fhzero(nml)

        with open(out_file, "w") as f:
            f90nml.write(nml, f)

    return 0


def namelist_overrides(path: Path, nml: dict, name: str):

    suffixes = (".nml", ".yaml", ".yml")

    for suffix in suffixes:
        _path = Path(path).with_suffix(suffix)
        if not _path.exists():
            continue
        override_nml = read_namelist(_path)

        if not override_nml:
            log.info(f"Namelist file: {path} is empty !")
            return nml

        log.info(f"Applying {name} nml overrides from: {path}")

        for section, entries in override_nml.items():
            if not entries:
                continue

            nml.setdefault(section, {}).update(entries)

        break  # Exit after the first matching suffix is found

    return nml


def update_fixed_files():
    dt = state.init_datetime
    year = dt.year
    fix_dir = state.fix_src / "am"

    required_files = [
        "aerosol.dat",
        f"co2historicaldata_{year}.txt",
        "co2historicaldata_glob.txt",
        "co2monthlycyc.txt",
        "sfc_emissivity_idx.txt",
        "solarconstant_noaa_an.txt",
        "volcanic_aerosols_1990-1999.txt",
        "global_h2oprdlos.f77",
        "global_o3prdlos.f77",
    ]

    missing_files = []

    for name in required_files:
        file = fix_dir / name
        if file.exists():
            dest = state.fix / name
            if not dest.exists():
                cp(file, dest)

            link = Path(state.input) / name
            link.unlink(missing_ok=True)

            rel_target = os.path.relpath(dest, start=state.input)
            link.symlink_to(rel_target)
        else:
            missing_files.append(file)

    if missing_files:
        report_missing_fixed_files(missing_files, sub_dir="am")


def update_table_files():

    update_fixed_files()

    field_table_path = state.work_dir / "field_table.yaml"
    user_field = state.run_dir / "field_table"
    template_field = state.configs / "field_table.yaml"
    field_file = user_field if user_field.exists() else template_field
    cp(field_file, field_table_path)

    validate_output_config()

    if state.use_modern_diag:
        (state.work_dir / "diag_table").unlink(missing_ok=True)
        write_diag_table_yaml()
        write_data_table_yaml()
    else:
        for f in state.work_dir.glob("diag_table*.yaml"):
            f.unlink()
        for f in state.work_dir.glob("data_table*.yaml"):
            f.unlink()
        write_diag_table_legacy()


def history_streams(streams: list[str]) -> list[str]:
    return [s for s in streams if "spec" not in s and "static" not in s]


def write_diag_table_legacy():
    """Stage the legacy ASCII diag_table (case-local diag_table or template)."""
    dt = state.init_datetime
    restart_no = state.get("restart_no", 0)

    diag_table_path = state.work_dir / "diag_table"
    user_diag = state.run_dir / "diag_table"
    template_diag = state.configs / "diag_table"
    cp(user_diag if user_diag.exists() else template_diag, diag_table_path)

    streams = get_stream_handles()

    with open(diag_table_path) as f:
        lines = f.readlines()
        lines = [line for line in lines if line and not line.strip().startswith("#")]
        lines = lines[2:]  # Skip the first two lines (title and base_date)

    dt_str = f"{dt.year} {dt.month:02d} {dt.day:02d} {dt.hour:02d} 0 0\n"
    desc_str = f"{state.description}\n"

    lines = set_hist_output(lines, history_streams(streams))

    # Match quoted handles exactly, so a handle that prefixes another
    # (fv3_hist vs fv3_hist_prcp) cannot duplicate or corrupt lines.
    renamed = []
    for line in lines:
        for stream in streams:
            pattern = rf"([\"']){re.escape(stream)}\1"
            if re.search(pattern, line):
                renamed.append(
                    re.sub(pattern, rf"\g<1>HIST/{stream}.{restart_no:02d}\g<1>", line)
                )
                break

    lines = [desc_str, dt_str] + renamed

    with open(diag_table_path, "w") as f:
        f.writelines(lines)


def instance_copies(path: Path) -> list[Path]:
    """path plus one copy name per nest (diag_table.nest02.yaml, ...).

    FMS looks for the instance file first once the nest filename appendix is
    set; identical copies keep every domain on the same table.
    """
    n_nests = int(state.n_nests or 0)
    return [path] + [
        path.with_name(f"{path.stem}.nest{i:02d}{path.suffix}")
        for i in range(2, n_nests + 2)
    ]


def write_diag_table_yaml():
    """Stage the case-local diag_table.yaml (modern diag manager).

    Applies the same changes as the legacy path: title, base_date, output
    interval and time averaging of the history files, and the
    HIST/<file>.<restart> file names.
    """
    dt = state.init_datetime
    restart_no = state.get("restart_no", 0)

    user_diag = state.run_dir / "diag_table.yaml"
    if not user_diag.exists():
        raise FileNotFoundError(
            f"use_modern_diag: true requires {user_diag}. Write one, or convert a "
            "legacy table with fms_yaml_tools: diag-table-to-yaml diag_table"
        )
    with open(user_diag) as f:
        table = yaml.safe_load(f)

    table["title"] = str(state.description)
    table["base_date"] = f"{dt.year} {dt.month} {dt.day} {dt.hour} 0 0"

    names = [d["file_name"] for d in table["diag_files"]]
    hist = history_streams(names)
    for diag_file in table["diag_files"]:
        name = diag_file["file_name"]
        if state.output_freq is not None and name in hist:
            n, unit = model_output_interval()
            diag_file["freq"] = f"{n} {unit}"
            for var in diag_file.get("varlist", []):
                var["reduction"] = "average"
        diag_file["file_name"] = f"HIST/{name}.{restart_no:02d}"

    for path in instance_copies(state.work_dir / "diag_table.yaml"):
        with open(path, "w") as f:
            yaml.safe_dump(table, f, default_flow_style=False, sort_keys=False)


def write_data_table_yaml():
    """Stage the case-local data_table.yaml when it defines entries.

    Without one no file is staged; FMS then runs with an empty data table.
    """
    for f in state.work_dir.glob("data_table*.yaml"):
        f.unlink()

    user_data = state.run_dir / "data_table.yaml"
    if not user_data.exists():
        return
    with open(user_data) as f:
        table = yaml.safe_load(f) or {}

    if not table.get("data_table"):
        return

    for path in instance_copies(state.work_dir / "data_table.yaml"):
        with open(path, "w") as f:
            yaml.safe_dump(table, f, default_flow_style=False, sort_keys=False)


def update_namsfc(nml):

    am_dir = Path(state.fix_src) / "am"

    namsfc = {
        "fnacna": "",
        "fnsnoa": "",
        "fntsfa": "",
        "fnzorc": "igbp",
        "fabsl": 99999,
        "faisl": 99999,
        "faiss": 99999,
        "fsicl": 99999,
        "fsics": 99999,
        "fslpl": 99999,
        "fsnol": 99999,
        "fsnos": 99999,
        "fsotl": 99999,
        "ftsfl": 99999,
        "ftsfs": 90,
        "fvetl": 99999,
        "fvmnl": 99999,
        "fvmxl": 99999,
        "ldebug": False,
        "fsmcl": [99999, 99999, 99999],
    }

    namsfc_files = {
        "fnabsc": "global_mxsnoalb.uariz.t1534.3072.1536.rg.grb",
        "fnaisc": "CFSR.SEAICE.1982.2012.monthly.clim.grb",
        "fnalbc": "global_snowfree_albedo.bosu.t1534.3072.1536.rg.grb",
        "fnalbc2": "global_albedo4.1x1.grb",
        "fnglac": "global_glacier.2x2.grb",
        "fnmldc": "mld_DR003_c1m_reg2.0.grb",
        "fnmskh": "seaice_newland.grb",
        "fnmxic": "global_maxice.2x2.grb",
        "fnslpc": "global_slope.1x1.grb",
        "fnsmcc": "global_soilmgldas.t1534.3072.1536.grb",
        "fnsnoc": "global_snoclim.1.875.grb",
        "fnsotc": "global_soiltype.statsgo.t1534.3072.1536.rg.grb",
        "fntg3c": "global_tg3clim.2.6x1.5.grb",
        "fntsfc": "RTGSST.1982.2012.monthly.clim.grb",
        "fnvegc": "global_vegfrac.0.144.decpercent.grb",
        "fnvetc": "global_vegtype.igbp.t1534.3072.1536.rg.grb",
        "fnvmnc": "global_shdmin.0.144x0.144.grb",
        "fnvmxc": "global_shdmax.0.144x0.144.grb",
    }
    missing_files = []
    for key, fname in namsfc_files.items():
        src = am_dir / fname
        dst = state.fix / fname

        if not src.exists():
            missing_files.append(src)
            continue
        if not dst.exists():
            cp(src, dst)

        namsfc[key] = f"FIXED/{fname}"

    nml["namsfc"] = namsfc

    if missing_files:
        report_missing_fixed_files(missing_files, sub_dir="am")

    # Rewrite the FIXED SST/ice climatologies (perturbed or pristine) and set
    # the surface-cycle options the experiment depends on.
    apply_tgrad_perturbations()
    nml = tgrad_namelist(nml)

    return nml
