from pathlib import Path

import f90nml
import numpy as np
from fv3_diag_files import update_table_files
from fv3_fixed_files import update_fixed_files
from fv3_runtime import log, read_namelist, report_missing_fixed_files
from fv3_state import state
from fv3_timings import get_timings
from fv3_utils import cp, cres_to_deg, env_setup
from regional_bc import BC_INTERVAL_HOURS, HALO_BLEND


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
        n_nests=state.n_nests,
        current_date=current_date,
        levels=state.levels,
        refine_ratios=state.refine_ratio,
        do_deep=state.do_deep,
        timings=timings,
    )
    update_nest_nml(
        c_res=state.c_res,
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


def coupler_run_length() -> tuple[str, int]:
    """coupler_nml run length. The SHiELD coupler ignores years, so they are
    passed as months; month runs must start on day 28 or earlier."""
    n, units = state.run_length, state.run_length_units
    if units == "years":
        n, units = 12 * n, "months"
    if units == "months" and state.init_datetime.day > 28:
        raise ValueError("run_length_units months/years requires a start day <= 28")
    return units, n


def update_global_nml(
    c_res: int,
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
    units, n = coupler_run_length()
    nml["coupler_nml"][units] = n

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

    with open(parent_save_path, "w") as f:
        f90nml.write(nml, f)

    return 0


def update_nest_nml(
    c_res: int,
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

    check_fhzero(nml, name)
    return nml


def check_fhzero(nml: dict, name: str) -> None:
    """Warn when the bucket reset (fhzero) differs from the output interval (fdiag).

    Bucket fields (totprcpb_ave, cnvprcpb_ave, lhtfl_ave, ...) are means since
    the last reset, so with fhzero != fdiag each record covers 1..fhzero/fdiag
    output intervals instead of one.
    """
    fdiag = nml.get("atmos_model_nml", {}).get("fdiag")
    fhzero = nml.get("gfs_physics_nml", {}).get("fhzero")
    if fdiag != fhzero:
        log.warning(
            f"{name}: fhzero {fhzero} != fdiag {fdiag}; bucket outputs are running means"
        )


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

    if state.tgrad_perturbations:
        # SST and sea-ice climatologies are written to FIXED/MODS by
        # apply_tgrad_perturbations(), either perturbed or as clean copies.
        nml["namsfc"]["fntsfc"] = f"FIXED/MODS/{namsfc_files['fntsfc']}"
        nml["namsfc"]["fnaisc"] = f"FIXED/MODS/{namsfc_files['fnaisc']}"

        # SST, ice mask and ice concentration follow the (perturbed) FIXED
        # climatologies at every surface cycle; fsicl covers points already
        # classified as ice (non-open-ocean branch of the sfcsub merge).
        nml["namsfc"].update({"ftsfs": 0, "faiss": 0, "fsics": 0, "fsicl": 0})

        phys = nml.setdefault("gfs_physics_nml", {})
        phys["use_ext_sst"] = False

        if state.tgrad_perturbations.get("ocean_mode", "prescribed") == "prescribed":
            phys["do_ocean"] = False
            phys["nstf_name"] = [0, 0, 1, 0, 5]
        else:
            phys["do_ocean"] = True
            nml.setdefault("ocean_nml", {})["restore_method"] = 1

    return nml
