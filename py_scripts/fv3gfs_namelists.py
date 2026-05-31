import os
from pathlib import Path

import f90nml
import numpy as np
from fv3gfs_runtime import log, read_namelist
from fv3gfs_state import state
from fv3gfs_timings import apply_user_timings, get_first_guess_timings
from fv3gfs_utils import cp, cres_to_deg, env_setup


def restart_config():

    log.info(f"Generating namelist for restart {state.restart_no}")

    for f in list(state.home.glob("*.nml")):
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
    first_guess_timings = get_first_guess_timings()

    log.info("Generating namelist files")

    update_global_nml(
        res=state.res,
        fhmax=state.run_nhours,
        n_nests=state.n_nests,
        current_date=current_date,
        levels=state.levels,
        refine_ratios=state.refine_ratio,
        do_deep=state.do_deep,
        first_guess_timings=first_guess_timings,
    )
    update_nest_nml(
        res=state.res,
        fhmax=state.run_nhours,
        n_nests=state.n_nests,
        current_date=current_date,
        levels=state.levels,
        refine_ratios=state.refine_ratio,
        do_deep=state.do_deep,
        first_guess_timings=first_guess_timings,
    )

    update_table_files()
    update_fixed_files()


def disable_deep_convection(nml: dict, tile: int, name: str):
    if name.startswith("nest"):
        i = tile - 7  # index for nests
        refine_ratio = state.refine_ratio

        res = state.res * refine_ratio[i]
        if state.nest_type == "telescoping":
            res = state.res * int(np.prod(refine_ratio[: i + 1]))
    else:
        res = state.res

    do_deep = state.do_deep

    res_km = cres_to_deg(res).km
    if do_deep or res_km > 4:
        return nml

    nml["gfs_physics_nml"]["do_deep"] = False
    nml["gfs_physics_nml"]["imfdeepcnv"] = 2  # -1
    nml["gfs_physics_nml"]["shal_cnv"] = True
    nml["gfs_physics_nml"]["imfshalcnv"] = 2  # -1

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

    return nml


def update_global_nml(
    res: int,
    fhmax: int,
    n_nests: int,
    current_date: str,
    levels: int,
    refine_ratios: list,
    do_deep: bool,
    first_guess_timings: dict,
):

    nml_template_path = state.configs / "input_nml.yaml"
    parent_save_path = state.home / "input.nml"

    nml = read_namelist(nml_template_path)
    nml = common_configs(nml)

    nml["fv_core_nml"]["target_lat"] = state.target_lat
    nml["fv_core_nml"]["target_lon"] = state.target_lon
    nml["fv_core_nml"]["stretch_fac"] = state.stretch_factor
    nml["coupler_nml"]["current_date"] = current_date
    nml["coupler_nml"]["hours"] = fhmax

    # Use first-guess timings unless overridden by user
    nml["coupler_nml"]["dt_atmos"] = first_guess_timings["dt_atmos"]
    nml["coupler_nml"]["dt_ocean"] = first_guess_timings["dt_ocean"]

    # FIX: Pull explicitly from the global keys
    nml["fv_core_nml"]["n_split"] = first_guess_timings["global_n_split"]
    nml["fv_core_nml"]["k_split"] = first_guess_timings["global_k_split"]
    nml["fv_core_nml"]["npx"] = state.npx[0]
    nml["fv_core_nml"]["npy"] = state.npy[0]
    nml["fv_core_nml"]["ntiles"] = state.ntiles[0]
    nml["fv_core_nml"]["layout"] = state.layout[0]
    nml["fv_core_nml"]["io_layout"] = state.io_layout[0]
    nml["atmos_model_nml"]["blocksize"] = state.blocksize[0]

    if n_nests > 0:
        nk = "nesting"
        nml["fv_nest_nml"]["grid_pes"] = state["grid_pes"]
        nml["fv_nest_nml"]["nest_refine"] = [0] + state["refine_ratio"]
        nml["fv_nest_nml"]["num_tile_top"] = 6  # use 7 if regional suppergrid is used
        nml["fv_nest_nml"]["tile_coarse"] = [0] + state[nk]["parent_tile"]
        nml["fv_nest_nml"]["nest_ioffsets"] = state[nk]["nest_ioffsets"]
        nml["fv_nest_nml"]["nest_joffsets"] = state[nk]["nest_joffsets"]
        nml["fv_nest_nml"]["p_split"] = 1

    else:
        del nml["fv_nest_nml"]

    nml = disable_deep_convection(nml, 1, "global")

    nml = apply_user_timings(nml, "global")
    nml = update_namsfc(nml)

    # check for nml overrides if user provided external nml
    nml = namelist_overrides(state.global_input_nml, nml, "global")

    with open(parent_save_path, "w") as f:
        f90nml.write(nml, f)

    return 0


def update_nest_nml(
    res: int,
    fhmax: int,
    n_nests: int,
    current_date: str,
    levels: int,
    refine_ratios: list,
    do_deep: bool,
    first_guess_timings: dict,
):
    if n_nests == 0:
        return

    nest_nml_template_path = state.configs / "input_nestXX_nml.yaml"
    save_paths = [state.home / f"input_nest{i:02d}.nml" for i in range(2, n_nests + 2)]
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

    for i, (out_file, tile) in enumerate(zip(save_paths, tiles), start=1):
        nml = read_namelist(nest_nml_template_path)
        nml = common_configs(nml)
        nml = disable_deep_convection(nml, tile, f"nest{i + 1:02d}")

        # Use first-guess timings unless overridden by user

        nml["fv_core_nml"]["n_split"] = first_guess_timings["nest_n_splits"][i - 1]
        nml["fv_core_nml"]["k_split"] = first_guess_timings["nest_k_splits"][i - 1]

        # Assign calculated values to namelist, add +1 to skip the global tile
        nml["fv_core_nml"]["npx"] = state.npx[i]
        nml["fv_core_nml"]["npy"] = state.npy[i]
        nml["fv_core_nml"]["ntiles"] = state.ntiles[i]
        nml["fv_core_nml"]["layout"] = state.layout[i]
        nml["fv_core_nml"]["io_layout"] = state.io_layout[i]
        nml["atmos_model_nml"]["blocksize"] = state.blocksize[i]

        nml = apply_user_timings(nml, "nest", nest=i)

        nml = update_namsfc(nml)

        overide_obj = state.get(f"nest{i + 1:02d}_input_nml") or state.nestXX_input_nml
        nml = namelist_overrides(overide_obj, nml, f"nest{i + 1:02d}")

        with open(out_file, "w") as f:
            f90nml.write(nml, f)

    return 0


def namelist_overrides(overide_obj: str | dict, nml: dict, name: str):

    if not overide_obj:
        return nml

    if isinstance(overide_obj, (dict)):
        override_nml = overide_obj
        src = f"run_config.yaml : {name}_input_nml"

    elif isinstance(overide_obj, (str, Path)):
        overide_file = Path(overide_obj)
        src = str(overide_file)

        if not Path(overide_file).exists(follow_symlinks=True):
            log.info(f"Namelist file: {overide_file} does not exist !")
            return nml

        override_nml = read_namelist(overide_file)

        if not override_nml:
            log.info(f"Namelist file: {overide_file} is empty !")
            return nml

    log.info(f"Applying {name} nml overrides from: {src}")

    for section, entries in override_nml.items():
        if not entries:
            continue

        nml.setdefault(section, {}).update(entries)

    return nml


def update_fixed_files():
    dt = state.init_datetime
    year = dt.year
    fix_dirs = [state.fix_am, state.fix / "lut"]

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
        found = None
        for fix_dir in fix_dirs:
            candidate = fix_dir / name
            if candidate.exists():
                found = candidate
                break
            else:
                # Attempt fuzzy match if file not found
                matches = list(fix_dir.glob(f"*{name}"))
                if matches:
                    found = matches[0]
                    break

        if found:
            dest = state.fixed / name
            if not dest.exists():
                cp(found, dest)
            link = Path(state.input) / name
            link.unlink(missing_ok=True)
            rel_target = os.path.relpath(dest, start=state.input)
            link.symlink_to(rel_target)

        else:
            missing_files.append(name)

    if missing_files:
        raise FileNotFoundError(
            "The following files were not found: " + ", ".join(missing_files)
        )


def update_table_files():

    dt = state.init_datetime
    update_fixed_files()

    restart_no = state.get("restart_no", 0)

    diag_table_path = state.home / "diag_table"
    field_table_path = state.home / "field_table.yaml"

    user_diag = state.rundir / "diag_table"
    user_field = state.rundir / "field_table"

    template_diag = state.configs / "diag_table"
    template_field = state.configs / "field_table.yaml"

    if user_diag.exists():
        diag_file = user_diag
    else:
        diag_file = template_diag

    if user_field.exists():
        field_file = user_field
    else:
        field_file = template_field

    cp(diag_file, diag_table_path)
    cp(field_file, field_table_path)

    with open(diag_table_path) as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.strip().startswith("#")]

    dt_str = f"{dt.year} {dt.month:02d} {dt.day:02d} {dt.hour:02d} 0 0\n"
    desc_str = f"{state.description}\n"
    lines = [desc_str, dt_str] + [
        line.replace("XX", f"{restart_no:02d}") for line in lines
    ]
    with open(diag_table_path, "w") as f:
        f.writelines(lines)


def update_namsfc(nml):

    am_dir = Path(state.fix) / "am"

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

    for key, fname in namsfc_files.items():
        src = am_dir / fname
        dst = state.fixed / fname

        if not src.exists():
            raise FileNotFoundError(src)

        if not dst.exists():
            cp(src, dst)

        namsfc[key] = f"FIXED/{fname}"

    nml["namsfc"] = namsfc

    return nml
