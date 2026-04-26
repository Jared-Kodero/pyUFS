import re
from pathlib import Path

import f90nml
import numpy as np
import yaml
from fv3gfs_runtime import log, nml_to_dict
from fv3gfs_state import state
from fv3gfs_timings import apply_user_timings, get_first_guess_timings
from fv3gfs_utils import cp, cres_to_deg, env_setup

time_int_log = []


def update_nml_configs():
    env_setup()

    dt = state.init_datetime

    current_date = [dt.year, dt.month, dt.day, dt.hour, 0, 0]
    state.model_start_date = current_date

    # Do nest namelists
    first_guess_timings = get_first_guess_timings()

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

    if state.debug:
        for i in time_int_log:
            log.info(i)


# --- Enable/disable deep convection for CP runs ---
def do_deep_false(nml, tile):
    i = tile - 7  # index for nests
    refine_ratio = state.refine_ratio
    do_deep = state.do_deep

    res = state.res * refine_ratio[i]
    if state.nest_type == "telescoping":
        res = state.res * int(np.prod(refine_ratio[: i + 1]))

    res_km = cres_to_deg(res).km
    if do_deep or res_km > 4:
        return nml

    log.info(f"Nested tile {tile}: deep convection disabled ({res_km:.2f} km)")

    # disable deep convection
    nml["gfs_physics_nml"]["do_deep"] = False
    nml["gfs_physics_nml"]["imfdeepcnv"] = -1

    # disable shallow convection as well
    nml["gfs_physics_nml"]["shal_cnv"] = False
    nml["gfs_physics_nml"]["imfshalcnv"] = -1

    return nml


def common_configs(nml):
    nml["fms_nml"]["domains_stack_size"] = 2**30  # 1 GiB
    nml["fv_core_nml"]["npz"] = state.levels - 1
    nml["external_ic_nml"]["levp"] = state.levels
    nml["gfs_physics_nml"]["lsm"] = 2  # Use the new noah land surface model
    nml["fv_core_nml"]["hord_tr"] = -5

    # enable coupled sfc for all runs
    nml["gfs_physics_nml"]["sfc_coupled"] = True

    return nml


def restart_config():

    log.info(f"Generating namelist for restart {state.restart_no}")

    for f in list(state.home.glob("*.nml")):
        with open(f, "r") as nml_in:
            nml = nml_to_dict(f90nml.read(nml_in))

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

    log.info("Generating global namelist")

    main_nml_path = state.configs / "input.nml"
    parent_save_path = state.home / "input.nml"

    nml = nml_to_dict(f90nml.read(main_nml_path))
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

    # check for nml overrides if user provided external nml
    nml = namelist_overrides(state.nml, nml, "global")
    nml = apply_user_timings(nml, "global")
    nml = update_namsfc(nml, res)

    # fmt:off
    time_int_log.append(f"FV3 time step: dt_atmos = {nml['coupler_nml']['dt_atmos']}")
    time_int_log.append(f"FV3 time step: dt_ocean = {nml['coupler_nml']['dt_ocean']}")
    time_int_log.append(f"FV3 splitting: global k_split = {nml['fv_core_nml']['k_split']}")
    time_int_log.append(f"FV3 splitting: global n_split = {nml['fv_core_nml']['n_split']}")

    # fmt:on

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

    log.info("Generating nest tiles namelists")

    nest_nml_path = state.configs / "input_nestXX.nml"
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
        nml = nml_to_dict(f90nml.read(nest_nml_path))
        nml = common_configs(nml)
        nml = do_deep_false(nml, tile)

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

        overide_file = state.get(f"tile{tile}_nml") or state.tileX_nml
        nml = namelist_overrides(overide_file, nml, f"nest{i + 1:02d}")
        nml = apply_user_timings(nml, "nest", nest=i)

        nml = update_namsfc(nml, res)

        time_int_log.append(
            f"FV3 splitting (tile {tile}): k_split = {nml['fv_core_nml']['k_split']}"
        )
        time_int_log.append(
            f"FV3 splitting (tile {tile}): n_split = {nml['fv_core_nml']['n_split']}"
        )

        with open(out_file, "w") as f:
            f90nml.write(nml, f)

    return 0


def namelist_overrides(overide_file, nml, name):

    if not overide_file:
        return nml

    if not Path(overide_file).exists(follow_symlinks=True):
        log.info(f"Override file: {overide_file} does not exist !")
        return nml

    if str(overide_file).endswith(".nml"):
        override_nml = nml_to_dict(f90nml.read(overide_file))
    elif str(overide_file).endswith((".yaml", ".yaml")):
        with open(overide_file, "r") as f:
            override_nml = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported override file format. Use .nml or .yaml/.yaml")

    if not override_nml:
        log.info(f"Override file: {overide_file} is empty !")
        return nml

    log.info(f"Applying {name} tile(s) nml overrides from: {overide_file}")
    for section, entries in override_nml.items():
        if not entries:
            continue

        if section not in nml:
            nml[section] = {}

        for key, value in entries.items():
            old_value = nml[section].get(key, "")

            if old_value == value:
                continue

            else:
                if section == "fv_nest_nml" and key == "grid_pes":
                    continue  # skip grid_pes overrides

                nml[section][key] = value
                log.debug(f"{name}[{section}][{key}]: {old_value} -> {value}")

    return nml


def update_namsfc(nml, res):
    for k, v in nml["namsfc"].items():
        if isinstance(v, str):
            nml["namsfc"][k] = str(v).replace("CXXX", f"C{res}")

    am_dir = Path(state.fix) / "am"

    constant_files = {
        "FNGLAC": "global_glacier",
        "FNMXIC": "global_maxice",
        "FNTSFC": "RTGSST",
        "FNSNOC": "global_snoclim",
        "FNAISC": "CFSR.SEAICE",
        "FNMLDC": "mld_DR003_c1m_reg",
    }

    variable_files = {
        "FNMSKH": "global_slmask",
        "FNSMCC": "global_soilmgldas",
    }

    # === Helper: extract resolution info from filename ===
    def parse_grid_info(fname):
        """
        Extract (Tres, NX, NY) from a file name like global_slmask.t574.1152.576.grb
        """
        m = re.search(r"t(\d+)\.(\d+)\.(\d+)", fname)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        return None, None, None

    for k, v in constant_files.items():
        files = list(am_dir.glob(f"{v}*"))
        files = [f for f in files if f.name.endswith((".grb", ".grb2"))]
        if not files or len(files) > 1:
            print(f"Files found for {k}: {[f.name for f in files]}")
            raise FileNotFoundError(f"Missing or ambiguous constant file for {k}")

        cp(files[0], state.fixed / files[0].name)
        nml["namsfc"][k] = f"FIXED/{files[0].name}"

    for k, v in variable_files.items():
        files = list(am_dir.glob(f"{v}*"))
        if not files:
            raise FileNotFoundError(f"Missing variable file for {k}")

        best_match = None

        matches = {}

        for f in files:
            if f.name.endswith(".grb") and ".rg." not in f.name:
                tres, nx, ny = parse_grid_info(f.name)
                matches[tres] = f

        # sort res dict by key (tres)
        sorted_res = dict(sorted(matches.items()))

        # take the last key less
        best_match = sorted_res[list(sorted_res.keys())[-1]]

        if best_match:
            cp(best_match, state.fixed / best_match.name)

            nml["namsfc"][k] = f"FIXED/{best_match.name}"
        else:
            raise FileNotFoundError(f"Could not find suitable match for {k}")

    return nml
