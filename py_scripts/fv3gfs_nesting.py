from pathlib import Path

import numpy as np
import xarray as xr
from fv3gfs_runtime import log
from fv3gfs_state import save_state, state
from fv3gfs_utils import cres_to_deg, run_cmd

nest_info = []


def get_centers(params):
    params.target_lon = (params.lon_min[0] + params.lon_max[0]) * 0.5
    params.target_lat = (params.lat_min[0] + params.lat_max[0]) * 0.5
    return params


def validate_nests(params) -> dict:
    x_min = params.lon_min
    x_max = params.lon_max
    y_min = params.lat_min
    y_max = params.lat_max
    n_nests = params.n_nests
    refine_ratios = params.refine_ratio

    if not all(isinstance(v, list) for v in [x_min, x_max, y_min, y_max]):
        raise TypeError(
            "Bounding box parameters must be provided as lists, when gtype='nest'."
        )

    if any(v is None for v in [x_min, x_max, y_min, y_max]):
        raise ValueError("Missing bounding box parameters for gtype='nest'.")

    if n_nests > 0:
        valid_bboxes = len(x_min) == len(x_max) == len(y_min) == len(y_max)
        if not valid_bboxes:
            raise ValueError("Mismatch between number of bounding box parameters.")

    params = get_centers(params)
    params = classify_nesting(params)

    if params.nest_type == "same_level":
        for i, r in enumerate(refine_ratios):
            resolution = cres_to_deg(params.res * r).km
            nest_info.append(f"Nested tile {7 + i} resolution: {resolution:.2f} km")
    elif params.nest_type == "telescoping":
        total_refine = 1
        nest_res_km = []
        for i, r in enumerate(refine_ratios):
            total_refine *= r
            n_res = cres_to_deg(params.res * total_refine).km

            nest_info.append(f"Nested tile {7 + i} resolution: {n_res:.2f} km")
            nest_res_km.append(n_res)

        params.nest_res_km = nest_res_km
    nest_info.append(f"Nest layout type: {params.nest_type}")
    return nest_info


def classify_nesting(params: dict) -> dict:
    lon_min = params.lon_min
    lon_max = params.lon_max
    lat_min = params.lat_min
    lat_max = params.lat_max
    n = len(lon_min)

    # 1. Basic integrity checks
    if not (len(lon_max) == len(lat_min) == len(lat_max) == n):
        raise ValueError("All coordinate lists must have the same length.")

    if n < 2:
        params["nest_type"] = "same_level"
        return params

    for i in range(n):
        # 2. Check if the individual boxes are physically valid
        if lon_min[i] >= lon_max[i] or lat_min[i] >= lat_max[i]:
            raise ValueError(
                f"Domain {i} has invalid bounds: min must be less than max."
            )

    for i in range(n - 1):
        parent_contains_child = (
            lon_min[i] <= lon_min[i + 1]
            and lon_max[i] >= lon_max[i + 1]
            and lat_min[i] <= lat_min[i + 1]
            and lat_max[i] >= lat_max[i + 1]
        )

        child_contains_parent = (
            lon_min[i] >= lon_min[i + 1]
            and lon_max[i] <= lon_max[i + 1]
            and lat_min[i] >= lat_min[i + 1]
            and lat_max[i] <= lat_max[i + 1]
        )

        is_nested = parent_contains_child or child_contains_parent

        if not is_nested:
            params["nest_type"] = "same_level"
            break

        if child_contains_parent:
            raise ValueError(
                f"Domains {i} and {i + 1} are nested but ordered incorrectly!"
            )

        # if we reach here, parent contains child
        params["nest_type"] = "telescoping"

    return params


def gen_global_nest_parent(res, grid_dir=None) -> Path:
    log_file = state.logs / "make_global_grid.log"
    make_hgrid = state.ufs_exe / "make_hgrid"

    nlon = res * 2

    cmd = [
        f"{make_hgrid}",
        "--grid_type",
        "gnomonic_ed",
        "--nlon",
        f"{nlon}",
        "--grid_name",
        f"C{res}_grid",
        "--do_schmidt",
        "--stretch_factor",
        f"{state.stretch_factor}",
        "--target_lon",
        f"{state.target_lon}",
        "--target_lat",
        f"{state.target_lat}",
        "--great_circle_algorithm",
    ]

    if grid_dir is None:
        grid_dir = state.tmp / ".tmp_make_grid"
        grid_dir.mkdir(parents=True, exist_ok=True)

    result, msgs = run_cmd(cmd, cwd=grid_dir, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate global uniform grid")
    return grid_dir


def calc_parent_grid_index(
    grid_fname, lon_min, lon_max, lat_min, lat_max, i_refine_ratio
):
    ds = xr.open_dataset(grid_fname)
    # 1. Normalize longitudes to [0, 360]
    lon_min = lon_min % 360
    lon_max = lon_max % 360

    # 2. On the chosen parent tile, find the indices for each corner of the bounding box
    lons, lats = ds.x.values, ds.y.values
    nxp, nyp = lons.shape[0], lons.shape[1]
    mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)

    j_idx, i_idx = np.where(mask)

    i_s = i_idx.min() - 1
    i_e = i_idx.max() + 1
    j_s = j_idx.min() - 1
    j_e = j_idx.max() + 1

    # Make sure start indices are odd for FV3 nest start/end
    istart_nest = i_s if i_s % 2 == 1 else i_s - 1
    jstart_nest = j_s if j_s % 2 == 1 else j_s - 1

    # Ensure even indices for FV3 nest start/end
    iend_nest = i_e if i_e % 2 == 0 else i_e - 1
    jend_nest = j_e if j_e % 2 == 0 else j_e - 1

    ds.close()

    return dict(
        istart_nest=int(istart_nest),
        iend_nest=int(iend_nest),
        jstart_nest=int(jstart_nest),
        jend_nest=int(jend_nest),
    )


def get_nest_indices(
    res: int,
    tile_idx: int,
    grid_dir: Path = None,
    parent_tile: list = None,
    i_refine_ratio: int = None,
) -> None:
    """
    normal: normal static nests each embedded directly in the same parent (global) grid.
    """

    nk = "nesting"
    keys = (
        "parent_tile",
        "istart_nest",
        "iend_nest",
        "jstart_nest",
        "jend_nest",
        "nest_ioffsets",
        "nest_joffsets",
    )

    state[nk] = {}
    for k in keys:
        state[nk].setdefault(k, [])

    if not grid_dir:
        grid_dir = gen_global_nest_parent(res)

    i = tile_idx  # Nest index (0-based)

    grid_fname = grid_dir / f"C{res}_grid.tile{parent_tile[i]}.nc"
    indices = calc_parent_grid_index(
        grid_fname,
        state.lon_min[i],
        state.lon_max[i],
        state.lat_min[i],
        state.lat_max[i],
        i_refine_ratio,
    )

    state[nk]["parent_tile"].append(parent_tile[i])
    state[nk]["istart_nest"].append(indices["istart_nest"])
    state[nk]["iend_nest"].append(indices["iend_nest"])
    state[nk]["jstart_nest"].append(indices["jstart_nest"])
    state[nk]["jend_nest"].append(indices["jend_nest"])

    # Convert supergrid (grid file) indices to FV3 parent cell indices
    nest_ioffsets = [999] + [(i // 2) + 1 for i in state[nk]["istart_nest"]]
    nest_joffsets = [999] + [(j // 2) + 1 for j in state[nk]["jstart_nest"]]
    state[nk]["nest_ioffsets"] = nest_ioffsets
    state[nk]["nest_joffsets"] = nest_joffsets

    save_state()


def get_nest_tele_indices(res, n_nests, refine_ratio, grid_dir) -> None:

    # Reset previous same_level indices if they exist
    nk = "nesting"
    keys = (
        "parent_tile",
        "istart_nest",
        "iend_nest",
        "jstart_nest",
        "jend_nest",
        "nest_ioffsets",
        "nest_joffsets",
    )
    for k in keys:
        state[nk][k] = []

    tiles = [i + 7 for i in range(n_nests)]

    for i, tile in enumerate(tiles):
        parent_tile = tile - 1
        grid_parent_fname = grid_dir / f"C{res}_grid.tile{parent_tile}.nc"

        i_refine_ratio = np.prod(refine_ratio[: i + 1])

        indices = calc_parent_grid_index(
            grid_parent_fname,
            state.lon_min[i],
            state.lon_max[i],
            state.lat_min[i],
            state.lat_max[i],
            i_refine_ratio,
        )

        state[nk]["parent_tile"].append(parent_tile)
        state[nk]["istart_nest"].append(indices["istart_nest"])
        state[nk]["iend_nest"].append(indices["iend_nest"])
        state[nk]["jstart_nest"].append(indices["jstart_nest"])
        state[nk]["jend_nest"].append(indices["jend_nest"])

    nest_ioffsets = [999] + [(i // 2) + 1 for i in state[nk]["istart_nest"]]
    nest_joffsets = [999] + [(j // 2) + 1 for j in state[nk]["jstart_nest"]]
    state[nk]["nest_ioffsets"] = nest_ioffsets
    state[nk]["nest_joffsets"] = nest_joffsets
    save_state()
