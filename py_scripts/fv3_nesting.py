# nesting.py

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from fv3_runtime import log
from fv3_state import FV3State, save_fv3_state, state
from fv3_utils import cres_to_deg, exit_code, run_cmd

nest_info = []


def get_centers(params: FV3State) -> FV3State:
    params.target_lon = round((params.lon_min[0] + params.lon_max[0]) * 0.5, 2)
    params.target_lat = round((params.lat_min[0] + params.lat_max[0]) * 0.5, 2)
    return params


def validate_nests(params: FV3State) -> list:
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
    nest_res_km = []

    if params.nest_type == "same_level":
        for i, r in enumerate(refine_ratios):
            res_km = cres_to_deg(params.c_res * r).km
            nest_res_km.append(res_km)
            nest_info.append(f"Nested tile {7 + i} resolution: {res_km:.2f} km")
    elif params.nest_type == "telescoping":
        total_refine = 1

        for i, r in enumerate(refine_ratios):
            total_refine *= r
            res_km = cres_to_deg(params.c_res * total_refine).km
            nest_res_km.append(res_km)
            nest_info.append(f"Nested tile {7 + i} resolution: {res_km:.2f} km")

    # res_km is preallocated in preprocess_input as [global, 0, 0, ...] with
    # one slot per nest. Assign into those slots; extend() would append past
    # them and leave the nest entries at zero.
    params.res_km[1:] = nest_res_km
    nest_info.append(f"Nest layout type: {params.nest_type}")
    return nest_info


def classify_nesting(params: FV3State) -> FV3State:
    lon_min = params.lon_min
    lon_max = params.lon_max
    lat_min = params.lat_min
    lat_max = params.lat_max
    n = len(lon_min)

    # 1. Basic integrity checks
    if not (len(lon_max) == len(lat_min) == len(lat_max) == n):
        raise ValueError("All coordinate lists must have the same length.")

    if n < 2:
        params.nest_type = "same_level"
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
            params.nest_type = "same_level"
            break

        if child_contains_parent:
            raise ValueError(
                f"Domains {i} and {i + 1} are nested but ordered incorrectly!"
            )

        # if we reach here, parent contains child
        params.nest_type = "telescoping"

    return params


def gen_global_nest_parent(c_res: int, grid_dir: Path = None) -> Path:
    log_file = state.logs / "make_global_grid.log"
    make_hgrid = state.ufs_exe / "make_hgrid"

    nlon = c_res * 2

    cmd = [
        f"{make_hgrid}",
        "--grid_type",
        "gnomonic_ed",
        "--nlon",
        f"{nlon}",
        "--grid_name",
        f"C{c_res}_grid",
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

    result, msgs = run_cmd(cmd, cwd=grid_dir, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to generate global uniform grid")
    return grid_dir


def calc_parent_grid_index(
    idx: int,
    parent_tile: int,
    grid_fname: Path,
):
    """"""

    lon_min = state.lon_min[idx]
    lon_max = state.lon_max[idx]
    lat_min = state.lat_min[idx]
    lat_max = state.lat_max[idx]

    with xr.open_dataset(grid_fname) as ds:
        lons = ds.x.values
        lats = ds.y.values
    nyp, nxp = lons.shape

    lon_min %= 360
    lon_max %= 360

    mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
    j_idx, i_idx = np.where(mask)

    # Initial bracket with one-cell padding, packed as [i, j] vectors.
    starts = np.array([i_idx.min(), j_idx.min()])
    ends = np.array([i_idx.max(), j_idx.max()])
    limits = np.array([nxp, nyp])

    # Parity: odd starts, even ends.
    starts = np.where(starts & 1, starts, starts - 1)
    ends = np.where(ends & 1, ends - 1, ends)

    if np.any(starts < 1) or np.any(ends > limits):
        nest_tile = parent_tile + 1
        log.error(
            f"Tile {nest_tile} bounding box is larger than or crosses the parent tile {parent_tile} bounds, This is not supported!"
        )
        exit_code(1)
        sys.exit(1)

    return dict(
        istart_nest=int(starts[0]),
        iend_nest=int(ends[0]),
        jstart_nest=int(starts[1]),
        jend_nest=int(ends[1]),
    )


def get_nest_indices(
    c_res: int,
    tile_idx: int,
    grid_dir: Path = None,
    parent_tile: int = None,
    i_refine_ratio: int = None,
    tile: int = None,
) -> None:
    """
    normal: normal static nests each embedded directly in the same parent (global) grid.
    """

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
        state[k] = []

    if not grid_dir:
        grid_dir = gen_global_nest_parent(c_res)

    i = tile_idx  # Nest index (0-based)

    grid_fname = grid_dir / f"C{c_res}_grid.tile{parent_tile}.nc"
    indices = calc_parent_grid_index(i, parent_tile, grid_fname)

    state.parent_tile.append(parent_tile)
    state.istart_nest.append(indices["istart_nest"])
    state.iend_nest.append(indices["iend_nest"])
    state.jstart_nest.append(indices["jstart_nest"])
    state.jend_nest.append(indices["jend_nest"])

    # Convert supergrid (grid file) indices to FV3 parent cell indices
    nest_ioffsets = [999] + [(i // 2) + 1 for i in state.istart_nest]
    nest_joffsets = [999] + [(j // 2) + 1 for j in state.jstart_nest]
    state.nest_ioffsets = nest_ioffsets
    state.nest_joffsets = nest_joffsets

    save_fv3_state()


def get_nest_tele_indices(
    c_res: int, n_nests: int, refine_ratio: list, grid_dir: Path
) -> None:

    # Reset previous same_level indices if they exist
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
        state[k] = []

    tiles = [i + 7 for i in range(n_nests)]

    for i, tile in enumerate(tiles):
        parent_tile = tile - 1
        grid_parent_fname = grid_dir / f"C{c_res}_grid.tile{parent_tile}.nc"
        i_refine_ratio = np.prod(refine_ratio[: i + 1])  # not used now
        indices = calc_parent_grid_index(i, parent_tile, grid_parent_fname)

        state.parent_tile.append(parent_tile)
        state.istart_nest.append(indices["istart_nest"])
        state.iend_nest.append(indices["iend_nest"])
        state.jstart_nest.append(indices["jstart_nest"])
        state.jend_nest.append(indices["jend_nest"])

    nest_ioffsets = [999] + [(i // 2) + 1 for i in state.istart_nest]
    nest_joffsets = [999] + [(j // 2) + 1 for j in state.jstart_nest]
    state.nest_ioffsets = nest_ioffsets
    state.nest_joffsets = nest_joffsets
    save_fv3_state()
