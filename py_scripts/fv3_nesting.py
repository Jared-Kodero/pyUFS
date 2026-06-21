# nesting.py

from pathlib import Path

import numpy as np
import xarray as xr
from fv3_runtime import log
from fv3_state import FV3State, save_state, state
from fv3_utils import cres_to_deg, run_cmd

nest_info = []


def get_centers(params: FV3State) -> FV3State:
    params.target_lon = (params.lon_min[0] + params.lon_max[0]) * 0.5
    params.target_lat = (params.lat_min[0] + params.lat_max[0]) * 0.5
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


def gen_global_nest_parent(res: int, grid_dir: Path = None) -> Path:
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
    grid_fname: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    i_refine_ratio: int,
    alignment: int = 16,
):
    """
    Compute supergrid index bounds for an FV3 two-way nest.

    The returned indices satisfy three conditions:
      1. Parity. Start indices are odd, end indices are even, so the
         supergrid-to-cell conversion n = (end - start + 1) // 2 is exact.
      2. Alignment. The nest cell count along each axis times
         i_refine_ratio is a multiple of `alignment`, as required by
         the FV3 layout decomposition and physics block loop.
      3. Containment. Indices lie inside [1, nxp] and [1, nyp].

    Parameters
    ----------
    grid_fname : str
        Path to the parent supergrid file with variables x and y of
        shape (nyp, nxp).
    lon_min, lon_max : float
        Longitude bounds in degrees east; reduced modulo 360.
    lat_min, lat_max : float
        Latitude bounds in degrees north.
    i_refine_ratio : int
        Nest refinement ratio.
    alignment : int
        Required divisor of the nest cell count along each axis.
    """
    with xr.open_dataset(grid_fname) as ds:
        lons = ds.x.values
        lats = ds.y.values
    nyp, nxp = lons.shape

    lon_min %= 360
    lon_max %= 360
    mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
    j_idx, i_idx = np.where(mask)

    # Initial bracket with one-cell padding, packed as [i, j] vectors.
    starts = np.array([i_idx.min() - 1, j_idx.min() - 1])
    ends = np.array([i_idx.max() + 1, j_idx.max() + 1])
    limits = np.array([nxp, nyp])

    # Parity: odd starts, even ends.
    starts = np.where(starts & 1, starts, starts - 1)
    ends = np.where(ends & 1, ends - 1, ends)

    # Symmetric expansion so parent_cells is a multiple of
    # needed = alignment / gcd(alignment, i_refine_ratio).
    needed = alignment // int(np.gcd(alignment, i_refine_ratio))
    parent_cells = (ends - starts + 1) // 2
    deficit = (needed - parent_cells % needed) % needed
    left = deficit // 2
    right = deficit - left
    starts -= 2 * left
    ends += 2 * right

    # Containment. Any required shift is rounded up to even to keep parity.
    under = np.maximum(0, 1 - starts)
    under += under & 1
    starts += under
    ends += under

    over = np.maximum(0, ends - limits)
    over += over & 1
    starts -= over
    ends -= over

    return dict(
        istart_nest=int(starts[0]),
        iend_nest=int(ends[0]),
        jstart_nest=int(starts[1]),
        jend_nest=int(ends[1]),
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

    keys = (
        "parent_tile",
        "istart_nest",
        "iend_nest",
        "jstart_nest",
        "jend_nest",
        "nest_ioffsets",
        "nest_joffsets",
    )

    state.nesting = {}
    for k in keys:
        state.nesting.setdefault(k, [])

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

    state.nesting["parent_tile"].append(parent_tile[i])
    state.nesting["istart_nest"].append(indices["istart_nest"])
    state.nesting["iend_nest"].append(indices["iend_nest"])
    state.nesting["jstart_nest"].append(indices["jstart_nest"])
    state.nesting["jend_nest"].append(indices["jend_nest"])

    # Convert supergrid (grid file) indices to FV3 parent cell indices
    nest_ioffsets = [999] + [(i // 2) + 1 for i in state.nesting["istart_nest"]]
    nest_joffsets = [999] + [(j // 2) + 1 for j in state.nesting["jstart_nest"]]
    state.nesting["nest_ioffsets"] = nest_ioffsets
    state.nesting["nest_joffsets"] = nest_joffsets

    save_state()


def get_nest_tele_indices(
    res: int, n_nests: int, refine_ratio: list, grid_dir: Path
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
        state.nesting[k] = []

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

        state.nesting["parent_tile"].append(parent_tile)
        state.nesting["istart_nest"].append(indices["istart_nest"])
        state.nesting["iend_nest"].append(indices["iend_nest"])
        state.nesting["jstart_nest"].append(indices["jstart_nest"])
        state.nesting["jend_nest"].append(indices["jend_nest"])

    nest_ioffsets = [999] + [(i // 2) + 1 for i in state.nesting["istart_nest"]]
    nest_joffsets = [999] + [(j // 2) + 1 for j in state.nesting["jstart_nest"]]
    state.nesting["nest_ioffsets"] = nest_ioffsets
    state.nesting["nest_joffsets"] = nest_joffsets
    save_state()
