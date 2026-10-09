# nesting.py

from pathlib import Path

import numpy as np
import xarray as xr
from fv3_runtime import log
from fv3_state import FV3State, save_fv3_state, state
from fv3_utils import cres_to_deg, run_cmd

nest_info = []


def lon_width(lon_min: float, lon_max: float) -> float:
    """Eastward width of a longitude range [deg]; a box may cross 0 or 180
    degrees (lon_max < lon_min), so 170 to -170 is 20 degrees wide."""
    return (lon_max - lon_min) % 360.0


def get_centers(params: FV3State) -> FV3State:
    """Centre the cube (target_lon, target_lat) on the first nest box."""
    width = lon_width(params.lon_min[0], params.lon_max[0])
    centre = (params.lon_min[0] + 0.5 * width + 180.0) % 360.0 - 180.0
    params.target_lon = round(centre, 2)
    params.target_lat = round((params.lat_min[0] + params.lat_max[0]) * 0.5, 2)
    return params


def validate_nests(params: FV3State) -> list:
    """Check the nest boxes against refine_ratio, classify them, centre the cube.

    One box per refine_ratio entry. Scalars are accepted for a single nest.
    """
    n_nests = params.n_nests
    refine_ratios = params.refine_ratio

    boxes = {}
    for key in ("lon_min", "lon_max", "lat_min", "lat_max"):
        value = params[key]
        if value is None or (isinstance(value, list) and None in value):
            raise ValueError(f"gtype nest requires {key} (one value per nest)")
        if not isinstance(value, list):
            value = [value]
        if len(value) != n_nests:
            raise ValueError(
                f"{key} has {len(value)} value(s) but refine_ratio defines "
                + f"{n_nests} nest(s); give one box per nest"
            )
        boxes[key] = [float(v) for v in value]
        params[key] = boxes[key]

    for i in range(n_nests):
        if lon_width(boxes["lon_min"][i], boxes["lon_max"][i]) == 0.0:
            raise ValueError(f"Nest {i + 2:02d}: lon_min equals lon_max")
        if boxes["lat_min"][i] >= boxes["lat_max"][i]:
            raise ValueError(f"Nest {i + 2:02d}: lat_min must be less than lat_max")

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


def _contains(outer: int, inner: int, b: dict) -> bool:
    """Box `inner` lies within box `outer` (longitudes on the circle)."""
    w_out = lon_width(b["lon_min"][outer], b["lon_max"][outer])
    w_in = lon_width(b["lon_min"][inner], b["lon_max"][inner])
    offset = (b["lon_min"][inner] - b["lon_min"][outer]) % 360.0
    return (
        offset + w_in <= w_out
        and b["lat_min"][outer] <= b["lat_min"][inner]
        and b["lat_max"][inner] <= b["lat_max"][outer]
    )


def classify_nesting(params: FV3State) -> FV3State:
    """telescoping when each box contains the next, otherwise same_level."""
    b = {k: params[k] for k in ("lon_min", "lon_max", "lat_min", "lat_max")}
    n = len(b["lon_min"])
    if not all(len(v) == n for v in b.values()):
        raise ValueError("All coordinate lists must have the same length.")

    params.nest_type = "same_level"
    if n < 2:
        return params

    for i in range(n - 1):
        if _contains(i + 1, i, b):
            raise ValueError(
                f"Domains {i} and {i + 1} are nested but ordered incorrectly!"
            )
    if all(_contains(i, i + 1, b) for i in range(n - 1)):
        params.nest_type = "telescoping"
    return params


def gen_global_nest_parent(c_res: int, grid_dir: Path | None = None) -> Path:
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

    lon_min = state.lon_min[idx]
    lon_max = state.lon_max[idx]
    lat_min = state.lat_min[idx]
    lat_max = state.lat_max[idx]

    with xr.open_dataset(grid_fname) as ds:
        lons = ds.x.values
        lats = ds.y.values
    nyp, nxp = lons.shape

    # Longitude test on the circle, so a box may cross 0 or 180 degrees: a
    # point is inside when its eastward offset from lon_min does not exceed
    # the eastward width of the box.
    width = (lon_max - lon_min) % 360.0
    in_lon = (lons - lon_min) % 360.0 <= width
    mask = in_lon & (lats >= lat_min) & (lats <= lat_max)
    j_idx, i_idx = np.where(mask)

    if i_idx.size == 0:
        raise ValueError(
            f"Nest {idx + 2:02d} bounding box [{lon_min}, {lon_max}] x "
            + f"[{lat_min}, {lat_max}] contains no point of parent tile {parent_tile}; "
            + "check parent_tile and the box"
        )

    # Smallest block of parent cells that covers the box. make_hgrid takes
    # 1-based supergrid indices (istart odd, iend even); parent cells
    # (istart+1)/2 .. iend/2 span 0-based supergrid points istart-1 .. iend,
    # and supergrid segment [k, k+1] lies in cell k//2 + 1. The box edges lie
    # on the segments just outside the first and last points inside it.
    n_cells = np.array([(nxp - 1) // 2, (nyp - 1) // 2])
    lo = np.array([i_idx.min(), j_idx.min()])
    hi = np.array([i_idx.max(), j_idx.max()])
    first = np.maximum(lo - 1, 0) // 2 + 1
    last = np.minimum(hi // 2 + 1, n_cells)

    # make_hgrid needs `halo` parent cells around the nest inside the tile
    # (create_gnomonic_cubic_grid.c).
    halo = int(state.halo or 0)
    if np.any(first - halo < 1) or np.any(last + halo > n_cells):
        raise ValueError(
            f"Nest {idx + 2:02d} box reaches within {halo} cells of the edge of parent "
            + f"tile {parent_tile} (cells {first.tolist()}..{last.tolist()} of "
            + f"{n_cells.tolist()}); move or shrink the box, or choose another parent_tile"
        )

    return {
        "istart_nest": int(2 * first[0] - 1),
        "iend_nest": int(2 * last[0]),
        "jstart_nest": int(2 * first[1] - 1),
        "jend_nest": int(2 * last[1]),
    }


NEST_INDEX_KEYS = (
    "parent_tile",
    "istart_nest",
    "iend_nest",
    "jstart_nest",
    "jend_nest",
    "nest_ioffsets",
    "nest_joffsets",
)


def _reset_nest_indices() -> None:
    for k in NEST_INDEX_KEYS:
        state[k] = []


def _append_nest_indices(parent_tile: int, indices: dict) -> None:
    """Record one nest and refresh the FV3 offsets of all recorded nests."""
    state.parent_tile.append(parent_tile)
    state.istart_nest.append(indices["istart_nest"])
    state.iend_nest.append(indices["iend_nest"])
    state.jstart_nest.append(indices["jstart_nest"])
    state.jend_nest.append(indices["jend_nest"])

    # Convert supergrid (grid file) indices to FV3 parent cell indices
    state.nest_ioffsets = [999] + [(i // 2) + 1 for i in state.istart_nest]
    state.nest_joffsets = [999] + [(j // 2) + 1 for j in state.jstart_nest]


def get_nest_indices(
    c_res: int,
    tile_idx: int,
    grid_dir: Path | None = None,
    parent_tile: int | None = None,
    i_refine_ratio: int | None = None,
    tile: int | None = None,
) -> dict:
    """Bracket nest `tile_idx` (0-based) on its parent tile and record it.

    Nests are recorded in order: tile_idx 0 starts a new list and each later
    call appends, so a same-level run keeps the indices of every nest for
    fv_nest_nml. Returns the indices of this nest.
    """
    if tile_idx == 0:
        _reset_nest_indices()
    if len(state.istart_nest) != tile_idx:
        raise RuntimeError(
            f"Nest {tile_idx} bracketed out of order ({len(state.istart_nest)} recorded)"
        )

    if not grid_dir:
        grid_dir = gen_global_nest_parent(c_res)

    grid_fname = grid_dir / f"C{c_res}_grid.tile{parent_tile}.nc"
    indices = calc_parent_grid_index(tile_idx, parent_tile, grid_fname)
    _append_nest_indices(parent_tile, indices)

    save_fv3_state()
    return indices
