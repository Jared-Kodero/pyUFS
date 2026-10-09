# pes_config.py

from math import isqrt
from pathlib import Path

import numpy as np
import xarray as xr
from fv3_runtime import read_namelist, sort_paths
from fv3_state import save_fv3_state, state
from fv3_timings import get_timings

grid_dir: Path | None = None


def calc_cpu_alloc(dir: Path) -> None:
    global grid_dir
    grid_dir = dir
    get_grid_info()
    if state.gtype == "nest":
        calc_nest_pes()
    elif state.gtype in ("regional_gfdl", "regional_esg"):
        calc_regional_pes()
    else:
        calc_uniform_pes()


def get_grid_info() -> None:
    state.ngrid_cells = [0 for _ in range(state.n_nests + 1)]
    state.ntiles = []
    state.npx = []
    state.npy = []

    if state.gtype in ("regional_gfdl", "regional_esg"):
        get_regional_grid_info()
        return

    files = sorted(grid_dir.glob("C*_grid.tile*.nc"), key=sort_paths)

    for f in files:
        tile_num = int(f.stem.split(".")[-1].replace("tile", ""))

        if tile_num < 6:
            continue

        with xr.open_dataset(f) as ds:
            nx = ds.nx.size
            ny = ds.ny.size
            cells = nx * ny

            npx = int((nx // 2) + 1)
            npy = int((ny // 2) + 1)

        if tile_num == 6:
            n = 6
            state.ngrid_cells[0] = cells * n
        else:
            n = 1
            idx = tile_num - 6
            state.ngrid_cells[idx] = cells * n

        state.ntiles.append(n)
        state.npx.append(npx)
        state.npy.append(npy)


def get_regional_grid_info() -> None:
    """Read the compute-domain (halo0) tile-7 grid of a regional domain.

    npx and npy count the compute-domain points, without the boundary halo
    (the halo0 grid written by fv3_shave).
    """
    files = sorted(grid_dir.glob("C*_grid.tile7.halo0.nc"), key=sort_paths)

    if not files:
        raise FileNotFoundError(
            f"No regional grid file matching C*_grid.tile7.halo0.nc found in {grid_dir}."
        )
    if len(files) > 1:
        names = ", ".join(path.name for path in files)
        raise ValueError(f"Multiple regional tile-7 grid files found: {names}")

    with xr.open_dataset(files[0]) as ds:
        nx = ds.nx.size
        ny = ds.ny.size

    state.ngrid_cells = [nx * ny]
    state.ntiles = [1]
    state.npx = [int((nx // 2) + 1)]
    state.npy = [int((ny // 2) + 1)]


def calc_regional_pes() -> None:
    """Allocate all available PEs to a standalone single-tile regional grid."""
    if state.n_cpus <= 0:
        raise ValueError(f"Invalid CPU count for regional grid: {state.n_cpus}")

    pes, layouts = _largest_decomposable(state.n_cpus, ntiles=1)
    state.grid_pes = [pes]
    state.total_pes = pes

    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]


def calc_uniform_pes() -> None:

    pes, layouts = _largest_decomposable(state.n_cpus, ntiles=6)
    state.grid_pes = [pes]
    state.total_pes = pes

    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]


def _largest_decomposable(ncpus: int, ntiles: int) -> tuple[int, dict]:
    """Largest PE count <= ncpus, a multiple of ntiles, with a valid layout."""
    for per_tile in range(ncpus // ntiles, 0, -1):
        try:
            return per_tile * ntiles, get_layouts([per_tile])
        except ValueError:
            continue
    raise ValueError(f"No domain decomposition fits within {ncpus} PEs")


def check_user_define_pes() -> bool:
    """Use fv_nest_nml grid_pes from the case input.nml/.yaml/.yml if set.

    The first existing file is read, matching fv3_namelists.namelist_overrides.
    """
    user_nml = state.run_dir / "input"
    suffixes = (".nml", ".yaml", ".yml")

    override_nml = None
    grid_pes = None
    for suffix in suffixes:
        _path = Path(user_nml).with_suffix(suffix)
        if _path.exists():
            override_nml = read_namelist(_path)
            break

    if override_nml:
        grid_pes = override_nml.get("fv_nest_nml", {}).get("grid_pes")
    if not grid_pes:
        return False

    grid_pes = [int(p) for p in grid_pes]
    if len(grid_pes) != state.n_nests + 1:
        raise ValueError(
            f"fv_nest_nml grid_pes needs {state.n_nests + 1} entries, got {grid_pes}"
        )
    if grid_pes[0] % 6 != 0:
        raise ValueError(f"Global grid_pes ({grid_pes[0]}) must be a multiple of 6")
    if sum(grid_pes) > state.n_cpus:
        raise ValueError(
            f"grid_pes {grid_pes} exceed the {state.n_cpus} tasks requested"
        )

    state.grid_pes = grid_pes
    state.total_pes = sum(grid_pes)

    layouts = get_layouts(p // d for p, d in zip(grid_pes, [6, *([1] * state.n_nests)]))

    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]

    return True


def calc_nest_pes() -> None:
    if check_user_define_pes():
        return

    timings = get_timings()
    k_split = np.asarray(timings["k_split"], dtype=np.float64)
    n_split = np.asarray(timings["n_split"], dtype=np.float64)

    grid_cells = np.asarray(
        [state.ngrid_cells[0], *state.ngrid_cells[1:]],
        dtype=np.float64,
    )
    subcycles = k_split * n_split

    if np.any(grid_cells <= 0) or np.any(subcycles <= 0):
        raise ValueError("Grid-cell counts and subcycle counts must be positive.")

    # Dynamics work is proportional to horizontal cells times acoustic subcycles.
    # Scale only for readability. allocate_pes() uses ratios, not magnitudes.
    global_base_pes = 6 * max(1, state.c_res // 96)
    weights = grid_cells * subcycles
    weights *= global_base_pes / weights[0]

    # Permit compact decompositions in four-rank increments. Restrict candidates
    # to layouts no more elongated than 2:1 before grid-specific orientation,
    # and per grid to counts it can be split into (get_layouts).
    counts = _square_pe_counts(range(16, state.n_cpus + 1, 4))
    valid = []
    for k in range(1, state.n_nests + 1):
        fits = _decomposable(counts, state.npx[k] - 1, state.npy[k] - 1)
        if fits.size == 0:
            raise ValueError(
                f"Nest {k + 1:02d} ({state.npx[k] - 1} x {state.npy[k] - 1} cells) is too "
                + f"small for 16 PEs with at least {MIN_LOCAL_CELLS} cells per edge"
            )
        valid.append(fits)

    final_pes = allocate_pes(
        weights=weights,
        ncpus=state.n_cpus,
        valid_nest_pes=valid,
        global_cells=(state.npx[0] - 1, state.npy[0] - 1),
    )

    ntiles_list = [6] + [1] * state.n_nests

    state.grid_pes = final_pes
    state.total_pes = sum(final_pes)

    layouts = get_layouts(
        [pes // ntiles for pes, ntiles in zip(final_pes, ntiles_list)]
    )

    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]

    save_fv3_state()


def _decomposable(counts: np.ndarray | range, nx: int, ny: int) -> np.ndarray:
    """Counts p with a layout lx * ly = p leaving at least MIN_LOCAL_CELLS
    cells per subdomain edge on an nx x ny grid (as required by get_layouts)."""
    ok = []
    for p in counts:
        p = int(p)
        if any(
            p % lx == 0
            and nx / lx >= MIN_LOCAL_CELLS
            and ny / (p // lx) >= MIN_LOCAL_CELLS
            for lx in range(1, p + 1)
        ):
            ok.append(p)
    return np.asarray(ok, dtype=np.int64)


def _square_pe_counts(counts: range, ratio: float = 2.0) -> np.ndarray:
    """Counts whose most nearly square factorization is at most ratio:1."""
    valid = []
    for pes in counts:
        layout_x = next(x for x in range(isqrt(pes), 0, -1) if pes % x == 0)
        if (pes // layout_x) / layout_x <= ratio:
            valid.append(pes)
    return np.asarray(valid, dtype=np.int64)


def allocate_pes(
    weights: list[float] | np.ndarray,
    ncpus: int,
    valid_nest_pes: list[int] | np.ndarray | list[np.ndarray],
    global_cells: tuple[int, int] | None = None,
) -> list[int]:
    """
    Allocate PEs by minimizing the largest estimated grid time:

        T_g ~ weight_g / P_g

    Rules:
        - global PE count is a multiple of 6 (and, with global_cells, its
          per-tile count decomposes the tile)
        - nest PE counts are selected from valid_nest_pes: one array for all
          nests, or one array per nest
        - use exactly ncpus when possible
        - among equivalent bottlenecks, prefer the smallest timing spread

    The last nest is determined by the others (the remainder of ncpus, or the
    largest valid count that fits), so the search spans one dimension fewer
    than the full product, and it is evaluated one global count at a time to
    bound memory. The result equals that of the full search.
    """
    weights = np.asarray(weights, dtype=np.float64)

    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError("weights must contain the global grid and at least one nest.")

    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("All PE weights must be finite and positive.")

    n_nest = len(weights) - 1
    per_nest = (
        list(valid_nest_pes)
        if len(valid_nest_pes) and np.ndim(valid_nest_pes[0]) == 1
        else [valid_nest_pes] * n_nest
    )
    if len(per_nest) != n_nest:
        raise ValueError("valid_nest_pes needs one array per nest.")
    per_nest = [np.asarray(v, dtype=np.int64) for v in per_nest]
    per_nest = [np.unique(v[(v >= 16) & (v <= ncpus)]) for v in per_nest]

    if any(v.size == 0 for v in per_nest):
        raise ValueError("No valid nest PE counts are available.")

    min_required = 6 + sum(int(v.min()) for v in per_nest)

    if min_required > ncpus:
        raise ValueError(
            f"Insufficient CPUs for PE allocation: ncpus={ncpus}, but at least {min_required} are required."
        )

    per_tile = _square_pe_counts(range(1, ncpus // 6 + 1))
    if global_cells is not None:
        per_tile = _decomposable(per_tile, *global_cells)
    global_valid = 6 * per_tile

    if global_valid.size == 0:
        raise ValueError("No valid global-grid PE counts are available.")

    n_inner = len(weights) - 2  # nests chosen freely; the last one is derived
    best_exact = None  # (bottleneck, spread, row)
    best_fit = None  # (total, bottleneck, spread, row)

    last_valid = per_nest[-1]
    if n_inner:
        mesh = np.meshgrid(*per_nest[:-1], indexing="ij")
        inner_all = np.stack([m.ravel() for m in mesh], axis=1)
    else:
        inner_all = np.empty((1, 0), dtype=np.int64)

    for g in global_valid:
        inner = inner_all
        remainder = ncpus - g - inner.sum(axis=1)

        # Largest valid count for the last nest that fits in the remainder.
        pos = np.searchsorted(last_valid, remainder, side="right") - 1
        fits = pos >= 0
        if not fits.any():
            continue
        last = last_valid[np.clip(pos, 0, None)]

        rows = np.column_stack([np.full(len(inner), g), inner, last])[fits]
        remainder = remainder[fits]

        predicted = weights[np.newaxis, :] / rows
        bottleneck = predicted.max(axis=1)
        spread = np.ptp(predicted, axis=1)

        # The last nest takes the whole remainder: all ncpus are used.
        exact = rows[:, -1] == remainder

        if exact.any():
            idx = np.flatnonzero(exact)
            k = idx[np.lexsort((spread[idx], bottleneck[idx]))[0]]
            cand = (bottleneck[k], spread[k], rows[k])
            if best_exact is None or cand[:2] < best_exact[:2]:
                best_exact = cand

        if best_exact is None:
            totals = rows.sum(axis=1)
            idx = np.flatnonzero(totals == totals.max())
            k = idx[np.lexsort((spread[idx], bottleneck[idx]))[0]]
            cand = (-int(totals[k]), bottleneck[k], spread[k], rows[k])
            if best_fit is None or cand[:3] < best_fit[:3]:
                best_fit = cand

    if best_exact is not None:
        return best_exact[2].astype(int).tolist()
    if best_fit is None:
        raise ValueError("No PE allocation fits within ncpus.")
    return best_fit[3].astype(int).tolist()


# Smallest compute-domain edge per PE. FV3 exchanges 3-point halos
# (fv_mp_mod ng = 3), and a subdomain narrower than the halo cannot be
# updated from its neighbours alone.
MIN_LOCAL_CELLS = 4


def get_layouts(pes: list[int]) -> dict[str, list[int]]:
    """Choose layout, io_layout and blocksize for each grid.

    Among the factor pairs of each PE count, the layout prefers, in order:
    subdomains no smaller than MIN_LOCAL_CELLS on either edge; a cell count
    that divides evenly in both directions, so every PE holds the same
    subdomain (mpp_define_domains otherwise distributes the remainder
    unevenly); and the most nearly square subdomain, which minimizes the
    halo-exchange perimeter per cell.
    """
    layouts = []
    io_layouts = []
    blocksizes = []

    for grid_index, grid_pes in enumerate(pes):
        if grid_pes <= 0:
            raise ValueError(f"Invalid PE count for grid {grid_index}: {grid_pes}")

        nx = state.npx[grid_index] - 1
        ny = state.npy[grid_index] - 1

        best_layout = None
        best_key = None

        for layout_x in range(1, isqrt(grid_pes) + 1):
            if grid_pes % layout_x != 0:
                continue

            layout_y = grid_pes // layout_x

            for x_layout, y_layout in (
                (layout_x, layout_y),
                (layout_y, layout_x),
            ):
                local_nx = nx / x_layout
                local_ny = ny / y_layout
                key = (
                    min(local_nx, local_ny) < MIN_LOCAL_CELLS,
                    nx % x_layout != 0 or ny % y_layout != 0,
                    abs(np.log(local_nx / local_ny)),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_layout = [x_layout, y_layout]

        if best_key[0]:
            raise ValueError(
                f"Grid {grid_index} ({nx} x {ny} cells) cannot be split over "
                + f"{grid_pes} PEs with at least {MIN_LOCAL_CELLS} cells per edge"
            )

        layouts.append(best_layout)
        io_layouts.append([1, 1])
        # Physics block of 32 columns, as in the SHiELD_build test cases.
        blocksizes.append(32)

    return {
        "layout": layouts,
        "io_layout": io_layouts,
        "blocksize": blocksizes,
    }
