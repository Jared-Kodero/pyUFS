# pes_config.py

from math import isqrt
from pathlib import Path

import numpy as np
import xarray as xr
from fv3_runtime import read_namelist, sort_paths
from fv3_state import save_fv3_state, state
from fv3_timings import get_timings

grid_dir: Path = None


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

    files = sorted(list(grid_dir.glob("C*_grid.tile*.nc")), key=sort_paths)

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
    """Read the single, unshaved tile-7 grid used by a regional domain."""
    files = sorted(grid_dir.glob("C*_grid.tile7.nc"), key=sort_paths)

    if not files:
        raise FileNotFoundError(
            f"No regional grid file matching C*_grid.tile7.nc found in {grid_dir}."
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

    state.grid_pes = [state.n_cpus]
    state.total_pes = state.n_cpus

    layouts = get_layouts(state.grid_pes)
    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]


def calc_uniform_pes() -> None:

    total_pes = 6 * (state.n_cpus // 6)
    state.grid_pes = [total_pes]
    state.total_pes = total_pes

    layouts = get_layouts([total_pes // 6])
    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]


def check_user_define_pes() -> bool:

    user_nml = state.run_dir / "input"
    suffixes = (".nml", ".yaml", ".yml")

    override_nml = None
    grid_pes = None
    for suffix in suffixes:
        _path = Path(user_nml).with_suffix(suffix)
        if not _path.exists():
            continue
        else:
            override_nml = read_namelist(_path)

    if override_nml:
        grid_pes = override_nml.get("fv_nest_nml", {}).get("grid_pes")
    if not grid_pes:
        return False

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
    # to layouts no more elongated than 2:1 before grid-specific orientation.
    valid_nest_pes = []

    for pes in range(16, state.n_cpus + 1, 4):
        for layout_x in range(isqrt(pes), 0, -1):
            if pes % layout_x == 0:
                layout_y = pes // layout_x

                if layout_y / layout_x <= 2.0:
                    valid_nest_pes.append(pes)

                break

    valid = np.asarray(valid_nest_pes, dtype=np.int64)

    final_pes = allocate_pes(
        weights=weights,
        ncpus=state.n_cpus,
        valid_nest_pes=valid,
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


def allocate_pes(
    weights: list[float] | np.ndarray,
    ncpus: int,
    valid_nest_pes: list[int] | np.ndarray,
) -> list[int]:
    """
    Allocate PEs by minimizing the largest estimated grid time:

        T_g ~ weight_g / P_g

    Rules:
        - global PE count is a multiple of 6
        - nest PE counts are selected from valid_nest_pes
        - use exactly ncpus when possible
        - among equivalent bottlenecks, prefer the smallest timing spread
    """
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.asarray(valid_nest_pes, dtype=np.int64)

    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError("weights must contain the global grid and at least one nest.")

    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("All PE weights must be finite and positive.")

    valid = np.unique(valid[(valid >= 16) & (valid <= ncpus)])

    if valid.size == 0:
        raise ValueError("No valid nest PE counts are available.")

    min_required = 6 + (len(weights) - 1) * int(valid.min())

    if min_required > ncpus:
        raise ValueError(
            f"Insufficient CPUs for PE allocation: ncpus={ncpus}, but at least {min_required} are required."
        )

    global_valid_list = []

    for global_pes in range(6, ncpus + 1, 6):
        pes_per_tile = global_pes // 6

        for layout_x in range(isqrt(pes_per_tile), 0, -1):
            if pes_per_tile % layout_x == 0:
                layout_y = pes_per_tile // layout_x

                if layout_y / layout_x <= 2.0:
                    global_valid_list.append(global_pes)

                break

    global_valid = np.asarray(global_valid_list, dtype=np.int64)

    if global_valid.size == 0:
        raise ValueError("No valid global-grid PE counts are available.")

    choices = [global_valid] + [valid] * (len(weights) - 1)

    mesh = np.meshgrid(*choices, indexing="ij")
    candidates = np.stack([entry.ravel() for entry in mesh], axis=1)

    exact = candidates[candidates.sum(axis=1) == ncpus]

    if exact.size == 0:
        candidates = candidates[candidates.sum(axis=1) <= ncpus]

        if candidates.size == 0:
            raise ValueError("No PE allocation fits within ncpus.")

        max_used_pes = candidates.sum(axis=1).max()
        candidates = candidates[candidates.sum(axis=1) == max_used_pes]
    else:
        candidates = exact

    predicted_time = weights[np.newaxis, :] / candidates

    bottleneck_time = predicted_time.max(axis=1)
    timing_spread = np.ptp(predicted_time, axis=1)

    best = np.lexsort((timing_spread, bottleneck_time))[0]

    return candidates[best].astype(int).tolist()


def get_layouts(pes: list[int]) -> dict[str, list[int]]:
    layouts = []
    io_layouts = []
    blocksizes = []

    for grid_index, grid_pes in enumerate(pes):
        if grid_pes <= 0:
            raise ValueError(f"Invalid PE count for grid {grid_index}: {grid_pes}")

        nx = state.npx[grid_index] - 1
        ny = state.npy[grid_index] - 1

        best_layout = None
        best_score = np.inf

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

                # Prefer locally square subdomains while respecting the grid shape.
                score = abs(np.log(local_nx / local_ny))

                if score < best_score:
                    best_score = score
                    best_layout = [x_layout, y_layout]

        if best_layout is None:
            # Fall back to the factor pair that gives the most nearly square
            # local domains, even when nx and ny are not exactly divisible.
            fallback_layout = None
            fallback_score = np.inf

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

                    score = abs(np.log(local_nx / local_ny))

                    if score < fallback_score:
                        fallback_score = score
                        fallback_layout = [x_layout, y_layout]

            # grid_pes > 0 guarantees that [1, grid_pes] is available.
            best_layout = fallback_layout or [1, grid_pes]

        layouts.append(best_layout)
        io_layouts.append([1, 1])
        blocksizes.append(32)

    return {
        "layout": layouts,
        "io_layout": io_layouts,
        "blocksize": blocksizes,
    }
