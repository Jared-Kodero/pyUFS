# pes_config.py

from math import isqrt
from pathlib import Path

import numpy as np
import xarray as xr
from fv3_runtime import read_namelist, sort_paths
from fv3_state import save_fv3_state, state

grid_dir: Path = None


def calc_cpu_alloc(dir: Path) -> None:
    global grid_dir
    grid_dir = dir
    get_grid_info()
    if state.gtype == "nest":
        calc_nest_pes()
    else:
        calc_uniform_pes()


def get_grid_info() -> None:
    state.nest_ngrid_cells = []
    state.global_ngrid_cells = 0
    state.ntiles = []
    state.npx = []
    state.npy = []

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
            state.global_ngrid_cells = cells * n
        else:
            n = 1
            state.nest_ngrid_cells.append(cells)

        state.ntiles.append(n)
        state.npx.append(npx)
        state.npy.append(npy)


def calc_uniform_pes() -> None:

    total_pes = 6 * (state.n_cpus // 6)
    state.grid_pes = [total_pes]
    state.total_pes = total_pes
    state.global_pes = total_pes

    layouts = get_layouts([total_pes // 6])
    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]


def check_user_define_pes() -> bool:
    global_input_nml = state.get("global_input_nml")
    if not global_input_nml:
        return False

    if isinstance(global_input_nml, (str, Path)):
        path = Path(global_input_nml)
        if not path.exists():
            return False
        global_input_nml = read_namelist(path)
        if not global_input_nml:
            return False

    grid_pes = global_input_nml.get("fv_nest_nml", {}).get("grid_pes")
    if not grid_pes:
        return False

    state.grid_pes = grid_pes
    state.total_pes = sum(grid_pes)
    state.global_pes = grid_pes[0]

    layouts = get_layouts(p // d for p, d in zip(grid_pes, [6, *([1] * state.n_nests)]))

    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]

    return True


def calc_nest_pes() -> None:
    if check_user_define_pes():
        return

    global_base_pes = 6 * max(1, state.res // 96)
    nest_base_pes = []

    for n_cells in state.nest_ngrid_cells:
        nest_i_base_pe = int((n_cells * global_base_pes) / state.global_ngrid_cells)
        nest_base_pes.append(nest_i_base_pe)

    weights = [global_base_pes] + nest_base_pes
    valid = np.array([16, 32, 64, 128, 256], dtype=np.int64)

    final_pes = allocate_pes(
        weights=weights,
        ncpus=state.n_cpus,
        valid_nest_pes=valid,
    )

    ntiles_list = [6] + [1] * state.n_nests

    total_pes = sum(final_pes)
    state.grid_pes = final_pes
    state.total_pes = total_pes
    state.global_pes = final_pes[0]

    layouts = get_layouts([p // n for p, n in zip(final_pes, ntiles_list)])

    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]

    save_fv3_state()


def allocate_pes(
    weights: list[int],
    ncpus: int,
    valid_nest_pes: list[int] | np.ndarray,
) -> list[int]:
    """
    Vectorized PE allocation.

    weights[0]  = global PE weight
    weights[1:] = nest PE weights

    Rules:
        - global PE count must be a multiple of 6
        - each nest PE count must be in valid_nest_pes
        - total PE count must equal ncpus if possible
        - selected layout stays closest to the weighted PE ratios
    """

    weights = np.asarray(weights, dtype=np.float64)
    valid = np.asarray(valid_nest_pes, dtype=np.int64)

    min_required = sum(weights)
    if min_required > ncpus:
        raise ValueError(
            f"Insufficient CPUs for PE allocation: ncpus={ncpus}, but at least {min_required} is needed!"
        )

    global_valid = np.arange(6, ncpus + 1, 6, dtype=np.int64)

    choices = [global_valid] + [valid] * (len(weights) - 1)
    mesh = np.meshgrid(*choices, indexing="ij")
    candidates = np.stack([m.ravel() for m in mesh], axis=1)

    exact = candidates[candidates.sum(axis=1) == ncpus]

    if exact.size == 0:
        candidates = candidates[candidates.sum(axis=1) <= ncpus]
        candidates = candidates[candidates.sum(axis=1) == candidates.sum(axis=1).max()]
    else:
        candidates = exact

    target = ncpus * weights / weights.sum()

    score = np.sum(((candidates - target) / target) ** 2, axis=1)

    return candidates[np.argmin(score)].astype(int).tolist()


def get_layouts(pes: list[int]) -> dict[str, list[int]]:
    layouts = []
    io_layouts = []
    blocksizes = []

    for p in pes:
        for layout_x in range(isqrt(p), 0, -1):
            if p % layout_x == 0:
                layout = [layout_x, p // layout_x]
                break

        layouts.append(layout)
        io_layouts.append([1, 1])
        blocksizes.append(32)

    return {
        "layout": layouts,
        "io_layout": io_layouts,
        "blocksize": blocksizes,
    }
