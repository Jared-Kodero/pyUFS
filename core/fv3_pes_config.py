# pes_config.py

from math import ceil, isqrt
from pathlib import Path

import xarray as xr

from fv3_state import save_fv3_state, state
from fv3_timings import get_timings
from fv3_utils import open_nml, open_yaml, sort_paths

# FV3 stops when npx/layout(1) or npy/layout(2) is below NG, with integer division
# and npx = cells + 1 (domain_decomp in tools/fv_mp_mod.F90, ng = 3).
# MIN_EDGE is the floor used when
# pyUFS chooses the layout itself; it is an engineering margin, not a model limit.
# PEs given explicitly in fv_nest_nml.grid_pes are only held to NG.
NG = 3
MIN_EDGE = 8
# Edges tried in turn, so that a small grid or an explicit PE count gets the best
# available layout instead of stopping the run.

# Range searched for a physics block size around the usual 32 points.
BLOCK_RANGE = range(16, 65)


def calc_cpu_alloc(dir: Path) -> None:
    get_grid_info(dir)
    if state.gtype == "nest":
        calc_nest_pes()
    elif state.gtype in ("regional_gfdl", "regional_esg"):
        calc_regional_pes()
    else:
        calc_uniform_pes()


def get_grid_info(grid_dir: Path) -> None:
    state.ngrid_cells = [0 for _ in range(state.n_nests + 1)]
    state.ntiles = []
    state.npx = []
    state.npy = []

    if state.gtype in ("regional_gfdl", "regional_esg"):
        get_regional_grid_info(grid_dir)
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


def get_regional_grid_info(grid_dir: Path) -> None:
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
    """Standalone single-tile regional grid: the most useful number of PEs up to n_cpus."""
    if state.n_cpus <= 0:
        raise ValueError(f"Invalid CPU count for regional grid: {state.n_cpus}")

    set_single_grid_pes(ntiles=1)


def calc_uniform_pes() -> None:
    """Uniform cubed sphere: six tiles of equal size, so a multiple of six PEs."""
    set_single_grid_pes(ntiles=6)


def set_single_grid_pes(ntiles: int) -> None:
    """Use the PE count (a multiple of ntiles, at most n_cpus) with the smallest subdomain.

    The largest rank sets the step time, and an awkward count, a prime per tile for
    example, forces thin strips. Such a count costs more than a smaller one, so the
    count is not always n_cpus. Equal cost goes to the smaller count, which has less
    halo exchange.
    """
    nx, ny = state.npx[0] - 1, state.npy[0] - 1
    table = cost_table(nx, ny, ntiles, state.n_cpus)
    if not table:
        raise ValueError(
            f"n_cpus={state.n_cpus} is fewer than the {ntiles} tile(s) of the grid."
        )
    total_pes = min(table, key=lambda total: (table[total][0], total))

    state.grid_pes = [total_pes]
    state.total_pes = total_pes
    apply_layouts([total_pes // ntiles])


def apply_layouts(pes_per_tile: list[int], min_edge: int = MIN_EDGE) -> None:
    layouts = get_layouts(pes_per_tile, min_edge)
    state.layout = layouts["layout"]
    state.io_layout = layouts["io_layout"]
    state.blocksize = layouts["blocksize"]


def check_user_define_pes() -> bool:

    user_nml = state.submit_dir / "input"
    suffixes = (".nml", ".yaml", ".yml")

    override_nml = None
    grid_pes = None
    for suffix in suffixes:
        _path = Path(user_nml).with_suffix(suffix)
        if not _path.exists():
            continue
        else:
            override_nml = open_nml(_path) if suffix == ".nml" else open_yaml(_path)

    if override_nml:
        grid_pes = override_nml.get("fv_nest_nml", {}).get("grid_pes")
    if not grid_pes:
        return False

    state.grid_pes = grid_pes
    state.total_pes = sum(grid_pes)

    apply_layouts(
        [p // d for p, d in zip(grid_pes, [6, *([1] * state.n_nests)])], min_edge=NG
    )

    return True


def calc_nest_pes() -> None:
    if check_user_define_pes():
        return

    timings = get_timings()
    subcycles = [float(k * n) for k, n in zip(timings["k_split"], timings["n_split"])]
    if any(value <= 0 for value in subcycles):
        raise ValueError("k_split and n_split must be positive.")

    # Dynamics work per cell and atmospheric step is proportional to k_split * n_split.
    ntiles = [6] + [1] * state.n_nests
    grids = [
        (weight, state.npx[i] - 1, state.npy[i] - 1, ntiles[i])
        for i, weight in enumerate(subcycles)
    ]
    if any(nx <= 0 or ny <= 0 for _, nx, ny, _ in grids):
        raise ValueError("Grid sizes must be positive.")

    final_pes = allocate_pes(grids, state.n_cpus)

    state.grid_pes = final_pes
    state.total_pes = sum(final_pes)
    apply_layouts([pes // nt for pes, nt in zip(final_pes, ntiles)])

    save_fv3_state()


def edge_ok(cells: int, ranks: int, min_edge: int) -> bool:
    """Whether ``cells`` split over ``ranks`` meets the floor ``min_edge``.

    Up to NG the test is FV3's own, (cells + 1) // ranks >= min_edge, since FV3 divides
    the number of corners. Above NG it is our margin, on the smallest rank."""
    if min_edge > NG:
        return cells // ranks >= min_edge
    return (cells + 1) // ranks >= min_edge


def tile_layout(
    nx: int, ny: int, pes: int, min_edge: int = MIN_EDGE
) -> tuple[tuple[int, int, int], tuple[int, int]] | None:
    """Best (lx, ly) layout of ``pes`` ranks on an nx by ny tile, or None if none fits.

    FMS gives each rank floor(nx/lx) or ceil(nx/lx) cells in x (likewise y), and every
    step waits for the largest rank, so the cost is ceil(nx/lx) * ceil(ny/ly) whether
    or not the layout divides the tile. Ties go to the shorter halo perimeter, then to
    an exact division. Returns ((cost, perimeter, uneven), (lx, ly)).
    """
    best = None
    for small in range(1, isqrt(pes) + 1):
        if pes % small:
            continue
        for lx, ly in {(small, pes // small), (pes // small, small)}:
            if not (edge_ok(nx, lx, min_edge) and edge_ok(ny, ly, min_edge)):
                continue
            cx, cy = ceil(nx / lx), ceil(ny / ly)
            key = (cx * cy, cx + cy, int(nx % lx != 0 or ny % ly != 0))
            if best is None or (key, lx) < (best[0], best[1][0]):
                best = (key, (lx, ly))
    return best


def fit_layout(nx: int, ny: int, pes: int, min_edge: int) -> list[int]:
    """Layout of ``pes`` ranks per tile; never fails.

    The edge floor is lowered step by step (min_edge, NG, 1, none) until a layout
    exists. Below NG cells per rank FV3 itself stops, which is the
    case only for a PE count far too large for the grid.
    """
    for edge in dict.fromkeys((min_edge, min(min_edge, NG), 1, 0)):
        best = tile_layout(nx, ny, pes, edge)
        if best is not None:
            return list(best[1])
    raise AssertionError("tile_layout(edge=0) always has a layout for pes >= 1")


def cost_table(
    nx: int, ny: int, ntiles: int, ncpus: int, min_edge: int = MIN_EDGE
) -> dict[int, tuple[int, tuple[int, int]]]:
    """{total PEs: (largest rank's cells, layout)} for a grid of ntiles equal tiles.

    The edge floor is lowered (MIN_EDGE, NG, 1) only when the grid is so small that
    no PE count meets it."""
    for edge in dict.fromkeys((min_edge, min(min_edge, NG), 1)):
        table = {}
        for total in range(ntiles, ncpus + 1, ntiles):
            best = tile_layout(nx, ny, total // ntiles, edge)
            if best:
                table[total] = (best[0][0], best[1])
        if table:
            return table
    return {}


def allocate_pes(grids: list[tuple[float, int, int, int]], ncpus: int) -> list[int]:
    """Split ncpus over the grids so that the slowest grid is as fast as possible.

    ``grids`` holds (weight per cell, nx, ny, ntiles) for the global grid and each
    nest. The time of a grid is weight * (cells of its largest rank), and the grids
    run concurrently, so the allocation minimises the largest time with
    sum(PEs) <= ncpus. The global grid takes a multiple of 6. The search is exact:
    for a trial time every grid takes the fewest PEs that meet it, and the smallest
    time that fits in ncpus is found by bisection. PEs left over then go to the
    slowest grids while that still lowers their time.
    """
    if not grids:
        raise ValueError("No grids to allocate.")
    for weight, nx, ny, ntiles in grids:
        if not weight > 0 or nx <= 0 or ny <= 0 or ntiles <= 0:
            raise ValueError("Grid weights and sizes must be finite and positive.")

    tables = [cost_table(nx, ny, nt, ncpus) for _, nx, ny, nt in grids]
    weights = [g[0] for g in grids]
    for index, table in enumerate(tables):
        if not table:
            raise ValueError(
                f"Insufficient CPUs for PE allocation: ncpus={ncpus} cannot cover grid {index}."
            )

    def fewest(index: int, limit: float) -> int | None:
        fits = [
            pes
            for pes, (cells, _) in tables[index].items()
            if weights[index] * cells <= limit
        ]
        return min(fits) if fits else None

    def needs(limit: float) -> list[int] | None:
        pes = [fewest(i, limit) for i in range(len(grids))]
        if None in pes or sum(pes) > ncpus:
            return None
        return pes

    times = sorted(
        {weights[i] * cells for i, t in enumerate(tables) for cells, _ in t.values()}
    )
    low, high = 0, len(times) - 1
    if needs(times[high]) is None:
        floor = sum(min(t) for t in tables)
        raise ValueError(
            f"Insufficient CPUs for PE allocation: ncpus={ncpus}, at least {floor} are needed."
        )
    while low < high:
        mid = (low + high) // 2
        if needs(times[mid]) is not None:
            high = mid
        else:
            low = mid + 1
    pes = needs(times[low])

    spare = ncpus - sum(pes)
    while spare > 0:
        slowest_first = sorted(
            range(len(pes)), key=lambda i: -weights[i] * tables[i][pes[i]][0]
        )
        for i in slowest_first:
            now = tables[i][pes[i]][0]
            better = [
                (cells, p)
                for p, (cells, _) in tables[i].items()
                if pes[i] < p <= pes[i] + spare and cells < now
            ]
            if better:
                _, p = min(better)
                spare -= p - pes[i]
                pes[i] = p
                break
        else:
            break

    return pes


def block_size(nx: int, ny: int, layout: tuple[int, int]) -> int:
    """Physics block size for a grid, or -1 (one block per rank).

    block_control_mod packs each rank's compute domain into blocks of this size and
    only warns when the last block is short. A size that divides every rank's domain
    avoids that. Rank domains are floor or ceil of the tile size over the layout, so
    at most four sizes occur. The size nearest 32 that divides all of them is used.
    """
    lx, ly = layout
    sizes = {
        a * b for a in {nx // lx, ceil(nx / lx)} for b in {ny // ly, ceil(ny / ly)}
    }
    for block in sorted(BLOCK_RANGE, key=lambda b: (abs(b - 32), b)):
        if all(size % block == 0 for size in sizes):
            return block
    return -1


def get_layouts(pes: list[int], min_edge: int = MIN_EDGE) -> dict[str, list]:
    """Layout, io_layout and block size of each grid from its PEs per tile."""
    layouts = []
    io_layouts = []
    blocksizes = []

    for grid_index, grid_pes in enumerate(pes):
        if grid_pes <= 0:
            raise ValueError(f"Invalid PE count for grid {grid_index}: {grid_pes}")

        nx = state.npx[grid_index] - 1
        ny = state.npy[grid_index] - 1

        layout = fit_layout(nx, ny, grid_pes, min_edge)
        layouts.append(layout)
        io_layouts.append([1, 1])
        blocksizes.append(block_size(nx, ny, layout))

    return {
        "layout": layouts,
        "io_layout": io_layouts,
        "blocksize": blocksizes,
    }
