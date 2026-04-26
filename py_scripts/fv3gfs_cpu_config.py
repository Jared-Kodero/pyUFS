from pathlib import Path

import xarray as xr
from fv3gfs_runtime import sort_paths
from fv3gfs_state import save_state, state

grid_dir: Path = None


def calc_cpu_alloc(dir: Path) -> None:
    global grid_dir
    grid_dir = dir
    get_n_grid_cells()
    if state.gtype == "nest":
        calc_nest_pes()
    else:
        calc_uniform_pes()


def get_n_grid_cells() -> None:
    state["nest_ngrid_cells"] = []
    state["global_ngrid_cells"] = 0

    files = sorted(list(grid_dir.glob("C*_grid.tile*.nc")), key=sort_paths)
    for f in files:
        tile_num = int(f.stem.split(".")[-1].replace("tile", ""))
        if tile_num < 6:
            continue

        with xr.open_dataset(f) as ds:
            cells = ds.nx.size * ds.ny.size

        if tile_num == 6:
            state["global_ngrid_cells"] = cells * 6
        else:
            state["nest_ngrid_cells"].append(cells)


def calc_uniform_pes() -> None:
    grid_path = grid_dir / f"C{state.res}_grid.tile{6}.nc"
    with xr.open_dataset(grid_path) as ds:
        npx = int((ds.nx.size // 2) + 1)
        npy = int((ds.ny.size // 2) + 1)

    nx = npx - 1
    ny = npy - 1
    target = state.n_cpus // 6

    valid = sorted(
        set(
            x * y
            for x in range(1, nx + 1)
            if nx % x == 0
            for y in range(1, ny + 1)
            if ny % y == 0
        )
    )

    pes_per_tile = max((v for v in valid if v <= target), default=1)
    total_pes = 6 * pes_per_tile

    state["grid_pes"] = [total_pes]
    state["total_pes"] = total_pes
    state["global_pes"] = total_pes

    set_layouts([total_pes], [6])


def calc_nest_pes() -> None:
    points = []
    nest_base_pes = []
    npx_list = []
    npy_list = []

    for t in range(6, 7 + state.n_nests):
        grid_path = grid_dir / f"C{state.res}_grid.tile{t}.nc"
        with xr.open_dataset(grid_path) as ds:
            npx = int((ds.nx.size // 2) + 1)
            npy = int((ds.ny.size // 2) + 1)
            nx = npx - 1
            ny = npy - 1

        npx_list.append(npx)
        npy_list.append(npy)
        points.append(nx * ny)

        if t > 6:
            nest_base_pes.append(_calc_nest_base_pes(nx, ny))

    # 2 PEs per 96x96 tile, scaled by resolution factor
    g_tile_pe = max(1, state.res // 96)
    g_base_pes = g_tile_pe * 6

    min_pes = g_base_pes + sum(nest_base_pes)
    if min_pes > state.n_cpus:
        raise ValueError(
            f"Insufficient CPUs: need at least {min_pes} ({g_base_pes} global + {sum(nest_base_pes)} per nest), got {state.n_cpus}."
        )

    nest_pes = []
    for p, nb in zip(points[1:], nest_base_pes):
        w = max(nb, int(p / points[0]))
        if w % nb != 0:
            w += (nb - w % nb) % nb
        nest_pes.append(w)

    base_pes = [g_base_pes] + nest_pes
    ntiles_list = [6] + [1] * state.n_nests

    snapped = []
    for p, npx, npy, ntiles in zip(
        nest_pes, npx_list[1:], npy_list[1:], ntiles_list[1:]
    ):
        valid = _valid_pes(npx - 1, npy - 1, ntiles=ntiles)
        snapped.append(_nearest_valid(p, valid, min(valid)))

    grid_pes = [base_pes[0]] + snapped

    if sum(grid_pes) > state.n_cpus:
        nest_pes = [
            min(_valid_pes(npx - 1, npy - 1, ntiles=ntiles))
            for npx, npy, ntiles in zip(npx_list[1:], npy_list[1:], ntiles_list[1:])
        ]
        grid_pes = [base_pes[0]] + nest_pes

    ratios = [max(1, round(p / points[0])) for p in points]
    grid_pes = _distribute_remaining_cpus(
        grid_pes=grid_pes,
        ratios=ratios,
        npx_list=npx_list,
        npy_list=npy_list,
        ntiles_list=ntiles_list,
        total_available=state.n_cpus,
    )

    total_pes = sum(grid_pes)
    state["grid_pes"] = grid_pes
    state["total_pes"] = total_pes
    state["global_pes"] = grid_pes[0]

    set_layouts(grid_pes, ntiles_list)


def _valid_pes(
    nx: int,
    ny: int,
    ntiles: int = 1,
    max_div: int | None = None,
) -> list[int]:
    if max_div is None:
        max_div = max(nx, ny)
    valid = set(
        ntiles * x * y
        for x in range(1, min(nx, max_div) + 1)
        if nx % x == 0
        for y in range(1, min(ny, max_div) + 1)
        if ny % y == 0
    )
    return sorted(valid)


def _calc_nest_base_pes(nx: int, ny: int, min_div: int = 1, max_div: int = 32) -> int:
    """
    Return a small valid base PE count for a nested FV3 tile.
    Selects the factor pair (x, y) with the most square subdomain layout,
    breaking ties by preferring smaller total PE count.
    Falls back to 1 if no valid factor pair is found.
    """
    best = None
    best_key = None

    for x in range(min_div, max_div + 1):
        if nx % x != 0:
            continue
        for y in range(min_div, max_div + 1):
            if ny % y != 0:
                continue
            pes = x * y
            key = (abs(x - y), pes)
            if best_key is None or key < best_key:
                best_key = key
                best = pes

    return best if best is not None else 1


def _nearest_valid(target: int, valid: list[int], minimum: int) -> int:
    """
    Return the largest valid value <= target, but never below minimum.
    Falls back to the smallest valid value >= minimum.
    """
    candidates = [v for v in valid if minimum <= v <= target]
    if candidates:
        return candidates[-1]
    candidates = [v for v in valid if v >= minimum]
    if candidates:
        return candidates[0]
    return minimum


def _distribute_remaining_cpus(
    grid_pes: list[int],
    ratios: list[int],
    npx_list: list[int],
    npy_list: list[int],
    ntiles_list: list[int],
    total_available: int,
) -> list[int]:
    """
    Distribute remaining CPUs across parent and nest grids proportionally,
    ensuring each allocation remains geometrically decomposable.
    """

    # reverse each list to prioritize nests
    current = grid_pes[::-1]
    ratios = ratios[::-1]
    npx_list = npx_list[::-1]
    npy_list = npy_list[::-1]
    ntiles_list = ntiles_list[::-1]

    def _best_increment(current: int, remaining: int, valid: list[int]) -> int | None:
        for v in valid:
            inc = v - current
            if 0 < inc <= remaining:
                return inc
        return None

    valid_sets = []
    minimums = []
    for i in range(len(current)):
        nx = npx_list[i] - 1
        ny = npy_list[i] - 1
        valid = _valid_pes(nx, ny, ntiles=ntiles_list[i])
        valid_sets.append(valid)
        minimums.append(min(valid))
        if current[i] not in valid:
            current[i] = _nearest_valid(current[i], valid, minimums[i])

    if sum(current) >= total_available:
        return current

    remaining = total_available - sum(current)
    total_ratio = sum(ratios)

    for i in range(len(current)):
        share = (ratios[i] / total_ratio) * remaining
        target = current[i] + int(share)
        current[i] = _nearest_valid(target, valid_sets[i], current[i])

    used = sum(current)
    while used > total_available:
        reduced = False
        for i in reversed(range(len(current))):  # Now checks Global (last index) first
            candidates = [v for v in valid_sets[i] if minimums[i] <= v < current[i]]
            if candidates:
                current[i] = candidates[-1]
                used = sum(current)
                reduced = True
                if used <= total_available:
                    break
        if not reduced:
            break

    remaining = total_available - sum(current)
    while remaining > 0:
        best_i, best_inc, best_score = None, None, None
        for i in range(len(current)):
            inc = _best_increment(current[i], remaining, valid_sets[i])
            if inc is None:
                continue
            score = (inc, -ratios[i])
            if best_score is None or score < best_score:
                best_score = score
                best_inc = inc
                best_i = i
        if best_i is None:
            break
        current[best_i] += best_inc
        remaining -= best_inc

    return current[::-1]  # reverse back to original order


def _best_layout(pes_per_tile: int, nx: int, ny: int) -> list[int]:
    """
    Return [layout_x, layout_y] such that layout_x * layout_y == pes_per_tile,
    nx % layout_x == 0, ny % layout_y == 0, and subdomain aspect ratio is
    closest to 1.0 (minimizes halo exchange imbalance).
    Falls back to [1, 1] if no valid factorization exists.
    """
    best_layout = None
    best_score = float("inf")

    for x in range(1, pes_per_tile + 1):
        if pes_per_tile % x != 0:
            continue
        y = pes_per_tile // x
        if nx % x != 0 or ny % y != 0:
            continue
        score = abs((nx // x) / (ny // y) - 1.0)
        if score < best_score:
            best_score = score
            best_layout = [x, y]

    return best_layout if best_layout is not None else [1, 1]


def set_layouts(pes: list, ntiles: list) -> None:
    for k in {"layout", "io_layout", "blocksize", "ntiles", "npx", "npy"}:
        state[k] = []

    tile_ids = list(range(6, 7 + state.n_nests))

    for p, t, n in zip(pes, tile_ids, ntiles):
        if p % n != 0:
            raise ValueError(f"pes ({p}) must be divisible by ntiles ({n})")

        grid_file = grid_dir / f"C{state.res}_grid.tile{t}.nc"
        with xr.open_dataset(grid_file) as ds:
            npx = int((ds.nx.size // 2) + 1)
            npy = int((ds.ny.size // 2) + 1)

        nx = npx - 1
        ny = npy - 1
        pes_per_tile = p // n

        layout = _best_layout(pes_per_tile, nx, ny)

        state.layout.append(layout)
        state.io_layout.append([1, 1])
        state.ntiles.append(n)
        state.npx.append(npx)
        state.npy.append(npy)
        state.blocksize.append(32)

    save_state()
