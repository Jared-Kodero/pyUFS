"""Build SHiELD SST, sea-ice and land-mask climatologies from the ESA SST CCI v3.0 climatology.

Product
-------
ESA SST CCI Climatology v3.0 (Embury et al., 2024, doi:10.1038/s41597-024-03147-w),
CEDA catalogue record 62800d3d2227449085b430b503d36b01. It is the 1991-2020
daily climatology of the SST CCI L4 analysis (20 cm depth SST): 365 files, one per
day of a 365-day year (February 29 is skipped; the time axis uses 2020 as the
nominal year), each on a global 0.05 degree grid (3600 x 7200) with the variables
``analysed_sst`` (K), ``sea_ice_fraction`` (1) and ``mask`` (bit flags: 1 water,
2 land, 4 lake, 8 sea ice, 16 river) plus minimum, maximum and standard deviation
fields. The files are public: no token is needed (an optional ``CEDA_TOKEN``
environment variable is sent as a bearer token when set).

How the files are downloaded
----------------------------
The 365 files ``D001..D365-ESACCI-L4_GHRSST-SSTdepth-Climatology-GLOB_CDR3.0-v02.0-fv01.0.nc``
(about 41 MB each, 15 GB in all) are fetched from
``https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/Climatology/L4/v3.0.1/``
with ``--workers`` parallel streams (default 4), each into a ``.part`` file that is
renamed when complete, five attempts per file. Existing complete files are reused,
so an interrupted run resumes. The daily files are deleted after they have been
aggregated unless ``--keep-daily`` is given.

How the files are built
-----------------------
1. Monthly means. Every daily field is area-averaged from 0.05 degree onto the 1/12 degree
   grid (4320 x 2160) of ``fix_src/am/RTGSST.1982.2012.monthly.clim.grb``, with the
   exact overlap of the cells (separable weights, so no interpolation error; the
   longitude is rolled to 0..360). Land cells of the CCI mask (bit 2) are excluded,
   so a cell holds the mean over its ocean part and its weight is the ocean fraction.
   The sum of the daily values and the sum of the weights are accumulated per
   calendar month of the 365-day year (31, 28, 31, ... days) and divided at the end:
   a day-weighted monthly mean. The result (about 1.9 GB) is cached as
   ``cache/monthly.npz``, so later runs only redo the final step.
2. Land mask. A target cell is land where at least half of its area is land in the CCI
   mask (the mask is the same on every day). The CCI mask counts inland water as land,
   but the stock SHiELD mask has the Great Lakes and other lakes as water with SST
   and ice. These are kept as water: a lake is a component of the stock 0.5 degree
   ocean, not the world ocean, that the CCI mask covers with land in at least 90 % of
   its cells. At lake cells the stock SST and ice values are used, as the CCI has none.
3. SST. Ocean cells take the monthly mean, limited to 271.2..313.0 K, the range of
   the surface cycle (``sfcsub.F``); cells with no valid value take the nearest valid
   one. Land cells keep the stock SHiELD values: ``sfcsub.F`` weights the climatology
   by the mask and never reads land values at ocean points, and the files must hold
   numbers, since GRIB1 simple packing has no NaN and SHiELD ignores a bitmap for SST.
4. Sea ice. Ocean cells take the monthly mean of ``sea_ice_fraction``, limited to 0..1;
   land cells carry the land flag 1.57 of the stock file, so mask, SST and ice agree at
   every cell by construction (the stock files do not: 3155 cells of the 0.5 degree
   grid differ).
5. GRIB. The three files use the grid of the stock SST file, which is below the
   5000 x 2500 limit of the mask and climatology readers of ``sfcsub.F`` (the
   full 0.05 degree grid would abort the run). SST is written with ``grib_io.save_grib``
   through the messages of the stock SST file (headers, packing, month stamps and
   order unchanged). Sea ice and the mask are written through skeletons of the stock
   0.5 degree files: each stock message is cloned with the 1/12 degree grid of the SST
   file and a placeholder pattern that keeps its packing (8 bits and 1 bit), opened with
   ``grib_io.open_grib``, filled with the values and saved with ``grib_io.save_grib``.
   The month stamps of the stock files are kept (1996 for SST, 1982 for sea ice): the
   surface cycle matches the month of the message, not the year. The written files are
   read back and checked against the arrays, and for consistency of mask, ice and SST.

Output
------
``<root>/external/ESA/`` (``<root>`` is ``fix_src`` of the case, as for
``tools/download_pamip_amip.py``)::

    sst.ESACCI_1991_2020.monthly.clim.grb   SST, K, 12 months (format of the stock FNTSFC file)
    sic.ESACCI_1991_2020.monthly.clim.grb   sea ice, 12 months, 1.57 on land (stock FNAISC format)
    landmask.ESACCI.grb                     land mask, 1 = land (stock FNMSKH format)
    cache/                                  daily files (removed) and monthly.npz

The stock files must be in ``<root>/am`` (``tools/update_fix.py``). The three outputs
go together: use all of them (FNTSFC, FNAISC and FNMSKH of namsfc), never a mix with
stock files. Memory is about 5 GB, disk 2 GB (17 GB with ``--keep-daily``), run time about
15 minutes. Differences from the stock climatology (1982-2012, RTG SST, CFSR ice) are
dataset and period differences, not only climate change: CCI SST is a 20 cm depth
analysis and its sea ice comes from a different product.

To use a set, give the three paths in ``run_config.yaml`` (relative to ``fix_src``);
they are copied to ``FIXED/external`` at the cold start::

    external_sst: external/ESA/sst.ESACCI_1991_2020.monthly.clim.grb
    external_sic: external/ESA/sic.ESACCI_1991_2020.monthly.clim.grb
    external_mask: external/ESA/landmask.ESACCI.grb

With ``pamip_experiment`` and ``pamip_baseline: external`` the PAMIP anomaly is added to the copies.

Usage::

    python tools/esa_sst.py --root <fix_src>
    python tools/esa_sst.py --root <fix_src> --workers 8 --keep-daily
    python tools/esa_sst.py --root <fix_src> --force       # download and aggregate again
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eccodes
import numpy as np
import requests
import xarray as xr
from scipy import ndimage, sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from grib_io import open_grib, save_grib

BASE = (
    "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/"
    "sea_surface_temperature/CDR_v3/Climatology/L4/v3.0.1/"
)
NAME = "D{:03d}-ESACCI-L4_GHRSST-SSTdepth-Climatology-GLOB_CDR3.0-v02.0-fv01.0.nc"
DAYS = range(1, 366)
EXTERNAL_DIR = (
    "external"  # directory of fix_src that holds the PAMIP, AMIP and ESA files
)
ESA_DIR = "ESA"
TAG = "ESACCI_1991_2020"
SST_STOCK = "RTGSST.1982.2012.monthly.clim.grb"
ICE_STOCK = "CFSR.SEAICE.1982.2012.monthly.clim.grb"
MASK_STOCK = "seaice_newland.grb"
SST_OUT = f"sst.{TAG}.monthly.clim.grb"
ICE_OUT = f"sic.{TAG}.monthly.clim.grb"
MASK_OUT = "landmask.ESACCI.grb"
CACHE_VERSION = "esa-cci-v3.0.1-1"

SST_MIN_K, SST_MAX_K = 271.2, 313.0  # sfcsub.F: SST outside this range is replaced
LAND_FLAG = 1.57  # sea-ice land flag of the stock file
LAKE_LAND_SHARE = 0.9  # share of a stock lake covered by CCI land
VALID_SHARE = 0.25  # least ocean share of a cell for a monthly mean
GRID_KEYS = (
    "Ni",
    "Nj",
    "latitudeOfFirstGridPoint",
    "latitudeOfLastGridPoint",
    "longitudeOfFirstGridPoint",
    "longitudeOfLastGridPoint",
    "iDirectionIncrement",
    "jDirectionIncrement",
)
SRC_NLAT, SRC_NLON = 3600, 7200  # 0.05 degree


def overlap(n_src: int, n_dst: int, span: float) -> sparse.csr_matrix:
    """Area weights of ``n_src`` equal cells in ``n_dst`` equal cells over ``span`` degrees."""
    edges_src = np.linspace(0, span, n_src + 1)
    edges_dst = np.linspace(0, span, n_dst + 1)
    lower = np.maximum(edges_dst[:-1, None], edges_src[None, :-1])
    upper = np.minimum(edges_dst[1:, None], edges_src[None, 1:])
    weights = np.clip(upper - lower, 0, None)
    weights /= weights.sum(1, keepdims=True)
    return sparse.csr_matrix(weights.astype("float32"))


class Averager:
    """Area average of a (3600, 7200) field, south to north and longitude -180..180,
    onto the (ny, nx) grid north to south with longitude 0..360."""

    def __init__(self, ny: int, nx: int):
        self.lat = overlap(SRC_NLAT, ny, 180.0)
        self.lon = overlap(SRC_NLON, nx, 360.0)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        field = np.roll(field, SRC_NLON // 2, axis=1)
        out = np.asarray(self.lat @ (self.lon @ field.T).T, dtype="float32")
        return out[::-1]


def fetch(day: int, cache: Path, token: str | None) -> Path:
    """Download one daily file unless it is there; five attempts, then raise."""
    path = cache / f"D{day:03d}.nc"
    if path.exists() and path.stat().st_size > 0:
        return path
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last = None
    for attempt in range(5):
        try:
            with requests.get(
                BASE + NAME.format(day), headers=headers, stream=True, timeout=(30, 300)
            ) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("HTML response instead of a NetCDF file")
                part = path.with_suffix(".part")
                with part.open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
                part.replace(path)
            return path
        except (requests.RequestException, RuntimeError) as error:
            last = error
            print(f"D{day:03d}: attempt {attempt + 1} failed: {error}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"D{day:03d} could not be downloaded: {last}")


def aggregate(cache: Path, ny: int, nx: int, workers: int, keep_daily: bool) -> Path:
    """Download the 365 daily files and write cache/monthly.npz with the monthly sums."""
    token = os.getenv("CEDA_TOKEN")
    average = Averager(ny, nx)
    total = {key: np.zeros((12, ny, nx), "float32") for key in ("sst", "sic")}
    weight = {key: np.zeros((12, ny, nx), "float32") for key in ("sst", "sic")}
    ndays = np.zeros(12, int)
    land_min = land_max = None
    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for count, path in enumerate(
            pool.map(lambda day: fetch(day, cache, token), DAYS), 1
        ):
            with xr.open_dataset(path) as ds:
                mask = ds["mask"].values[0].astype("int16")
                values = {
                    "sst": ds["analysed_sst"].values[0],
                    "sic": ds["sea_ice_fraction"].values[0],
                }
                month = int(ds["time"].dt.month.values[0])
            ocean = (mask & 2) == 0
            share = average((~ocean).astype("float32"))
            land_min = share if land_min is None else np.minimum(land_min, share)
            land_max = share if land_max is None else np.maximum(land_max, share)
            for key, field in values.items():
                valid = np.isfinite(field) & ocean
                total[key][month - 1] += average(
                    np.where(valid, field, 0).astype("float32")
                )
                weight[key][month - 1] += average(valid.astype("float32"))
            ndays[month - 1] += 1
            if not keep_daily:
                path.unlink()
            if count % 20 == 0 or count == len(DAYS):
                print(
                    f"{count}/{len(DAYS)} days, {time.time() - started:.0f} s",
                    flush=True,
                )
    out = cache / "monthly.npz"
    np.savez(
        out,
        version=CACHE_VERSION,
        ndays=ndays,
        land_min=land_min,
        land_max=land_max,
        **{f"total_{k}": v for k, v in total.items()},
        **{f"weight_{k}": v for k, v in weight.items()},
    )
    return out


def nearest_fill(field: np.ndarray) -> np.ndarray:
    """Replace NaN by the nearest valid value."""
    iy, ix = ndimage.distance_transform_edt(
        np.isnan(field), return_distances=False, return_indices=True
    )
    return field[iy, ix]


def squeeze_levels(var: xr.DataArray) -> xr.DataArray:
    """``(time, level, latitude, longitude)`` to ``(time, latitude, longitude)``."""
    return var.isel(
        {d: 0 for d in var.dims if d not in ("time", "latitude", "longitude")}
    )


def stock_index(var: xr.DataArray, lat: np.ndarray, lon: np.ndarray):
    """Rows and columns of the grid ``lat``, ``lon`` nearest to the points of ``var``."""
    rows = np.abs(var["latitude"].values[None, :] - lat[:, None]).argmin(1)
    gap = (var["longitude"].values[None, :] - lon[:, None] + 180.0) % 360.0 - 180.0
    return rows, np.abs(gap).argmin(1)


def skeleton(stock: Path, grid: Path, out: Path, pattern: np.ndarray) -> None:
    """Clone the messages of ``stock`` with the grid of the first message of ``grid``.

    The placeholder ``pattern`` spans the value range of the field, so the packing
    of the stock message (8 bits for sea ice, 1 bit for the mask) is kept.
    """
    with Path(grid).open("rb") as handle:
        template = eccodes.codes_grib_new_from_file(handle)
    keys = {k: eccodes.codes_get(template, k) for k in GRID_KEYS}
    bits = None
    with Path(stock).open("rb") as source, Path(out).open("wb") as target:
        while gid := eccodes.codes_grib_new_from_file(source):
            bits = eccodes.codes_get(gid, "bitsPerValue")
            for key, value in keys.items():
                eccodes.codes_set(gid, key, value)
            eccodes.codes_set_values(gid, pattern)
            if eccodes.codes_get(gid, "bitsPerValue") != bits:
                raise RuntimeError(f"{stock.name}: packing changed in the skeleton")
            eccodes.codes_write(gid, target)
            eccodes.codes_release(gid)
    eccodes.codes_release(template)


def fill_grib(template: Path, out: Path, fields: dict[int, np.ndarray]) -> None:
    """Write ``fields`` (calendar month, or 0 for a single field, to a ``(lat, lon)`` array
    in the scan order of the template) through the messages of ``template``."""
    tree = open_grib(template)
    node = tree.leaves[0]
    name = next(iter(node.data_vars))
    var = node[name]
    if var.sizes["time"] != len(fields):
        raise ValueError(
            f"{template.name}: {var.sizes['time']} fields, {len(fields)} given"
        )
    levels = [d for d in var.dims if d not in ("time", "latitude", "longitude")]
    months = var["time"].dt.month.values if len(fields) > 1 else [0]
    data = np.empty(var.shape, "float32")
    for i, month in enumerate(months):
        data[i] = fields[int(month)].reshape(
            (1,) * len(levels) + fields[int(month)].shape
        )
    node[name] = var.copy(data=data)
    save_grib(tree, out, template)


def build(root: Path, workers: int, keep_daily: bool, force: bool, clean: bool) -> None:
    am = root / "am"
    for stock in (SST_STOCK, ICE_STOCK, MASK_STOCK):
        if not (am / stock).is_file():
            raise FileNotFoundError(
                f"{am / stock} not found; fetch the stock fix files with tools/update_fix.py"
            )
    out_dir = root / EXTERNAL_DIR / ESA_DIR
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    sst_node = open_grib(am / SST_STOCK).leaves[0]
    sst_var = squeeze_levels(sst_node[next(iter(sst_node.data_vars))])
    ny, nx = sst_var.sizes["latitude"], sst_var.sizes["longitude"]
    if sst_var.sizes["time"] != 12 or ny * nx > 5000 * 2500:
        raise ValueError(
            f"{SST_STOCK}: unexpected grid {ny} x {nx} with {sst_var.sizes['time']} months"
        )
    # the grid of the stock SST file, 1/12 degree: centres 89.958 - k/12 north to south, 0.042 + k/12 east
    lat = 89.958 - np.arange(ny) / 12.0
    lon = 0.042 + np.arange(nx) / 12.0
    rows, cols = stock_index(sst_var, lat, lon)
    if len(set(rows)) != ny or len(set(cols)) != nx:
        raise ValueError(f"{SST_STOCK} is not the expected 1/12 degree grid")
    flip = np.ix_(
        rows, cols
    )  # aggregated arrays (north to south, 0..360) to the scan order of the file

    monthly = cache / "monthly.npz"
    if (
        force
        or not monthly.is_file()
        or str(np.load(monthly)["version"]) != CACHE_VERSION
    ):
        monthly = aggregate(cache, ny, nx, workers, keep_daily)
    data = np.load(monthly)
    print("days per month", data["ndays"].tolist())
    if not np.array_equal(
        data["ndays"], [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ):
        raise ValueError("the monthly means do not cover 365 days")
    share = 0.5 * (data["land_min"] + data["land_max"])[flip]
    print(
        f"land share changes between days by at most {float((data['land_max'] - data['land_min']).max()):.3g}"
    )
    land_cci = share >= 0.5

    # lakes of the stock mask that the CCI mask calls land stay water
    mask_node = open_grib(am / MASK_STOCK).leaves[0]
    mask_var = squeeze_levels(mask_node[next(iter(mask_node.data_vars))]).isel(time=0)
    old_land = mask_var.values > 0.5
    labels, _ = ndimage.label(~old_land)
    for r in range(labels.shape[0]):  # join components across the 0/360 seam
        first, last = labels[r, 0], labels[r, -1]
        if first and last and first != last:
            labels[labels == last] = first
    m_rows, m_cols = stock_index(mask_var, lat, lon)
    label_new = labels[m_rows][:, m_cols]
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    world = ids[np.argmax(counts)]
    lakes = [
        i
        for i in ids
        if i != world
        and (label_new == i).any()
        and land_cci[label_new == i].mean() >= LAKE_LAND_SHARE
    ]
    lake = np.isin(label_new, lakes) & land_cci
    land = land_cci & ~lake
    print(
        f"{len(lakes)} lakes kept as water ({int(lake.sum())} cells); land share {land.mean():.4f}"
    )

    ice_node = open_grib(am / ICE_STOCK).leaves[0]
    ice_var = squeeze_levels(ice_node[next(iter(ice_node.data_vars))])
    ice_months = ice_var["time"].dt.month.values
    sst_months = sst_var["time"].dt.month.values
    if sorted(ice_months) != list(range(1, 13)) or sorted(sst_months) != list(
        range(1, 13)
    ):
        raise ValueError("the stock SST and sea-ice files need twelve monthly fields")
    i_rows, i_cols = stock_index(ice_var, lat, lon)

    sst_fields, ice_fields, clipped = {}, {}, 0
    for k, month in enumerate(sst_months):
        m = int(month) - 1
        stock_sst = sst_var.isel(time=k).values.astype("float64")
        ratio = np.where(
            data["weight_sst"][m][flip] > VALID_SHARE,
            data["total_sst"][m][flip] / np.maximum(data["weight_sst"][m][flip], 1e-6),
            np.nan,
        )
        ratio = np.where(lake, stock_sst, ratio)
        ratio = nearest_fill(np.where(land, np.nan, ratio))
        clipped += int(((ratio < SST_MIN_K) | (ratio > SST_MAX_K))[~land].sum())
        sst_fields[int(month)] = np.where(
            land, stock_sst, np.clip(ratio, SST_MIN_K, SST_MAX_K)
        )
    for k, month in enumerate(ice_months):
        m = int(month) - 1
        stock_ice = ice_var.isel(time=k).values
        ratio = np.where(
            data["weight_sic"][m][flip] > VALID_SHARE,
            data["total_sic"][m][flip] / np.maximum(data["weight_sic"][m][flip], 1e-6),
            np.nan,
        )
        ratio = np.where(lake, np.minimum(stock_ice[i_rows][:, i_cols], 1.0), ratio)
        ratio = nearest_fill(np.where(land, np.nan, ratio))
        ice_fields[int(month)] = np.where(land, LAND_FLAG, np.clip(ratio, 0.0, 1.0))
    print(f"ocean SST values limited to {SST_MIN_K}..{SST_MAX_K} K: {clipped}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (SST_OUT, ICE_OUT, MASK_OUT):
        (out_dir / name).unlink(missing_ok=True)
    fill_grib(am / SST_STOCK, out_dir / SST_OUT, sst_fields)
    ice_skeleton, mask_skeleton = (
        cache / "ice.skeleton.grb",
        cache / "mask.skeleton.grb",
    )
    size = ny * nx
    skeleton(
        am / ICE_STOCK, am / SST_STOCK, ice_skeleton, (np.arange(size) % 158) / 100.0
    )
    skeleton(
        am / MASK_STOCK,
        am / SST_STOCK,
        mask_skeleton,
        (np.arange(size) % 2).astype("float64"),
    )
    fill_grib(ice_skeleton, out_dir / ICE_OUT, ice_fields)
    fill_grib(mask_skeleton, out_dir / MASK_OUT, {0: land.astype("float64")})
    ice_skeleton.unlink()
    mask_skeleton.unlink()

    verify(out_dir, sst_fields, ice_fields, land, sst_var, ny, nx)
    if clean:
        for path in cache.iterdir():
            path.unlink()
        cache.rmdir()
    print(f"wrote {SST_OUT}, {ICE_OUT} and {MASK_OUT} to {out_dir}")


def verify(out_dir, sst_fields, ice_fields, land, sst_var, ny, nx) -> None:
    """Read the written files back and check them against the arrays and each other."""
    sst = open_grib(out_dir / SST_OUT).leaves[0]
    ice = open_grib(out_dir / ICE_OUT).leaves[0]
    msk = open_grib(out_dir / MASK_OUT).leaves[0]
    for node, fields, label, tolerance in (
        (sst, sst_fields, "SST", 0.0051),
        (ice, ice_fields, "sea ice", 0.0051),
    ):
        var = squeeze_levels(node[next(iter(node.data_vars))])
        if var.shape != (12, ny, nx):
            raise ValueError(f"{label}: shape {var.shape}")
        worst = max(
            float(np.abs(var.isel(time=k).values - fields[int(m)]).max())
            for k, m in enumerate(var["time"].dt.month.values)
        )
        print(f"{label}: largest difference from the arrays {worst:.4f}")
        if worst > tolerance:
            raise ValueError(
                f"{label}: written values differ from the arrays by {worst}"
            )
    wrote = squeeze_levels(msk[next(iter(msk.data_vars))]).isel(time=0).values > 0.5
    if not np.array_equal(wrote, land):
        raise ValueError("land mask differs from the array")
    sst_ocean = max(
        float(squeeze_levels(sst[next(iter(sst.data_vars))]).values[:, ~land].max()), 0
    )
    ice_var = squeeze_levels(ice[next(iter(ice.data_vars))])
    if not (
        np.allclose(ice_var.values[:, land], LAND_FLAG, atol=0.006)
        and ice_var.values[:, ~land].max() <= 1.0 + 1e-6
    ):
        raise ValueError("sea ice land flags disagree with the mask")
    print(
        f"checks passed: mask, ice flags and SST agree (largest ocean SST {sst_ocean:.2f} K)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="fix_src of the case; output goes to <root>/external/ESA",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="parallel downloads (default 4)"
    )
    parser.add_argument(
        "--keep-daily", action="store_true", help="keep the 365 daily files (15 GB)"
    )
    parser.add_argument(
        "--force", action="store_true", help="download and aggregate again"
    )
    parser.add_argument(
        "--clean", action="store_true", help="remove the cache directory when done"
    )
    args = parser.parse_args()
    build(args.root.resolve(), args.workers, args.keep_daily, args.force, args.clean)


if __name__ == "__main__":
    main()
