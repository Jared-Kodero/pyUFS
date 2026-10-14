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
renamed when complete, five attempts per file, into ``/tmp/<user>/esa_sst``, which is
emptied at the start of every run. The daily files are deleted after they have been
averaged.

How the files are built
-----------------------
1. Monthly means. The daily fields (0.05 degree) are averaged per calendar month of the
   365-day year (31, 28, 31, ... days) on their own grid, over the ocean days: land cells
   of the CCI mask (bit 2) are left out, so a cell has the mean of its valid days and
   is NaN where there is none. The twelve monthly means of SST and of sea-ice fraction
   and the land mask (the same on every day) are kept as ``monthly.npz`` (about 2.6 GB)
   in the temporary directory.
   They are then regridded in one call of ``ll_regrid`` of ``src/latlon_to_fv3.py`` (xESMF,
   bilinear, NaN-skipping, periodic in longitude, as for the PAMIP files) onto the 1/12
   degree grid (4320 x 2160) of ``fix_src/am/RTGSST.1982.2012.monthly.clim.grb``.
2. Land mask. A target cell is land where the regridded CCI land mask (1 = land) is at
   least 0.5. The CCI mask counts inland water as land,
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

Difference figures
------------------
Next to the files, ``sst_change_vs_stock.png``, ``sic_change_vs_stock.png`` and
``mask_change_vs_stock.png`` show the new files minus the stock SHiELD files
(``am/``): twelve Robinson panels (xarray facet grid) for SST in K and sea ice as a
fraction, one panel for the land mask. Only cells that are ocean in both files are
compared; the others are white. The colour scale is symmetric about zero and rounded
from the 99.9th percentile; the largest absolute difference is in the title. The
stock sea ice and mask are sampled at the nearest 0.5 degree cell. Grids finer than
720 points in longitude are block-averaged for drawing. Natural Earth coastlines come
from ``<root>/external/CARTOPY`` as in ``fv3_plot_grid.py`` (without them the maps
have no coastlines).

Land values
-----------
The files are not empty over land: GRIB1 simple packing has no missing value and
SHiELD ignores a bitmap for SST and sea ice. Land cells hold the stock SHiELD SST,
the sea-ice land flag 1.57 and 1 in the mask. ``sfcsub.F`` weights the climatology by the
mask and never reads these values at ocean points. The land share and the land values
are printed at the end of a run.

Output
------
``<root>/external/ESA/`` (``<root>`` is ``fix_src`` of the case, as for
``tools/download_pamip_amip.py``)::

    sst.ESACCI_1991_2020.monthly.clim.grb   SST, K, 12 months (format of the stock FNTSFC file)
    sic.ESACCI_1991_2020.monthly.clim.grb   sea ice, 12 months, 1.57 on land (stock FNAISC format)
    landmask.ESACCI.grb                     land mask, 1 = land (stock FNMSKH format)

The stock files must be in ``<root>/am`` (``tools/update_fix.py``). The three outputs
go together: use all of them (FNTSFC, FNAISC and FNMSKH of namsfc), never a mix with
stock files. Memory: the xESMF weights for this grid take about 15 GB, as for the PAMIP
regridding (``README.md``), and the monthly means 3 GB; disk 3 GB in ``/tmp/<user>/esa_sst``. Differences from the stock climatology (1982-2012, RTG SST, CFSR ice) are
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
    python tools/esa_sst.py --root <fix_src> --workers 8
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eccodes
import numpy as np
import requests
import xarray as xr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from grib_io import open_grib, save_grib
from latlon_to_fv3 import ll_regrid

script_dir = Path(__file__).resolve().parent

BASE = (
    "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/"
    "sea_surface_temperature/CDR_v3/Climatology/L4/v3.0.1/"
)
NAME = "D{:03d}-ESACCI-L4_GHRSST-SSTdepth-Climatology-GLOB_CDR3.0-v02.0-fv01.0.nc"
DAYS = range(1, 366)
TAG = "ESACCI_1991_2020"
STOCK = {  # stock SHiELD files in <root>/am
    "sst": "RTGSST.1982.2012.monthly.clim.grb",
    "sic": "CFSR.SEAICE.1982.2012.monthly.clim.grb",
    "mask": "seaice_newland.grb",
}
OUT = {
    "sst": f"sst.{TAG}.monthly.clim.grb",
    "sic": f"sic.{TAG}.monthly.clim.grb",
    "mask": "landmask.ESACCI.grb",
}
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
SST_MIN_K, SST_MAX_K = 271.2, 313.0  # sfcsub.F: SST outside this range is replaced
LAND_FLAG = 1.57  # sea-ice land flag of the stock file
LAKE_LAND_SHARE = 0.9  # share of a stock lake covered by CCI land
MAX_MASK_DIFF = 0.08  # largest share of cells that may differ from the stock mask
PLOT_DPI = 300
PLOT_LON = 720  # longitude points drawn per panel; finer grids are block-averaged
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def download(day: int, cache: Path) -> Path:
    """Fetch one daily file (five attempts); ``CEDA_TOKEN`` is optional."""
    path = cache / f"D{day:03d}.nc"
    token = os.getenv("CEDA_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(5):
        try:
            with requests.get(
                BASE + NAME.format(day), headers=headers, stream=True, timeout=(30, 300)
            ) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("HTML response instead of a NetCDF file")
                with path.with_suffix(".part").open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
                path.with_suffix(".part").replace(path)
            return path
        except (requests.RequestException, RuntimeError) as error:
            print(f"D{day:03d}: attempt {attempt + 1} failed: {error}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"D{day:03d} could not be downloaded")


def monthly_means(cache: Path, workers: int) -> Path:
    """Download the 365 daily files and write ``cache/monthly.npz``.

    It holds, on the 0.05 degree grid of the files, the mean of each calendar month of
    the SST and of the sea-ice fraction over the ocean days (NaN where there is none),
    the land mask, the number of days of each month and the coordinates.
    """
    total = {k: np.zeros((12, 3600, 7200), "float32") for k in ("sst", "sic")}
    count = {k: np.zeros((12, 3600, 7200), "int16") for k in ("sst", "sic")}
    ndays, land, started = np.zeros(12, int), None, time.time()
    with ThreadPoolExecutor(workers) as pool:
        for n, path in enumerate(pool.map(lambda d: download(d, cache), DAYS), 1):
            with xr.open_dataset(path) as ds:
                today = (ds["mask"].values[0].astype("int16") & 2) != 0  # bit 2 is land
                fields = {
                    "sst": ds["analysed_sst"].values[0],
                    "sic": ds["sea_ice_fraction"].values[0],
                }
                month = int(ds["time"].dt.month.values[0]) - 1
                lat, lon = ds["lat"].values, ds["lon"].values
            if land is None:
                land = today
            elif not np.array_equal(land, today):
                raise ValueError(f"{path.name}: the land mask differs from day 1")
            for key, field in fields.items():
                valid = np.isfinite(field) & ~land
                total[key][month] += np.where(valid, field, 0)
                count[key][month] += valid
            ndays[month] += 1
            path.unlink()
            if n % 20 == 0 or n == len(DAYS):
                print(
                    f"{n}/{len(DAYS)} days, {time.time() - started:.0f} s", flush=True
                )
    out = cache / "monthly.npz"
    # in place, to keep the memory low: the mean, NaN where no day is valid
    for key, mean in total.items():
        mean[count[key] == 0] = np.nan
        mean /= np.maximum(count[key], 1)
    np.savez(out, ndays=ndays, land=land, lat=lat, lon=lon, **total)
    return out


def read_grib(path: Path) -> xr.DataArray:
    """The variable of a one-variable GRIB file as ``(time, latitude, longitude)`` (one level)."""
    node = open_grib(path).leaves[0]
    var = node[next(iter(node.data_vars))]
    return var.isel(
        {d: 0 for d in var.dims if d not in ("time", "latitude", "longitude")}
    )


def nearest(have: np.ndarray, want: np.ndarray, period: float = 0.0) -> np.ndarray:
    """Index in ``have`` of the point nearest to each point of ``want`` (``period`` for longitude)."""
    gap = np.asarray(have)[None, :] - np.asarray(want)[:, None]
    if period:
        gap = (gap + period / 2) % period - period / 2
    return np.abs(gap).argmin(1)


def write_grib(
    stock: Path,
    out: Path,
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    grid: Path | None = None,
) -> None:
    """Write ``values`` ``(messages, lat, lon)`` through the messages of the stock file ``stock``.

    ``lat`` (north to south) and ``lon`` (0..360) are the coordinates of ``values``; they
    are put in the order of the file. Headers, packing, month stamps and message order of
    ``stock`` are kept (``grib_io.save_grib``). With ``grid`` the stock messages are first
    cloned with the grid of that file and a placeholder of their own values, which keeps
    their packing (8 bits for sea ice, 1 bit for the mask), as the grid differs from the stock grid.
    """
    out.unlink(missing_ok=True)
    template = stock
    if grid is not None:
        template = out.with_name(out.name + ".skeleton")
        with grid.open("rb") as handle:
            gid = eccodes.codes_grib_new_from_file(handle)
        keys = {k: eccodes.codes_get(gid, k) for k in GRID_KEYS}
        eccodes.codes_release(gid)
        with stock.open("rb") as src, template.open("wb") as dst:
            while gid := eccodes.codes_grib_new_from_file(src):
                bits = eccodes.codes_get(gid, "bitsPerValue")
                pattern = np.resize(eccodes.codes_get_values(gid), len(lat) * len(lon))
                for key, value in keys.items():
                    eccodes.codes_set(gid, key, value)
                eccodes.codes_set_values(gid, pattern)
                if eccodes.codes_get(gid, "bitsPerValue") != bits:
                    raise RuntimeError(f"{stock.name}: packing changed in the skeleton")
                eccodes.codes_write(gid, dst)
                eccodes.codes_release(gid)
    tree = open_grib(template)
    node = tree.leaves[0]
    name = next(iter(node.data_vars))
    var = node[name]
    rows = nearest(lat, var["latitude"].values)
    cols = nearest(lon, np.mod(var["longitude"].values, 360.0), 360.0)
    data = values[:, rows][:, :, cols].astype("float32")
    node[name] = var.copy(data=data.reshape(var.shape))
    save_grib(tree, out, template)
    if grid is not None:
        template.unlink()


def plot_change(
    path: Path,
    diff: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    label: str,
    cmap: str,
    carto: Path,
    limit: float | None = None,
) -> None:
    """Draw ``diff`` ``(panels, lat, lon)`` as an xarray facet grid of Robinson maps.

    NaN is white. The colour scale is symmetric about zero, from ``limit`` or the 99.9th
    percentile rounded up to one figure; the largest absolute value is in the title. Grids
    finer than ``PLOT_LON`` points in longitude are block-averaged for drawing only
    (``lat`` and ``lon`` must be regular and start on a block edge). Natural Earth
    coastlines are read from ``carto`` when it exists; without them the maps are drawn
    without coastlines. A failure is reported and does not stop the script.
    """
    try:
        import cartopy
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"no plots: {error}")
        return
    if carto.is_dir():
        cartopy.config["pre_existing_data_dir"] = carto
        cartopy.config["data_dir"] = carto
    peak = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 0.0
    f = max(len(lon) // PLOT_LON, 1)
    if f > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            diff = np.stack(
                [
                    np.nanmean(
                        d.reshape(len(lat) // f, f, len(lon) // f, f), axis=(1, 3)
                    )
                    for d in diff
                ]
            )
        lat, lon = lat.reshape(-1, f).mean(1), lon.reshape(-1, f).mean(1)
    finite = np.abs(diff[np.isfinite(diff)])
    if limit is None:
        top = float(np.percentile(finite, 99.9)) if finite.size else 1e-3
        unit = 10.0 ** np.floor(np.log10(top))
        limit = float(np.ceil(top / unit) * unit)
    da = (
        xr.DataArray(
            diff,
            dims=("month", "lat", "lon"),
            coords={
                "month": MONTHS if len(diff) == 12 else list(range(len(diff))),
                "lat": lat,
                "lon": (lon + 180.0) % 360.0 - 180.0,
            },
        )
        .sortby("lat")
        .sortby("lon")
    )
    for coast in (True, False):  # the coastline data may be missing offline
        try:
            grid = da.plot(
                x="lon", y="lat", col="month", col_wrap=4 if len(diff) == 12 else None,
                transform=ccrs.PlateCarree(),
                subplot_kws={"projection": ccrs.Robinson(central_longitude=0.0)},
                cmap=cmap, vmin=-limit, vmax=limit, size=2.4, aspect=1.55,
                rasterized=True, add_colorbar=False,
            )  # fmt: skip
            grid.set_titles("{value}", size=10)
            grid.fig.set_layout_engine("constrained")
            grid.fig.colorbar(
                grid.axs.flat[0].collections[0], ax=list(grid.axs.flat),
                orientation="horizontal", label=label, shrink=0.5, aspect=40,
            )  # fmt: skip
            for ax in grid.axs.flat:
                ax.set_global()
                if coast:
                    ax.add_feature(
                        cfeature.COASTLINE, edgecolor="#66655D", linewidth=0.4
                    )
            grid.fig.suptitle(f"{title}. Largest {peak:.3g}", fontsize=12)
            path.unlink(missing_ok=True)
            grid.fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(grid.fig)
            print(f"wrote {path}")
            return
        except Exception as error:
            plt.close("all")
            print(f"{path.name}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir.parent / "fix",
        help="fix_src of the case; output goes to <root>/external/ESA",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="parallel downloads (default 4)"
    )

    args = parser.parse_args()
    root = args.root.resolve()
    am, out_dir = root / "am", root / "external" / "ESA"
    cache = Path(f"/tmp/{getpass.getuser()}/esa_sst")
    # remove the cache directory if it exists
    if cache.exists():
        shutil.rmtree(cache)

    for name in STOCK.values():
        if not (am / name).is_file():
            raise FileNotFoundError(
                f"{am / name} not found; fetch the stock fix files with tools/update_fix.py"
            )
    cache.mkdir(parents=True, exist_ok=True)

    # Work on the grid of the stock SST file: cells 89.958 - k/12 (north to south) and
    # 0.042 + k/12 (east), the order of the aggregated arrays. write_grib puts them in file order.
    stock = {key: read_grib(am / name) for key, name in STOCK.items()}
    ny, nx = stock["sst"].sizes["latitude"], stock["sst"].sizes["longitude"]
    lat, lon = 89.958 - np.arange(ny) / 12.0, 0.042 + np.arange(nx) / 12.0
    file_lat, file_lon = (
        stock["sst"]["latitude"].values,
        np.mod(stock["sst"]["longitude"].values, 360.0),
    )
    if (
        stock["sst"].sizes["time"] != 12
        or ny * nx
        > 5000 * 2500  # limit of the mask and climatology readers of sfcsub.F
        or np.abs(lat[nearest(lat, file_lat)] - file_lat).max() > 1e-3
        or np.abs(lon[nearest(lon, file_lon, 360.0)] - file_lon).max() > 1e-3
    ):
        raise ValueError(f"{STOCK['sst']} is not the expected 1/12 degree grid")

    data = np.load(monthly_means(cache, args.workers))
    if not np.array_equal(
        data["ndays"], [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ):
        raise ValueError("the monthly means do not cover 365 days")

    # one ll_regrid call (one set of xESMF weights) for the 12 SST, the 12 sea-ice means and the land mask
    lon_src = np.mod(data["lon"], 360.0)  # the target longitudes are 0..360
    by_lon = np.argsort(lon_src)
    stacked = np.empty((25, len(data["lat"]), len(lon_src)), "float32")
    for i, key in enumerate(("sst", "sic")):
        stacked[12 * i : 12 * i + 12] = data[key][:, :, by_lon]
    stacked[24] = data["land"][:, by_lon]
    regridded = ll_regrid(
        xr.DataArray(
            stacked,
            dims=("field", "lat", "lon"),
            coords={"lat": data["lat"], "lon": lon_src[by_lon]},
        ),
        xr.Dataset(coords={"lat": lat, "lon": lon}),
    ).values.astype("float32")
    del stacked, data
    means = {"sst": regridded[:12], "sic": regridded[12:24]}

    # Land: the regridded CCI land mask is at least 0.5. The CCI mask counts inland water as land, but the
    # stock mask has lakes as water: a component of the stock ocean that is not the world
    # ocean and that CCI covers with land in at least LAKE_LAND_SHARE of its cells stays water.
    land_cci = regridded[24] >= 0.5
    old = stock["mask"].isel(time=0)
    labels, _ = ndimage.label(old.values <= 0.5)
    for row in labels:  # join components across the 0/360 seam
        if row[0] and row[-1] and row[0] != row[-1]:
            labels[labels == row[-1]] = row[0]
    labels = labels[nearest(old["latitude"].values, lat)][
        :, nearest(old["longitude"].values, lon, 360.0)
    ]
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    lakes = [
        i
        for i in ids
        if i != ids[counts.argmax()] and land_cci[labels == i].mean() >= LAKE_LAND_SHARE
    ]
    lake = np.isin(labels, lakes) & land_cci
    land = land_cci & ~lake
    old_land = (
        old.values[nearest(old["latitude"].values, lat)][
            :, nearest(old["longitude"].values, lon, 360.0)
        ]
        > 0.5
    )
    differ = float((land != old_land).mean())
    print(
        f"{len(lakes)} lakes kept as water ({int(lake.sum())} cells); land {land.mean():.1%}, stock {old_land.mean():.1%}, differing from the stock mask {differ:.1%}"
    )
    if (
        differ > MAX_MASK_DIFF
    ):  # coasts only; more means the arrays are not on the same grid
        raise ValueError(
            "the new land mask does not match the geography of the stock mask"
        )

    # Ocean cells take the monthly mean (nearest valid value where there is none), land cells
    # keep the stock SST and take the sea-ice land flag; lakes keep the stock values.
    old_fields, new_fields = {}, {}
    for key, low, high, land_value in (
        ("sst", SST_MIN_K, SST_MAX_K, None),
        ("sic", 0.0, 1.0, LAND_FLAG),
    ):
        var = stock[key]
        rows, cols = (
            nearest(var["latitude"].values, lat),
            nearest(var["longitude"].values, lon, 360.0),
        )
        old_fields[key] = var.values[:, rows][:, :, cols].astype("float32")
        new_fields[key] = np.empty_like(old_fields[key])
        for k, month in enumerate(var["time"].dt.month.values - 1):
            field = np.where(
                lake, np.minimum(old_fields[key][k], high), means[key][month]
            )
            field[land] = np.nan
            iy, ix = ndimage.distance_transform_edt(
                np.isnan(field), return_distances=False, return_indices=True
            )
            new_fields[key][k] = np.where(
                land,
                old_fields[key][k] if land_value is None else land_value,
                np.clip(field[iy, ix], low, high),
            )

    # write, read back and check against the arrays and the mask
    sst_grid = am / STOCK["sst"]
    for key, values, grid in (
        ("sst", new_fields["sst"], None),
        ("sic", new_fields["sic"], sst_grid),
        ("mask", land[None].astype("float32"), sst_grid),
    ):
        write_grib(am / STOCK[key], out_dir / OUT[key], values, lat, lon, grid)
        var = read_grib(out_dir / OUT[key])
        back = var.values[:, nearest(var["latitude"].values, lat)][
            :, :, nearest(np.mod(var["longitude"].values, 360.0), lon, 360.0)
        ]
        worst = float(np.abs(back - values).max())
        print(
            f"{OUT[key]}: {back.shape[0]} messages, largest difference from the arrays {worst:.4f}"
        )
        if worst > 0.0051:
            raise ValueError(f"{OUT[key]}: written values differ from the arrays")
    print(
        f"over land: SST minus stock SST up to {float(np.abs(new_fields['sst'] - old_fields['sst'])[:, land].max()):.4f} K, "
        f"sea ice {float(new_fields['sic'][:, land].min()):.2f}..{float(new_fields['sic'][:, land].max()):.2f} (land flag); "
        f"ocean SST {float(new_fields['sst'][:, ~land].min()):.2f}..{float(new_fields['sst'][:, ~land].max()):.2f} K"
    )

    # new minus stock, where both are ocean (stock land cells are not SST)
    both_ocean = ~land & ~old_land
    carto = root / "external" / "CARTOPY"
    order = {
        k: np.argsort(stock[k]["time"].dt.month.values) for k in ("sst", "sic")
    }  # calendar order
    sea_ice_ok = both_ocean & (old_fields["sic"] <= 1.0).all(0)
    for key, ok, title, label, cmap in (
        (
            "sst",
            both_ocean,
            "ESA CCI minus stock SHiELD SST",
            "SST change (K)",
            "RdBu_r",
        ),
        (
            "sic",
            sea_ice_ok,
            "ESA CCI minus stock SHiELD sea ice",
            "Sea-ice concentration change (fraction)",
            "RdBu",
        ),
    ):
        diff = np.where(ok, new_fields[key] - old_fields[key], np.nan)[order[key]]
        plot_change(
            out_dir / f"{key}_change_vs_stock.png",
            diff,
            lat,
            lon,
            title,
            label,
            cmap,
            carto,
        )
    plot_change(
        out_dir / "mask_change_vs_stock.png",
        (land.astype("float32") - old_land)[None], lat, lon,
        "ESA CCI minus stock SHiELD land mask",
        "Land mask change (+1 land only in the new mask, -1 land only in the stock mask)",
        "RdBu_r", carto, limit=1.0,
    )  # fmt: skip
    print(f"wrote {', '.join(OUT.values())} to {out_dir}")
    # eccodes and cartopy (GEOS, PROJ) can abort together when the interpreter shuts down
    # ("double free or corruption") after all work is done: leave directly
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
