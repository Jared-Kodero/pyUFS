"""Build SHiELD SST, sea-ice and land-mask climatologies from the ESA SST CCI v3.0 climatology.

Product: ESA SST CCI Climatology v3.0 (Embury et al., 2024, doi:10.1038/s41597-024-03147-w), the 1991-2020 daily
climatology of the SST CCI L4 analysis (20 cm depth SST): 365 files on a 0.05 degree grid with ``analysed_sst``,
``sea_ice_fraction`` and ``mask`` (bit 2 = land, lakes included). The files are public (about 41 MB each, 15 GB in all)
and are deleted after they have been averaged.

Steps
-----
1. Monthly means over the days of each month of the 365-day year (31, 28, 31, ...), then ``ll_regrid`` (xESMF,
   bilinear, NaN-skipping) of SST, sea ice and land share onto the 1/12 degree grid of the SHiELD SST file.
   Land is where the regridded share is at least 0.5. Memory: the xESMF weights take about 15 GB.
2. Lakes. The SHiELD mask has lakes as water, the CCI as land. A SHiELD water body that the CCI covers with land in
   at least 90 % of its cells is a lake and stays water: ERA5-Land skin temperature (limited to the SST range of
   ``sfcsub.F``, SHiELD SST where ERA5-Land has none) and the SHiELD ice. The SHiELD ice marks the Great Lakes with the land flag 1.57; there
   a lake at freezing (below 273.15 K) is ice covered and any other is ice free.
3. SST. Water takes the CCI value, limited to 271.2..313.0 K (``sfcsub.F``). GRIB1 has no missing value and SHiELD
   ignores a bitmap, so land is filled: land within 3 cells of water takes the nearest water value, so that the
   coastlines never mix a land value into an ocean point, and the rest takes the 1991-2020 monthly ERA5-Land skin
   temperature (limited to 173..353 K, the bare-land limits of ``sfcsub.F``), downloaded once by ``get_era5.py``.
4. Sea ice. Water takes the CCI fraction (0..1), land the flag 1.57 of the SHiELD file, which ``sfcsub.F`` needs to find
   the land of the ice file. The mask is 1 over land.
5. GRIB. The files use the grid of the SHiELD SST file, below the 5000 x 2500 limit of the readers of ``sfcsub.F``.
   Each is written through the messages of a SHiELD file (``grib_io.save_grib``), so headers, packing and month
   stamps stay. Sea ice and mask are first cloned onto the 1/12 degree grid.

Output, in ../fix/external/ESA (the SHiELD files are read from ../fix/am), with the difference figures
``*_change_vs_shield.png`` (ESA_SST minus RTGSST, ESA_SST minus CFSR.SEAICE, Robinson panels)::

    sst.ESACCI_1991_2020.monthly.clim.grb   sic.ESACCI_1991_2020.monthly.clim.grb   landmask.ESACCI.grb

The three go together (namsfc FNTSFC, FNAISC, FNMSKH), never mixed with SHiELD files. In ``run_config.yaml``::

    external_sst: external/ESA/sst.ESACCI_1991_2020.monthly.clim.grb
    external_sic: external/ESA/sic.ESACCI_1991_2020.monthly.clim.grb
    external_mask: external/ESA/landmask.ESACCI.grb

Usage: ``python tools/esa_sst.py`` (no options). Differences from the SHiELD set (RTG SST, CFSR ice, 1982-2012) are
differences of data set and period, not only of climate.
"""

import logging
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eccodes
import numpy as np
import xarray as xr
from get_era5 import get_era5
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from grib_io import open_grib, save_grib
from latlon_to_fv3 import fill_nearest, ll_regrid

log = logging.getLogger(__name__)

FIX = Path(__file__).resolve().parent.parent / "fix"
OUT = FIX / "external" / "ESA"
URL = (
    "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/"
    "CDR_v3/Climatology/L4/v3.0.1/D{:03d}-ESACCI-L4_GHRSST-SSTdepth-Climatology-GLOB_CDR3.0-v02.0-fv01.0.nc"
)
SHIELD = {
    "sst": FIX / "am" / "RTGSST.1982.2012.monthly.clim.grb",
    "sic": FIX / "am" / "CFSR.SEAICE.1982.2012.monthly.clim.grb",
    "mask": FIX / "am" / "seaice_newland.grb",
}
SST_RANGE, LAND_RANGE = (271.2, 313.0), (173.0, 353.0)  # sfcsub.F limits
BUFFER, LAND_FLAG, LAKE_SHARE, FREEZING = 3, 1.57, 0.9, 273.15
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
MONTH_OF_DAY = np.repeat(
    np.arange(1, 13), [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
)


def read(path: Path) -> xr.DataArray:
    """A SHiELD GRIB file as ``(month, lat, lon)``."""
    log.info("Reading SHiELD GRIB: %s", path)
    node = open_grib(path).leaves[0]
    da = next(iter(node.data_vars.values()))
    da = da.isel({da.dims[1]: 0}, drop=True)
    result = (
        da.assign_coords(month=da.time.dt.month)
        .swap_dims(time="month")
        .rename(latitude="lat", longitude="lon")
        .load()
    )
    log.info("Loaded SHiELD GRIB: %s", path)
    return result


def download(day: int, folder: Path) -> Path:
    file = folder / f"D{day:03d}.nc"
    for attempt in range(1, 6):
        try:
            urllib.request.urlretrieve(URL.format(day), f"{file}.part")
            return Path(f"{file}.part").replace(file)
        except OSError as exc:
            log.warning("Download D%03d failed (%d/5): %s", day, attempt, exc)
            time.sleep(5)
    raise RuntimeError(f"D{day:03d} could not be downloaded")


def monthly_means() -> xr.Dataset:
    """SST and sea-ice fraction as ``(month, lat, lon)`` on the 0.05 degree grid, and the land mask."""
    log.info("Building ESA SST CCI monthly climatologies")
    months = []
    with tempfile.TemporaryDirectory(dir=OUT) as folder, ThreadPoolExecutor(4) as pool:
        for month in range(1, 13):
            days = np.flatnonzero(MONTH_OF_DAY == month) + 1
            log.info("Month %02d/12: downloading %d daily files", month, len(days))
            files = []
            for i, file in enumerate(
                pool.map(lambda day: download(day, Path(folder)), days), 1
            ):
                files.append(file)
                log.info(
                    "Month %02d/12: downloaded %d/%d (%s)",
                    month,
                    i,
                    len(days),
                    file.name,
                )

            log.info("Month %02d/12: computing monthly means (chunks=auto)", month)
            with xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim="time",
                data_vars="all",
                chunks="auto",
            ) as ds:
                months.append(
                    ds[["analysed_sst", "sea_ice_fraction"]].mean("time").compute()
                )
                if month == 1:
                    log.info("Extracting ESA land mask")
                    land = (
                        ((ds["mask"].isel(time=0).astype("int16") & 2) > 0)
                        .astype("float32")
                        .compute()
                    )
            log.info("Month %02d/12: mean computed; removing daily files", month)
            for file in files:
                file.unlink()
    log.info("Combining 12 monthly means")
    ds = xr.concat(months, dim=xr.Variable("month", np.arange(1, 13)))
    return ds.rename(analysed_sst="sst", sea_ice_fraction="sic").assign(land=land)


def write(values: xr.DataArray, template: Path, out: Path) -> None:
    """Write ``(month, lat, lon)`` values on the SHiELD SST grid through the messages of ``template``."""
    log.info("Writing GRIB %s: preparing reference grid", out.name)
    with SHIELD["sst"].open("rb") as handle:
        gid = eccodes.codes_grib_new_from_file(handle)
        grid = {key: eccodes.codes_get(gid, key) for key in GRID_KEYS}
        eccodes.codes_release(gid)
    skeleton = out.with_suffix(
        ".skeleton"
    )  # the messages of the template on the grid of the SST file
    with template.open("rb") as src, skeleton.open("wb") as dst:
        while gid := eccodes.codes_grib_new_from_file(src):
            for key, value in grid.items():
                eccodes.codes_set(gid, key, value)
            eccodes.codes_set_values(
                gid,
                np.resize(
                    eccodes.codes_get_values(gid), values.shape[-2] * values.shape[-1]
                ),
            )
            eccodes.codes_write(gid, dst)
            eccodes.codes_release(gid)
    log.info("Writing GRIB %s: populating monthly fields", out.name)
    tree = open_grib(skeleton)
    node = tree.leaves[0]
    name = next(iter(node.data_vars))
    native = node[name]
    data = (
        values.transpose("month", "lat", "lon")
        .sel(month=native.time.dt.month.values)
        .values
    )
    node[name] = native.copy(data=data.reshape(native.shape).astype("float32"))
    out.unlink(missing_ok=True)
    save_grib(tree, out, skeleton)
    skeleton.unlink()
    log.info("Saved GRIB: %s", out)


def plot(diff: xr.DataArray, path: Path, title: str, label: str, cmap: str) -> None:
    """Robinson panels of ``(month, lat, lon)`` differences, NaN white, symmetric colour scale (99.9th percentile).

    ``title`` names the two fields, ``new - old``. Blocks of cells are reduced to their value of largest
    magnitude, so single-cell differences (coastline, lakes, ice flag) are not averaged away and the
    largest change in the title is that of the full grid.
    """
    log.info("Plotting differences: %s", path.name)
    import cartopy
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    cartopy.config["data_dir"] = (
        FIX / "external" / "CARTOPY"
    )  # Natural Earth cache of fv3_plot_grid.py
    step = max(diff.sizes["lon"] // 720, 1)
    blocks = diff.coarsen(lat=step, lon=step, boundary="trim")
    high, low = blocks.max(), blocks.min()
    diff = high.where(high >= -low, low)
    diff = diff.assign_coords(lon=(diff.lon + 180.0) % 360.0 - 180.0).sortby("lon")
    top = max(float(abs(diff).quantile(0.999)), 1e-3)
    unit = 10.0 ** np.floor(np.log10(top))
    limit = np.ceil(top / unit) * unit
    grid = diff.plot(
        x="lon", y="lat", col="month", col_wrap=min(4, diff.sizes["month"]),
        transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.Robinson()},
        cmap=cmap, vmin=-limit, vmax=limit, size=2.4, aspect=1.55, rasterized=True, add_colorbar=False,
    )  # fmt: skip
    for ax in grid.axs.flat:
        ax.coastlines(linewidth=0.4)
    grid.fig.set_layout_engine("constrained")
    grid.fig.colorbar(
        grid.axs.flat[0].collections[0],
        ax=list(grid.axs.flat),
        orientation="horizontal",
        label=label,
        shrink=0.5,
    )
    grid.fig.suptitle(f"{title}: largest change {float(abs(diff).max()):.3g}")
    grid.fig.savefig(path, dpi=300, bbox_inches="tight")
    log.info("Saved plot: %s", path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log.info("Preparing ERA5-Land skin temperature climatology")
    skin = (
        xr.open_dataset(get_era5()["skin"])["skt"]
        .rename(latitude="lat", longitude="lon")
        .sortby("lat")
    )
    log.info("Loading SHiELD SST, sea-ice and land-mask fields")
    shield_sst, shield_sic = read(SHIELD["sst"]), read(SHIELD["sic"])
    shield_land = read(SHIELD["mask"]).isel(month=0, drop=True) > 0.5
    grid = shield_sst.assign_coords(lon=shield_sst.lon % 360.0)
    esa = monthly_means()
    log.info("Monthly ESA SST, sea-ice and land-mask fields ready")

    def regrid(da: xr.DataArray) -> xr.DataArray:
        log.info("Regridding %s to the SHiELD SST grid", da.name)
        result = ll_regrid(da, grid).assign_coords(
            lat=shield_sst.lat, lon=shield_sst.lon
        )
        log.info("Regridded %s", da.name)
        return result

    def on_grid(
        da: xr.DataArray,
    ) -> xr.DataArray:  # a 0.5 degree SHiELD field on the 1/12 degree grid
        return da.reindex(lat=shield_sst.lat, lon=shield_sst.lon, method="nearest")

    sst, sic, share, skin = (
        regrid(da) for da in (esa["sst"], esa["sic"], esa["land"], skin)
    )
    log.info("Applying ESA land threshold")
    land = share >= 0.5

    log.info("Identifying lakes using the SHiELD mask")
    # SHiELD water bodies that the CCI covers with land are lakes
    coarse = share.reindex(lat=shield_land.lat, lon=shield_land.lon, method="nearest")
    labels, count = ndimage.label(~shield_land.values)
    covered = ndimage.mean(coarse.values, labels, np.arange(1, count + 1)) >= LAKE_SHARE
    lake = (
        on_grid(shield_land.copy(data=np.isin(labels, 1 + np.flatnonzero(covered))))
        & land
    )
    water = ~land | lake

    log.info("Filling and constraining SST over ocean, lakes and land")
    lake_sst = skin.clip(*SST_RANGE).fillna(shield_sst)
    sst = sst.where(~lake, lake_sst).where(water)
    sst = (
        fill_nearest(sst, BUFFER)
        .fillna(fill_nearest(sst).where(water))
        .fillna(fill_nearest(skin))
    )
    sst = xr.where(water, sst.clip(*SST_RANGE), sst.clip(*LAND_RANGE))
    log.info("Constructing sea-ice fractions and land mask")
    # SHiELD lake ice; the Great Lakes carry the land flag there, so ice is where the lake is at freezing
    lake_ice = on_grid(shield_sic)
    lake_ice = lake_ice.where(lake_ice <= 1.0, (lake_sst < FREEZING).astype("float32"))
    sic = sic.where(~lake, lake_ice).where(water)
    sic = xr.where(water, fill_nearest(sic).clip(0.0, 1.0), LAND_FLAG)
    mask = (~water).astype("float32")

    log.info("Writing SST, sea-ice and land-mask GRIB outputs")
    write(sst, SHIELD["sst"], OUT / "sst.ESACCI_1991_2020.monthly.clim.grb")
    write(sic, SHIELD["sic"], OUT / "sic.ESACCI_1991_2020.monthly.clim.grb")
    write(mask.expand_dims(month=[1]), SHIELD["mask"], OUT / "landmask.ESACCI.grb")

    log.info("Preparing difference fields relative to SHiELD data")
    both_ocean = water & on_grid(~shield_land)
    both_land = ~water & on_grid(shield_land)
    plot(
        (sst - shield_sst).where(both_ocean),
        OUT / "sst_change_vs_shield.png",
        "ESA_SST - RTGSST (ocean)",
        "SST difference (K)",
        "RdBu_r",
    )
    plot(
        (sst - shield_sst).where(both_land),
        OUT / "sst_land_change_vs_shield.png",
        "ESA_SST.ERA5_LAND - RTGSST (land)",
        "Land SST difference (K)",
        "RdBu_r",
    )
    plot(
        (sic - on_grid(shield_sic)).where(both_ocean),
        OUT / "sic_change_vs_shield.png",
        "ESA_SST - CFSR.SEAICE",
        "Sea-ice fraction difference",
        "RdBu",
    )
    plot(
        (mask - on_grid(shield_land.astype("float32"))).expand_dims(month=[1]),
        OUT / "mask_change_vs_shield.png",
        "ESA_SST land mask - seaice_newland (1 = land)",
        "Land mask difference",
        "RdBu_r",
    )
    log.info("ESA SST climatology processing completed")
    sys.stdout.flush()
    logging.shutdown()
    os._exit(0)  # eccodes and cartopy can abort together at interpreter shutdown


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
