"""PAMIP SST and sea-ice boundary conditions for SHiELD."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import ndimage

from plugins.pamip.config import CONTROL, REGISTRY_FILE

log = logging.getLogger("PLUGINS")

SST_FILE = "RTGSST.1982.2012.monthly.clim.grb"
ICE_FILE = "CFSR.SEAICE.1982.2012.monthly.clim.grb"
LAND_FILE = "seaice_newland.grb"
SST_MIN_K, SST_MAX_K = 271.2, 313.0
AISLIM = 0.15
LAND_FLAG = 1.0
LAKE_CELLS = 500

DAYHF = np.array(
    [
        15.5,
        45.0,
        74.5,
        105.0,
        135.5,
        166.0,
        196.5,
        227.5,
        258.0,
        288.5,
        319.0,
        349.5,
        380.5,
    ]
)
TGICE = 271.2
SIHNEW = 0.2
T_MELT = 273.16
ZORL_NEWICE = 1.0


def write_boundary_files(
    *, state, open_grib, save_grib, ll_regrid, fill_nearest
) -> tuple[xr.Dataset, dict[str, str]]:
    """Write SST and sea-ice climatologies of state.pamip_experiment to FIXED/PAMIP.

    Returns the fields and the namsfc entries pointing at the files."""
    experiment = state.pamip_experiment
    baseline = state.pamip_baseline or "pamip"
    fields = state.pamip_fields or "both"
    registry = yaml.safe_load(REGISTRY_FILE.read_text())["experiments"]
    entry, control = registry[experiment], registry[CONTROL]
    family = entry["family"]

    if baseline == "external":
        base_files = {
            kind: Path(state.fix) / "external" / Path(state[key]).name
            for kind, key in (
                ("sst", "external_sst"),
                ("sic", "external_sic"),
                ("land", "external_mask"),
            )
        }
    else:
        am_dir = Path(state.fix_src) / "am"
        base_files = {
            "sst": am_dir / SST_FILE,
            "sic": am_dir / ICE_FILE,
            "land": am_dir / LAND_FILE,
        }

    pamip = {}
    for kind in ("sst", "sic"):
        chosen = entry if fields in ("both", kind) else control
        read = {}
        for role, spec in (("experiment", chosen[kind]), ("control", control[kind])):
            variable, source = spec["variable_id"], spec["source_id"]
            path = (
                Path(state.fix_src)
                / "external"
                / family
                / source
                / variable
                / f"{variable}_input4MIPs_SSTsAndSeaIce_PAMIP_{source}_gn_{spec['period']}.nc"
            )
            if not path.is_file():
                raise FileNotFoundError(f"PAMIP file {path} not found.")
            with xr.open_dataset(
                path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
            ) as ds:
                da = ds[variable].squeeze(drop=True).load()
            da = da.assign_coords(time=da.time.dt.month, lon=da.lon % 360.0)
            da = da.rename(time="month").sortby(["month", "lat", "lon"])
            da = da.where(abs(da) < 1e10)
            read[role] = da.clip(0.0, 1.0) if kind == "sic" else da
        if baseline == "pamip":
            pamip[kind] = read["experiment"]
        else:
            with xr.set_options(arithmetic_join="exact"):
                pamip[kind] = read["experiment"] - read["control"]

    node = open_grib(base_files["land"]).leaves[0]
    land = next(iter(node.data_vars.values())).squeeze(drop=True)
    land = land.rename(latitude="lat", longitude="lon")
    land = land.assign_coords(lon=land.lon % 360.0).sortby(["lat", "lon"]).load()
    labels, count = ndimage.label(land.values < 0.5)
    size = ndimage.sum(labels > 0, labels, np.arange(1, count + 1))
    lake = land.copy(data=np.isin(labels, 1 + np.flatnonzero(size < LAKE_CELLS)))
    land = land.where(~lake, 1.0)
    ocean_pamip = ll_regrid(land, pamip["sic"]) < 0.5
    pamip["sic"] = pamip["sic"].where(ocean_pamip)
    if baseline != "pamip":
        pamip["sst"] = pamip["sst"].where(ocean_pamip)

    poles = [-90.0, *pamip["sst"].lat.values, 90.0]
    pamip = {
        kind: da.pad(lat=1, mode="edge").assign_coords(lat=poles)
        for kind, da in pamip.items()
    }
    land = land.reindex(lat=[-90.0, *land.lat.values, 90.0], method="nearest")
    lake = lake.reindex(lat=land.lat, method="nearest").astype("float32")

    tag = (
        f"{family}_{experiment}"
        if baseline == "pamip"
        else f"{family}_{baseline}_{experiment}"
    )
    if fields != "both":
        tag = f"{tag}_{fields}"
    out_dir = Path(state.fix) / family
    out_dir.mkdir(parents=True, exist_ok=True)
    out, base, names = {}, {}, {}

    for kind, template in (("sst", base_files["sst"]), ("sic", base_files["sic"])):
        tree = open_grib(template)
        node = tree.leaves[0]
        variable = next(iter(node.data_vars))
        native = node[variable]
        base[kind] = (
            native.squeeze(drop=True)
            .assign_coords(month=native.time.dt.month)
            .swap_dims(time="month")
            .rename(latitude="lat", longitude="lon")
            .load()
        )
        grid = base[kind].assign_coords(lon=base[kind].lon % 360.0)
        new = fill_nearest(ll_regrid(pamip[kind], grid)).assign_coords(
            lat=base[kind].lat, lon=base[kind].lon
        )
        ocean = (
            ll_regrid(land, grid).assign_coords(lat=base[kind].lat, lon=base[kind].lon)
            < 0.5
        )
        inland = (
            ll_regrid(lake, grid).assign_coords(lat=base[kind].lat, lon=base[kind].lon)
            > 0.5
        )

        if kind == "sic" and baseline == "pamip":
            out[kind] = new.clip(0.0, 1.0).where(
                ocean & (base[kind] <= LAND_FLAG), base[kind]
            )
        elif kind == "sic":
            out[kind] = (
                (base[kind] + new)
                .clip(0.0, 1.0)
                .where(ocean & (base[kind] <= LAND_FLAG), base[kind])
            )
        elif baseline == "pamip":
            out[kind] = new.where(~inland, base[kind])
        else:
            inside = ocean & (base[kind] >= SST_MIN_K) & (base[kind] <= SST_MAX_K)
            freezing = base[kind].where(inside).min(("lat", "lon"))
            out[kind] = (
                (base[kind] + new).clip(freezing, SST_MAX_K).where(inside, base[kind])
            )

        names[kind] = (SST_FILE if kind == "sst" else ICE_FILE).replace(
            "1982.2012", tag
        )
        node[variable] = native.copy(data=out[kind].values[:, np.newaxis])
        destination = out_dir / names[kind]
        destination.unlink(missing_ok=True)
        save_grib(tree, destination, template)

    if state.restart_no == 0:
        try:
            import cartopy
            import cartopy.crs as ccrs
            import matplotlib

            matplotlib.use("Agg")
            carto = Path(state.fix_src) / "external" / "CARTOPY"
            if carto.is_dir():
                cartopy.config["pre_existing_data_dir"] = carto
                cartopy.config["data_dir"] = carto
            for kind, cmap, label in (
                ("sst", "RdBu_r", "SST difference (K)"),
                ("sic", "RdBu", "Sea-ice fraction difference"),
            ):
                figure = Path(state.run_dir) / f"pamip_{tag}_{kind}_change.png"
                lock = figure.with_suffix(".lock")
                try:
                    if figure.exists():
                        continue
                    lock.touch(exist_ok=False)
                except FileExistsError:
                    continue
                try:
                    diff = (out[kind] - base[kind]).sortby("month")
                    step = max(diff.sizes["lon"] // 720, 1)
                    blocks = diff.coarsen(lat=step, lon=step, boundary="trim")
                    high, low = blocks.max(), blocks.min()
                    diff = high.where(high >= -low, low)
                    diff = diff.assign_coords(
                        lon=(diff.lon + 180.0) % 360.0 - 180.0
                    ).sortby("lon")
                    top = max(float(abs(diff).quantile(0.999)), 1e-3)
                    unit = 10.0 ** np.floor(np.log10(top))
                    limit = np.ceil(top / unit) * unit
                    grid = diff.plot(
                        x="lon",
                        y="lat",
                        col="month",
                        col_wrap=4,
                        transform=ccrs.PlateCarree(),
                        subplot_kws={"projection": ccrs.Robinson()},
                        cmap=cmap,
                        vmin=-limit,
                        vmax=limit,
                        size=2.4,
                        aspect=1.55,
                        rasterized=True,
                        add_colorbar=False,
                    )
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
                    grid.fig.suptitle(
                        f"{names[kind].removesuffix('.monthly.clim.grb')} - "
                        + f"{base_files[kind].name.removesuffix('.monthly.clim.grb')}: "
                        + f"largest change {float(abs(out[kind] - base[kind]).max()):.3g}"
                    )
                    grid.fig.savefig(figure, dpi=300, bbox_inches="tight")
                finally:
                    lock.unlink(missing_ok=True)
        except Exception:
            log.exception("PAMIP change figure failed; continuing")

    files = {
        "fntsfc": f"FIXED/{family}/{names['sst']}",
        "fnaisc": f"FIXED/{family}/{names['sic']}",
    }
    if baseline == "pamip":
        return xr.Dataset({"sst": pamip["sst"], "sic": pamip["sic"]}), files
    sic = out["sic"].where(out["sic"] <= LAND_FLAG)
    sst = out["sst"].reindex(lat=sic.lat, lon=sic.lon, method="nearest")
    return xr.Dataset({"sst": sst.where(sic.notnull()), "sic": sic}), files


def set_cold_start_surface(
    fields: xr.Dataset,
    *,
    state,
    fill_nearest,
    sfc_filename,
    to_fv3_grid,
    load_grid,
    cp,
) -> None:
    """Set SST and sea ice of surface files to start date fields."""
    date = pd.Timestamp(state.init_datetime)
    rjday = date.dayofyear + date.hour / 24.0
    fields = fields.sortby("month")
    cycle = xr.concat([fields, fields.isel(month=[0])], "month")
    cycle = cycle.assign_coords(day=("month", DAYHF)).swap_dims(month="day")
    now = fill_nearest(cycle.interp(day=rjday + 365.0 * (rjday < DAYHF[0])))

    tiles = (
        [1]
        if state.gtype in ("regional_gfdl", "regional_esg")
        else list(range(1, 7 + state.n_nests))
    )
    for tile in tiles:
        filename = sfc_filename(tile)
        path = Path(state.input) / filename
        out = to_fv3_grid(now, load_grid(filename, tile), periodic=True)
        with xr.open_dataset(path, decode_cf=False, engine="netcdf4") as handle:
            ds = handle.load()
        slmsk = ds["slmsk"]
        sst_t, sic_t = (
            slmsk.copy(data=out[v].transpose(*slmsk.dims).values)
            for v in ("sst", "sic")
        )
        valid = (slmsk != 1) & sst_t.notnull() & sic_t.notnull()
        sst_c, sic_c = sst_t.clip(SST_MIN_K, SST_MAX_K), sic_t.clip(0.0, 1.0)
        ice = valid & (sic_c >= AISLIM)
        water = valid & ~ice
        stay, new = ice & (slmsk == 2), ice & (slmsk != 2)
        changed = new | (water & (slmsk == 2))

        fice = xr.where(water, 0.0, xr.where(ice, sic_c, ds["fice"]))
        reset = new | (stay & ~(ds["hice"] > 0.0))
        hice = xr.where(water, 0.0, xr.where(reset, SIHNEW, ds["hice"]))
        tisfc = xr.where(
            water,
            sst_c,
            xr.where(
                new, TGICE, xr.where(stay, ds["tisfc"].clip(max=T_MELT), ds["tisfc"])
            ),
        )
        tsea = xr.where(
            water, sst_c, xr.where(ice, fice * tisfc + (1.0 - fice) * TGICE, ds["tsea"])
        )

        update = {
            "slmsk": xr.where(water, 0.0, xr.where(ice, 2.0, slmsk)),
            "tsea": tsea,
            "fice": fice,
            "hice": hice,
            "tisfc": tisfc,
            "sheleg": ds["sheleg"].where(~changed, 0.0),
            "snwdph": ds["snwdph"].where(~changed, 0.0),
            "zorl": ds["zorl"].where(~changed, ZORL_NEWICE),
            "stc": ds["stc"].where(~changed, TGICE),
        }
        for v, a in update.items():
            a = a.transpose(*ds[v].dims).astype(ds[v].dtype)
            a.attrs = {k: x for k, x in ds[v].attrs.items() if k != "checksum"}
            a.encoding = ds[v].encoding
            ds[v] = a
        backup = Path(state.ic_data) / "perts" / f"{filename.stem}.pamip.original.nc"
        if not backup.exists():
            cp(path, backup)
        ds.to_netcdf(path, engine="netcdf4")
