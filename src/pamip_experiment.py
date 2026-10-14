"""PAMIP SST and sea-ice boundary conditions for SHiELD.

The monthly SST and sea-ice climatologies read by the surface cycle (namsfc
FNTSFC and FNAISC) are replaced by the official PAMIP boundary conditions
(Smith et al., 2019, doi:10.5194/gmd-12-1139-2019, distributed through
input4MIPs). Each experiment is a pair of absolute monthly climatologies (``tos``
and ``siconc``) that already holds the SST adjustment at the changed ice edge of
the protocol, so nothing is added to them: experiments differ only through the
files they use, the control (1.1) included.

The PAMIP fields are regridded with ``latlon_to_fv3.ll_regrid`` onto the grids of
the native GRIB files and written through the native messages
(``grib_io.save_grib``), so headers, packing, month stamps and the land flags of
the sea-ice file stay as they are. Only ocean points of the SHiELD land mask
(``seaice_newland.grb``) change; land points keep the native values, as
``sfcsub.F`` weights the climatologies by that mask and never reads land values
at ocean points. At the cold start the same fields,
interpolated to the start date, are also written to the surface files of every
tile. As with ``sm_perturbations``, the module is driven only by
``run_config.yaml``::

    pamip_experiment: "1.6"
    pamip_baseline: pamip   # pamip (default) or native
    pamip_fields: both      # both (default), sst or sic

With ``pamip_baseline: native`` the PAMIP files are not used as they are: the
perturbation of the experiment (experiment minus the control 1.1) is added to the
native SHiELD climatologies, so the control of that baseline is experiment 1.1
(no perturbation, the native fields with the surface-cycle settings of the
PAMIP runs). ``pamip_fields`` takes only the SST or only the sea ice from the
experiment; the other field is the control (PAMIP 1.1, or the native field with
the native baseline). The protocol pairs hold the ice-edge SST adjustment, which
``sst`` or ``sic`` alone leaves out.

The experiment table is ``templates/pamip_experiments.yaml``; the NetCDF files
are read from ``<fix_src>/external/PAMIP/<source_id>/<variable_id>/``
(``tools/download_pamip_amip.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.ndimage import distance_transform_edt

from fv3_runtime import log
from fv3_state import state
from fv3_utils import cp
from grib_io import open_grib, save_grib
from latlon_to_fv3 import ll_regrid, load_grid, sfc_filename, to_fv3_grid

SST_FILE = "RTGSST.1982.2012.monthly.clim.grb"
ICE_FILE = "CFSR.SEAICE.1982.2012.monthly.clim.grb"
LAND_FILE = "seaice_newland.grb"  # 0.5 degree land mask, 1 = land; only read
REGISTRY_FILE = (
    Path(__file__).resolve().parent.parent / "templates" / "pamip_experiments.yaml"
)
CONTROL = "1.1"  # the perturbation of the native baseline is experiment minus CONTROL
BASELINES = ("pamip", "native")
FIELDS = ("both", "sst", "sic")

# sfcsub.F: SST outside 271.2..313.0 K over open ocean is replaced by the surface
# cycle, aislim = 0.15 is the sea-ice threshold, sea-ice values above 1 are land.
SST_MIN_K, SST_MAX_K = 271.2, 313.0
AISLIM = 0.15
LAND_FLAG = 1.0

PLOT_DPI = 300  # twelve global panels per change figure
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
GRIB_TOLERANCE = 0.0051  # half the 0.01 quantum of the native files

# Surface cycle (sfcsub.F): mid-month julian days of the monthly interpolation (dayhf),
# freezing temperature of sea water (tgice), thickness of new ice (sihnew).
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
# newice of sfcsub.F at points that change between ice and open water: snow (snoomx, snoimx)
# 0, roughness (zoromx, zorimx) 1 cm, soil (sea-ice layer) temperature TGICE
ZORL_NEWICE = 1.0

# PAMIP experiments of Smith et al. (2019, GMD 12, 1139) and the protocol page
# https://www.cesm.ucar.edu/projects/cmip6/pamip/protocol. "name" is the short name and "namelist" the
# SHiELD namelist settings of the experiment, by namelist group, merged into the model namelist.
# Experiments 1 to 4 and 6 use the radiative forcing of the year 2000 (Appendix B): ictm = yyyy0 makes
# SHiELD read the CO2 table of year yyyy (radiation_gases.f). Set 5 has time-varying forcing and keeps
# the model default. Protocol (tier, start, length, spin-up) is described in templates/pamip_experiments.yaml.
# Only experiments with SST and sea-ice files in that file can be run.
FORCING_2000 = {"gfs_physics_nml": {"ictm": 20000}}
FORCING_VARYING: dict[str, dict] = {}
PAMIP_EXPERIMENTS: dict[str, dict] = {
    # Group 1: Atmosphere-only time-slice experiments
    "1.1": {"name": "pdSST-pdSIC", "namelist": FORCING_2000},
    "1.2": {"name": "piSST-piSIC", "namelist": FORCING_2000},
    "1.3": {"name": "piSST-pdSIC", "namelist": FORCING_2000},
    "1.4": {"name": "futSST-pdSIC", "namelist": FORCING_2000},
    "1.5": {"name": "pdSST-piArcSIC", "namelist": FORCING_2000},
    "1.6": {"name": "pdSST-futArcSIC", "namelist": FORCING_2000},
    "1.7": {"name": "pdSST-piAntSIC", "namelist": FORCING_2000},
    "1.8": {"name": "pdSST-futAntSIC", "namelist": FORCING_2000},
    "1.9": {"name": "pdSST-pdSICSIT", "namelist": FORCING_2000},
    "1.10": {"name": "pdSST-futArcSICSIT", "namelist": FORCING_2000},
    # Group 2: Partially coupled ocean-atmosphere time-slice experiments
    "2.1": {"name": "pa-pdSIC", "namelist": FORCING_2000},
    "2.2": {"name": "pa-piArcSIC", "namelist": FORCING_2000},
    "2.3": {"name": "pa-futArcSIC", "namelist": FORCING_2000},
    "2.4": {"name": "pa-piAntSIC", "namelist": FORCING_2000},
    "2.5": {"name": "pa-futAntSIC", "namelist": FORCING_2000},
    # Group 3: Regional sea ice forcing experiments
    "3.1": {"name": "pdSST-futOkhotskSIC", "namelist": FORCING_2000},
    "3.2": {"name": "pdSST-futBKSeasSIC", "namelist": FORCING_2000},
    # Group 4: Background SST state experiments
    "4.1": {"name": "modelSST-pdSIC", "namelist": FORCING_2000},
    "4.2": {"name": "modelSST-futArcSIC", "namelist": FORCING_2000},
    # Group 5: Atmosphere-only transient experiments (1979-2014)
    "5.1": {"name": "amip-climSST", "namelist": FORCING_VARYING},
    "5.2": {"name": "amip-climSIC", "namelist": FORCING_VARYING},
    # Group 6: Extended partially coupled ocean-atmosphere experiments
    "6.1": {"name": "pa-pdSIC-ext", "namelist": FORCING_2000},
    "6.2": {"name": "pa-futArcSIC-ext", "namelist": FORCING_2000},
    "6.3": {"name": "pa-futAntSIC-ext", "namelist": FORCING_2000},
}


def validate_pamip_experiment(experiment: object) -> str | None:
    """Validate the ``pamip_experiment`` option before any file is written.

    Returns the experiment number, or None when the option is not set. The number
    must be quoted text such as ``"1.6"``, because YAML reads ``1.10`` as the
    float 1.1. Raises TypeError for other types and ValueError for an experiment
    that is not a PAMIP experiment or has no SST and sea-ice files in
    ``templates/pamip_experiments.yaml``, and for ``pamip_baseline`` or
    ``pamip_fields`` outside their values or set without an experiment.
    """
    baseline, fields = state.pamip_baseline, state.pamip_fields
    if experiment is None or experiment == "":
        if baseline is not None or fields is not None:
            raise ValueError(
                "`pamip_baseline` and `pamip_fields` need `pamip_experiment`"
            )
        return None
    if baseline is not None and baseline not in BASELINES:
        raise ValueError(
            f"`pamip_baseline` must be one of {BASELINES}. Got {baseline!r}"
        )
    if fields is not None and fields not in FIELDS:
        raise ValueError(f"`pamip_fields` must be one of {FIELDS}. Got {fields!r}")
    if not isinstance(experiment, str):
        raise TypeError(
            f'`pamip_experiment` must be quoted text such as "1.6". Got {experiment!r}'
        )
    experiment = experiment.strip()
    if experiment not in PAMIP_EXPERIMENTS:
        raise ValueError(
            f"`pamip_experiment` must be one of {tuple(PAMIP_EXPERIMENTS)}. Got `{experiment}`"
        )
    experiments = yaml.safe_load(REGISTRY_FILE.read_text())["experiments"]
    if experiment not in experiments:
        raise ValueError(
            f"PAMIP experiment {experiment} ({PAMIP_EXPERIMENTS[experiment]['name']}) has no SST and sea-ice boundary files in "
            f"{REGISTRY_FILE.name}. Supported: {tuple(experiments)}"
        )
    return experiment


def pamip_namelist() -> dict[str, dict]:
    """SHiELD namelist settings of the configured experiment by namelist group, empty when none is set."""
    experiment = validate_pamip_experiment(state.pamip_experiment)
    return {} if experiment is None else PAMIP_EXPERIMENTS[experiment]["namelist"]


def read_pamip(root: Path, spec: dict, kind: str) -> xr.DataArray:
    """One PAMIP climatology as ``(month, lat, lon)``: SST in kelvin or sea ice as a fraction.

    Latitude ascends, longitude is 0..360 ascending, filled or missing values are NaN.
    """
    variable = spec["variable_id"]
    path = (
        Path(root)
        / spec["source_id"]
        / variable
        / f"{variable}_input4MIPs_SSTsAndSeaIce_PAMIP_{spec['source_id']}_gn_{spec['period']}.nc"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download it with tools/download_pamip_amip.py"
        )
    with xr.open_dataset(
        path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
    ) as ds:
        if ds.attrs.get("source_id") != spec["source_id"]:
            raise ValueError(
                f"{path.name}: source_id is `{ds.attrs.get('source_id')}`, expected `{spec['source_id']}`"
            )
        da = ds[variable].squeeze(drop=True).load()
    months = da["time"].dt.month.values
    if sorted(months) != list(range(1, 13)):
        raise ValueError(
            f"{path.name}: a 12-month climatology is required. Got months {sorted(months)}"
        )
    da = da.assign_coords(time=months).rename(time="month")
    da = (
        da.assign_coords(lon=np.mod(da["lon"], 360.0))
        .sortby("month")
        .sortby("lon")
        .sortby("lat")
    )
    da = da.where(np.isfinite(da) & (np.abs(da) < 1e10)).rename(kind)
    low, high = float(da.min()), float(da.max())
    if kind == "sst" and not (
        da.attrs.get("units") == "K" and 250.0 <= low and high <= 320.0
    ):
        raise ValueError(
            f"{path.name}: SST must be in K and plausible. Got units `{da.attrs.get('units')}`, {low:.1f}..{high:.1f}"
        )
    if kind == "sic" and not (
        da.attrs.get("units") in ("1", "", "fraction")
        and -1e-3 <= low
        and high <= 1.0 + 1e-3
    ):
        raise ValueError(
            f"{path.name}: sea ice must be a fraction. Got units `{da.attrs.get('units')}`, {low:.3f}..{high:.3f}"
        )
    log.info(f"Using PAMIP {kind} file: {path}")
    return da.clip(0.0, 1.0) if kind == "sic" else da


def combine_month(
    old: np.ndarray,
    values: np.ndarray,
    ocean: np.ndarray,
    kind: str,
    perturbation: bool,
) -> np.ndarray:
    """One month of the file to write, ``old`` being the native field and ``values`` the regridded PAMIP field.

    Only ocean points of the SHiELD land mask change; land points keep the native
    values, which the surface cycle never reads at ocean points (sfcsub.F weights
    the source by the land mask). ``values`` may hold NaN where no ocean source
    point is near: these take the nearest valid value.
    """
    iy, ix = distance_transform_edt(
        np.isnan(values), return_distances=False, return_indices=True
    )
    values = values[iy, ix]
    if kind == "sst":
        if perturbation:
            inside = ocean & (old >= SST_MIN_K) & (old <= SST_MAX_K)
            # native freezing value of the month
            freezing = np.nanmin(np.where(inside, old, np.nan))
            new = np.clip(old + values, freezing, SST_MAX_K)
        else:
            inside, new = ocean, values
        return np.where(np.isnan(old) | ~inside, old, new)
    new = np.clip(old + values if perturbation else values, 0.0, 1.0)
    return np.where(np.isnan(old) | ~ocean | (old > LAND_FLAG), old, new)


def write_pamip_grib(
    template: Path,
    destination: Path,
    field: xr.DataArray,
    kind: str,
    land: xr.DataArray,
    perturbation: bool = False,
) -> xr.DataArray:
    """Regrid a climatology onto the native GRIB grid and write it through the messages of the native file.

    ``land`` is the 0.5 degree land mask (1 = land) of SHiELD, regridded here to
    the template grid and thresholded at 0.5 as in sfcsub.F. Only ocean points
    change; land points keep the native values (including the 1.57 sea-ice land
    flags), which the surface cycle ignores at ocean points. SST replaces the
    native ocean value; sea ice is clipped to 0..1 and the points above 1 keep
    their value. With ``perturbation`` the field is a perturbation added to the
    native values: sea ice is clipped to 0..1, and SST is changed only where the
    native value lies in the SST_MIN_K..SST_MAX_K range of the surface cycle and
    is kept between the native freezing value of the month and SST_MAX_K.
    Returns the values written and the stock values of the template, both
    ``(month, lat, lon)`` in the month order of the template.
    """
    tree = open_grib(template)
    node = tree.leaves[0]
    name = next(iter(node.data_vars))
    native = node[name]
    months = native.time.dt.month.values
    if native.shape[:2] != (12, 1) or sorted(months) != list(range(1, 13)):
        raise ValueError(
            f"{template.name}: expected twelve monthly fields on one level"
        )
    grid = xr.Dataset(
        coords={
            "lat": ("lat", native.latitude.values),
            "lon": ("lon", np.mod(native.longitude.values, 360.0)),
        }
    )
    data = native.values.copy()
    stock = native.values  # the stock SHiELD values, kept for the change plots
    # One regridder for the twelve months: xESMF builds the weights of the 0.083 degree
    # SST grid (9.3 million points) in about 3 minutes and 15 GB.
    field = field.transpose("month", "lat", "lon")  # contiguous, as xESMF expects
    regridded = ll_regrid(field.copy(data=np.ascontiguousarray(field.values)), grid)
    ocean = (
        ll_regrid(land.copy(data=np.ascontiguousarray(land.values)), grid).values < 0.5
    )
    for i, month in enumerate(months):
        data[i, 0] = combine_month(
            data[i, 0],
            regridded.sel(month=int(month)).values,
            ocean,
            kind,
            perturbation,
        )
    if kind == "sst" and not perturbation:
        written = data[:, 0][:, ocean]
        if (
            np.nanmin(written) < SST_MIN_K - GRIB_TOLERANCE
            or np.nanmax(written) > SST_MAX_K + GRIB_TOLERANCE
        ):
            raise ValueError(
                f"SST {np.nanmin(written):.2f}..{np.nanmax(written):.2f} K leaves the {SST_MIN_K}..{SST_MAX_K} K range of the surface cycle"
            )
    node[name] = native.copy(data=data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)  # never write through a link into fix_src
    save_grib(tree, destination, template)
    coords = {"month": months, "lat": grid["lat"].values, "lon": grid["lon"].values}
    return (
        xr.DataArray(data[:, 0], dims=("month", "lat", "lon"), coords=coords),
        xr.DataArray(stock[:, 0], dims=("month", "lat", "lon"), coords=coords),
    )


def nice_limit(x: float) -> float:
    """Smallest of 1, 2, 2.5, 5 times a power of ten that is at least x (0 gives 1e-3)."""
    if not np.isfinite(x) or x <= 0:
        return 1e-3
    mag = 10.0 ** np.floor(np.log10(x))
    return float(next(m * mag for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * mag >= x))


def plot_pamip_change(
    stock: xr.DataArray,
    written: xr.DataArray,
    kind: str,
    title: str,
    out_path: Path,
) -> float | None:
    """Draw the change of a PAMIP climatology file against the stock SHiELD file.

    ``stock`` and ``written`` are ``(month, lat, lon)`` DataArrays on the same
    grid (SST in K, sea ice as a fraction). The figure is an xarray facet grid,
    one Robinson panel per month of ``written - stock``, with a colour scale
    symmetric about zero shared by the twelve panels. Points that did not
    change (land, and ocean where the experiment matches the stock field) are
    white. Grids finer than ``PLOT_LON`` points in longitude are block-averaged
    for drawing only. Returns the largest absolute change of the full-resolution
    difference. A failure is logged and does not stop the run (returns None).
    """
    try:
        import cartopy
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Natural Earth cache of fv3_plot_grid.py
        carto = Path(state.fix_src) / "external" / "CARTOPY"
        if carto.is_dir():
            cartopy.config["pre_existing_data_dir"] = carto
            cartopy.config["data_dir"] = carto

        diff = (written - stock).sortby("month").sortby("lat").sortby("lon")
        peak = float(np.nanmax(np.abs(diff.values)))
        factor = max(diff.sizes["lon"] // PLOT_LON, 1)
        if (
            factor > 1
            and diff.sizes["lon"] % factor == 0
            and diff.sizes["lat"] % factor == 0
        ):
            diff = diff.coarsen(lat=factor, lon=factor).mean()
        diff = diff.assign_coords(
            lon=((diff["lon"] + 180.0) % 360.0) - 180.0,
            month=[MONTHS[int(m) - 1] for m in diff["month"].values],
        ).sortby("lon")
        limit = nice_limit(peak)
        cmap, label = (
            ("RdBu_r", "SST change (K)")
            if kind == "sst"
            else ("RdBu", "Sea-ice concentration change (fraction)")
        )

        grid = diff.plot(
            x="lon",
            y="lat",
            col="month",
            col_wrap=4,
            transform=ccrs.PlateCarree(),
            subplot_kws={"projection": ccrs.Robinson(central_longitude=0.0)},
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            size=2.4,
            aspect=1.55,
            rasterized=True,
            add_colorbar=False,
        )
        grid.set_titles("{value}", size=10)
        grid.fig.set_layout_engine("constrained")
        grid.fig.colorbar(
            grid.axs.flat[0].collections[0],
            ax=list(grid.axs.flat),
            orientation="horizontal",
            label=label,
            shrink=0.5,
            aspect=40,
        )
        for ax in grid.axs.flat:
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, edgecolor="#66655D", linewidth=0.4)
        grid.fig.suptitle(
            f"{title}: file minus stock SHiELD file. Largest change {peak:.3g}",
            fontsize=12,
        )
        out_path = Path(out_path)
        out_path.unlink(missing_ok=True)
        grid.fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(grid.fig)
        log.info(f"Wrote {out_path}")
        return peak
    except Exception:
        log.exception(f"PAMIP change plot {Path(out_path).name} failed; continuing")
        return None


def plot_changes(
    tag, experiment, baseline, fields, sst_stock, sst_out, sic_stock, sic_out
):
    """Draw the SST and sea-ice change of the written files against the stock files.

    The figures ``pamip_<tag>_sst_change.png`` and ``pamip_<tag>_sic_change.png``
    go to ``state.run_dir``, where fv3_plot_grid.py saves its figures. Ensemble
    members share that directory: the first takes a lock and draws, the others
    find the figure and skip it (delete a figure to draw it again).
    """
    title = f"PAMIP {experiment}, baseline {baseline}, fields {fields}"
    for kind, stock, written in (
        ("sst", sst_stock, sst_out),
        ("sic", sic_stock, sic_out),
    ):
        out = Path(state.run_dir) / f"pamip_{tag}_{kind}_change.png"
        lock = out.with_suffix(".lock")
        try:
            if out.exists():
                continue
            lock.touch(exist_ok=False)
        except FileExistsError:
            continue
        try:
            peak = plot_pamip_change(stock, written, kind, title, out)
            log.info(f"PAMIP {kind} change against the stock file: largest {peak}")
        finally:
            lock.unlink(missing_ok=True)


def apply_pamip_experiment() -> bool:
    """Write the SST and sea-ice climatologies of the PAMIP experiment ``state.pamip_experiment``.

    The files go to ``FIXED/<family>`` (``FIXED/PAMIP``) as
    ``RTGSST.<family>_<exp>.monthly.clim.grb`` and
    ``CFSR.SEAICE.<family>_<exp>.monthly.clim.grb`` (``<family>_native_<exp>``
    for the native baseline, with ``_sst`` or ``_sic`` appended when
    ``pamip_fields`` selects one field); their paths are stored in
    ``state.fntsfc`` and ``state.fnaisc``, which ``update_namsfc`` writes to namsfc.
    ``fix_src`` is only read. At the cold start the SST and sea ice of the surface
    files in ``state.input`` are also set to the fields of the start date.

    Returns True when files are written, False when no experiment is configured.
    Raises FileNotFoundError for a missing native or PAMIP file and ValueError
    when an input or an output fails its checks.
    """
    experiment = validate_pamip_experiment(state.pamip_experiment)
    if experiment is None:
        return False

    baseline = state.pamip_baseline or "pamip"
    fields = state.pamip_fields or "both"
    experiments = yaml.safe_load(REGISTRY_FILE.read_text())["experiments"]
    entry, control = experiments[experiment], experiments[CONTROL]
    family = entry[
        "family"
    ]  # PAMIP: directory below FIXED and fix_src/external, and tag of the file names
    out_dir = Path(state.fix) / family
    am_dir = Path(state.fix_src) / "am"
    pamip_dir = Path(state.fix_src) / "external" / family
    # a field that pamip_fields leaves out comes from the control
    specs = {
        kind: (entry if fields in ("both", kind) else control)[kind]
        for kind in ("sst", "sic")
    }
    src = xr.merge(
        [read_pamip(pamip_dir, specs[kind], kind) for kind in ("sst", "sic")],
        join="exact",
    )
    if baseline == "native":
        # the PAMIP perturbation (experiment minus control), added to the native files
        src = src - xr.merge(
            [read_pamip(pamip_dir, control[kind], kind) for kind in ("sst", "sic")],
            join="exact",
        )
    log.info(f"PAMIP experiment {experiment}, baseline {baseline}, fields {fields}")

    # The PAMIP files hold filled values over land: keep the ocean of the 0.5 degree land mask
    # (bilinear, threshold 0.5 as in sfcsub.F) and add one row at each pole for tile points beyond the last row.
    node = open_grib(am_dir / LAND_FILE).leaves[0]
    land = node[next(iter(node.data_vars))]
    land = land.isel(
        {d: 0 for d in land.dims if d not in ("latitude", "longitude")}
    ).rename(latitude="lat", longitude="lon")
    land = (
        land.assign_coords(lon=np.mod(land["lon"], 360.0))
        .sortby("lat")
        .sortby("lon")
        .load()  # in memory, so that src and the regridding below are not lazy (dask)
    )
    src = src.where(ll_regrid(land, src) < 0.5)
    src = xr.concat(
        [
            src.isel(lat=[0]).assign_coords(lat=[-90.0]),
            src,
            src.isel(lat=[-1]).assign_coords(lat=[90.0]),
        ],
        dim="lat",
    )

    land = xr.concat(  # pole rows too, for the template points beyond the last row
        [
            land.isel(lat=[0]).assign_coords(lat=[-90.0]),
            land,
            land.isel(lat=[-1]).assign_coords(lat=[90.0]),
        ],
        dim="lat",
    )

    tag = (
        f"{family}_{experiment}"
        if baseline == "pamip"
        else f"{family}_native_{experiment}"
    )
    if fields != "both":
        tag = f"{tag}_{fields}"
    names = {
        key: file.replace("1982.2012", tag)
        for key, file in (("sst", SST_FILE), ("sic", ICE_FILE))
    }
    perturbation = baseline == "native"
    sst_out, sst_stock = write_pamip_grib(
        am_dir / SST_FILE, out_dir / names["sst"], src["sst"], "sst", land, perturbation
    )
    sic_out, sic_stock = write_pamip_grib(
        am_dir / ICE_FILE, out_dir / names["sic"], src["sic"], "sic", land, perturbation
    )
    sic_ocean = sic_out.values[sic_out.values <= LAND_FLAG]
    if (
        np.isnan(sic_ocean).any()
        or sic_ocean.min() < -GRIB_TOLERANCE
        or sic_ocean.max() > 1.0 + GRIB_TOLERANCE
    ):
        raise ValueError(
            "Sea-ice concentration outside 0..1 or missing in the written file"
        )

    if state.restart_no == 0:
        plot_changes(
            tag, experiment, baseline, fields, sst_stock, sst_out, sic_stock, sic_out
        )

    # update_namsfc writes these paths to namsfc FNTSFC and FNAISC.
    state.fntsfc = f"FIXED/{family}/{names['sst']}"
    state.fnaisc = f"FIXED/{family}/{names['sic']}"
    if (
        state.restart_no != 0
    ):  # later segments continue from the cycled state of the model
        return True

    # Surface files: the fields of the start date, interpolated between mid-month values as in the
    # surface cycle (sfcsub.F), regridded onto each tile and written to slmsk, tsea, fice, hice and
    # tisfc over non-land points, so the initial state agrees with the forcing. Rules after sfcsub.F
    # and the CHGRES_CUBE conventions: ice where the concentration is at least AISLIM, with slmsk = 2,
    # below it open water (slmsk = 0, fice = hice = 0); open water has tsea = tisfc = SST; ice that
    # stays ice keeps its thickness and ice-surface temperature (at most the melting point), new ice gets
    # SIHNEW and TGICE; over ice tsea = fice * tisfc + (1 - fice) * TGICE, which sfcsub.F inverts to
    # get tisfc. Land points are not touched. As slmsk already agrees, newice of sfcsub.F does not
    # fire at points that change between ice and open water, so its resets are applied here: snow
    # (sheleg, snwdph) 0, roughness ZORL_NEWICE and soil (sea-ice layer) temperature stc TGICE.
    date = pd.Timestamp(state.init_datetime)
    rjday = date.dayofyear + date.hour / 24.0
    if rjday < DAYHF[0]:
        rjday += 365.0
    m = int(np.searchsorted(DAYHF, rjday, side="right") - 1)
    w2 = (rjday - DAYHF[m]) / (DAYHF[m + 1] - DAYHF[m])
    if perturbation:
        # the written fields on the 0.5 degree sea-ice grid (SST sampled at its points, land missing);
        # sorted, as the GRIB longitudes start at -0.25 (359.75) and nearest selection needs order
        sic = sic_out.where(sic_out <= LAND_FLAG).sortby("lat").sortby("lon")
        sst = sst_out.sortby("lat").sortby("lon")
        sst = sst.sel(lat=sic["lat"], lon=sic["lon"], method="nearest")
        sst = sst.assign_coords(lat=sic["lat"], lon=sic["lon"]).where(sic.notnull())
        written = xr.Dataset({"sst": sst, "sic": sic}).sortby("month")
    else:
        written = src
    now = written.isel(month=m) * (1.0 - w2) + written.isel(month=(m + 1) % 12) * w2
    for v in ("sst", "sic"):  # land takes the nearest ocean value
        iy, ix = distance_transform_edt(
            now[v].isnull().values, return_distances=False, return_indices=True
        )
        now[v] = now[v].copy(data=now[v].values[iy, ix])

    tiles = (
        [1]
        if state.gtype in ("regional_gfdl", "regional_esg")
        else list(range(1, 7 + state.n_nests))
    )
    for tile in tiles:
        filename = sfc_filename(tile)
        path = Path(state.input) / filename
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        out = to_fv3_grid(now, load_grid(filename, tile), periodic=True)
        sst_t, sic_t = (
            out[v].isel(Time=0).transpose("yaxis_1", "xaxis_1").values.astype(float)
            for v in ("sst", "sic")
        )

        with xr.open_dataset(path, decode_cf=False, engine="netcdf4") as handle:
            ds = handle.load()
        slmsk = ds["slmsk"].values[0].copy()
        tsea, fice, hice, tisfc, sheleg, snwdph, zorl, stc = (
            ds[v].values[0].copy()
            for v in (
                "tsea",
                "fice",
                "hice",
                "tisfc",
                "sheleg",
                "snwdph",
                "zorl",
                "stc",
            )
        )
        valid = (slmsk != 1) & np.isfinite(sst_t) & np.isfinite(sic_t)
        sst_c = np.clip(sst_t, SST_MIN_K, SST_MAX_K)
        sic_c = np.clip(sic_t, 0.0, 1.0)
        ice = valid & (sic_c >= AISLIM)
        water = valid & ~ice
        stay = ice & (slmsk == 2)
        new = ice & (slmsk != 2)
        changed = new | (water & (slmsk == 2))  # new ice and melted ice

        tsea[water] = tisfc[water] = sst_c[water]
        fice[water] = hice[water] = slmsk[water] = 0.0
        slmsk[ice] = 2.0
        fice[ice] = sic_c[ice]
        hice[new] = SIHNEW
        hice[stay] = np.where(hice[stay] > 0.0, hice[stay], SIHNEW)
        tisfc[new] = TGICE
        tisfc[stay] = np.minimum(tisfc[stay], T_MELT)
        tsea[ice] = fice[ice] * tisfc[ice] + (1.0 - fice[ice]) * TGICE
        sheleg[changed] = snwdph[changed] = 0.0
        zorl[changed] = ZORL_NEWICE
        stc[:, changed] = TGICE

        for v, a in (
            ("slmsk", slmsk),
            ("tsea", tsea),
            ("fice", fice),
            ("hice", hice),
            ("tisfc", tisfc),
            ("sheleg", sheleg),
            ("snwdph", snwdph),
            ("zorl", zorl),
            ("stc", stc),
        ):
            ds[v] = ds[v].copy(data=a[np.newaxis].astype(ds[v].dtype))
            ds[v].attrs.pop(
                "checksum", None
            )  # the data no longer match an FMS checksum
        backup = Path(state.ic_data) / "perts" / f"{filename.stem}.pamip.original.nc"
        if not backup.exists():
            cp(path, backup)
        ds.to_netcdf(path, engine="netcdf4")
    return True
