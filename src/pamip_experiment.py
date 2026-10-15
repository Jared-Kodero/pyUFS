"""PAMIP SST and sea-ice boundary conditions for SHiELD.

The monthly SST and sea-ice climatologies read by the surface cycle (namsfc
FNTSFC and FNAISC) are replaced by the official PAMIP boundary conditions
(Smith et al., 2019, doi:10.5194/gmd-12-1139-2019, distributed through
input4MIPs). Each experiment is a pair of absolute monthly climatologies (``tos``
and ``siconc``) that already holds the SST adjustment at the changed ice edge of
the protocol, so nothing is added to them: experiments differ only through the
files they use, the control (1.1) included.

The PAMIP fields are regridded with ``latlon_to_fv3.ll_regrid`` onto the grids of
the stock GRIB files and written through the stock messages
(``grib_io.save_grib``), so headers, packing, month stamps and the land flags of
the sea-ice file stay as they are. Only ocean points of the SHiELD land mask
(``seaice_newland.grb``) change in the sea-ice file and in the perturbed
baselines; land points keep the stock values, as ``sfcsub.F`` weights the
climatologies by that mask and never reads land values at ocean points. The sea-ice file
must keep its land flag (values above 1), because ``sfcsub.F`` takes the sea-ice
land mask from the data itself. With the ``pamip`` baseline the SST file is the
exception: the SST of the PAMIP files, which holds filled values over land, is written at
every point, land included, so that the file is the PAMIP file on the grid of the stock file. At the cold start the same fields,
interpolated to the start date, are also written to the surface files of every
tile. As with ``sm_perturbations``, the module is driven only by
``run_config.yaml``::

    pamip_experiment: "1.6"
    pamip_baseline: pamip   # pamip (default), stock or external
    pamip_fields: both      # both (default), sst or sic

With ``pamip_baseline: stock`` the PAMIP files are not used as they are: the
perturbation of the experiment (experiment minus the control 1.1) is added to the
stock SHiELD climatologies (``fix_src/am``), so the control of that baseline is
experiment 1.1 (no perturbation, the stock fields with the surface-cycle settings
of the PAMIP runs). ``pamip_fields`` takes only the SST or only the sea ice from the
experiment; the other field is the control (PAMIP 1.1, or the stock field with
the stock baseline). The protocol pairs hold the ice-edge SST adjustment, which
``sst`` or ``sic`` alone leaves out.

The experiment table is ``templates/pamip_experiments.yaml``; the NetCDF files
are read from ``<fix_src>/external/PAMIP/<source_id>/<variable_id>/``
(``tools/get_pamip.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import ndimage

from fv3_runtime import log
from fv3_state import state
from fv3_utils import cp
from grib_io import open_grib, save_grib
from latlon_to_fv3 import (
    fill_nearest,
    ll_regrid,
    load_grid,
    sfc_filename,
    to_fv3_grid,
)

SST_FILE = "RTGSST.1982.2012.monthly.clim.grb"
ICE_FILE = "CFSR.SEAICE.1982.2012.monthly.clim.grb"
LAND_FILE = "seaice_newland.grb"  # 0.5 degree land mask, 1 = land; only read
REGISTRY_FILE = (
    Path(__file__).resolve().parent.parent / "templates" / "pamip_experiments.yaml"
)
CONTROL = "1.1"  # the perturbation of the stock baseline is experiment minus CONTROL
BASELINES = ("pamip", "stock", "external")
# run_config options of the climatologies a case provides, and the namsfc entry of each
SURFACE_FILES = {
    "external_sst": "fntsfc",
    "external_sic": "fnaisc",
    "external_mask": "fnmskh",
}
SURFACE_KEYS = tuple(SURFACE_FILES)
EXTERNAL_DIR = (
    "external"  # directory of FIXED that holds the copies of the provided files
)
FIELDS = ("both", "sst", "sic")

# sfcsub.F: SST outside 271.2..313.0 K over open ocean is replaced by the surface
# cycle, aislim = 0.15 is the sea-ice threshold, sea-ice values above 1 are land.
SST_MIN_K, SST_MAX_K = 271.2, 313.0
AISLIM = 0.15
LAND_FLAG = 1.0
LAKE_CELLS = 500  # water bodies of the 0.5 degree mask below this size (lakes, Baltic, Black, Caspian) keep the stock values

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
    if baseline == "external":
        missing = [k for k in SURFACE_KEYS if not getattr(state, k)]
        if missing:
            raise ValueError(
                f"`pamip_baseline: external` needs `external_sst`, `external_sic` and `external_mask` to be set to existing files. Not set: {missing}"
            )
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


def validate_surface_files() -> None:
    """Check the ``external_sst``, ``external_sic`` and ``external_mask`` options before any file is written.

    A relative path is taken from ``fix_src``. Raises TypeError for a value that
    is not text and FileNotFoundError for a file that does not exist. Call it once,
    at the cold start, before ``stage_surface_files`` replaces the paths.
    """
    for key in SURFACE_KEYS:
        value = getattr(state, key)
        if value is None or value == "":
            setattr(state, key, None)
            continue
        if not isinstance(value, str):
            raise TypeError(f"`{key}` must be a path as text. Got {value!r}")
        if not surface_source(value).is_file():
            raise FileNotFoundError(
                f"`{key}`: {surface_source(value)} is not a file (a relative path is taken from fix_src)"
            )
    names = [Path(getattr(state, k)).name for k in SURFACE_KEYS if getattr(state, k)]
    if len(set(names)) != len(names):
        raise ValueError(
            f"`external_sst`, `external_sic` and `external_mask` need different file names. Got {names}"
        )


def surface_source(value: str) -> Path:
    """Path of a provided file: absolute, or relative to ``fix_src``."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(state.fix_src) / path


def stage_surface_files() -> None:
    """Copy the files given by ``external_sst``, ``external_sic`` and ``external_mask`` to ``FIXED/external``.

    Each is a copy, never a link, so the case does not depend on the original
    and nothing is written to it. ``state.fntsfc``, ``state.fnaisc`` and
    ``state.fnmskh`` (the namsfc entries) then hold ``FIXED/external/<name>``, which ``update_namsfc``
    writes to namsfc; a PAMIP experiment with ``pamip_baseline: external`` adds
    its perturbation to these copies. Files not given stay the stock ones.
    """
    for key, entry in SURFACE_FILES.items():
        value = getattr(state, key)
        if not value:
            continue
        source = surface_source(value)
        destination = Path(state.fix) / EXTERNAL_DIR / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        shutil.copyfile(source, destination)
        setattr(state, entry, f"FIXED/{EXTERNAL_DIR}/{source.name}")


def pamip_namelist() -> dict[str, dict]:
    """SHiELD namelist settings of the configured experiment by namelist group, empty when none is set."""
    experiment = validate_pamip_experiment(state.pamip_experiment)
    return {} if experiment is None else PAMIP_EXPERIMENTS[experiment]["namelist"]


def read_pamip(root: Path, spec: dict, kind: str) -> xr.DataArray:
    """One PAMIP climatology as ``(month, lat, lon)``: SST in kelvin or sea ice as a fraction, filled values NaN."""
    variable, source = spec["variable_id"], spec["source_id"]
    path = (
        Path(root)
        / source
        / variable
        / f"{variable}_input4MIPs_SSTsAndSeaIce_PAMIP_{source}_gn_{spec['period']}.nc"
    )
    with xr.open_dataset(
        path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
    ) as ds:
        da = ds[variable].squeeze(drop=True).load()
    da = da.assign_coords(time=da.time.dt.month, lon=da.lon % 360.0)
    da = da.rename(time="month").sortby(["month", "lat", "lon"])
    da = da.where(abs(da) < 1e10).rename(kind)
    return da.clip(0.0, 1.0) if kind == "sic" else da


def write_pamip_grib(
    template: Path,
    destination: Path,
    field: xr.DataArray,
    kind: str,
    land: xr.DataArray,
    lake: xr.DataArray,
    perturbation: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Regrid a climatology onto the grid of a stock GRIB file and write it through the messages of that file.

    ``land`` is the 0.5 degree land mask (1 = land, lakes included) of SHiELD, regridded to the
    template grid and thresholded at 0.5 as in sfcsub.F; ``lake`` marks the lakes, which keep the stock values. Only ocean points
    change; land points keep the stock values (including the 1.57 sea-ice land
    flag), which the surface cycle ignores at ocean points. Without ``perturbation``
    the SST is the PAMIP SST at every point, land included (the PAMIP files hold
    filled values there); sea ice is clipped to 0..1 at ocean points. With
    ``perturbation`` the field is added to the stock values: sea ice is clipped to
    0..1, and SST changes only where the stock value lies in the
    SST_MIN_K..SST_MAX_K range of the surface cycle and stays between the stock
    freezing value of the month and SST_MAX_K. Returns the values written and the
    stock values, both ``(month, lat, lon)`` in the month order of the template.
    """
    tree = open_grib(template)
    node = tree.leaves[0]
    name = next(iter(node.data_vars))
    native = node[name]
    stock = (
        native.squeeze(drop=True)
        .assign_coords(month=native.time.dt.month)
        .swap_dims(time="month")
        .rename(latitude="lat", longitude="lon")
        .load()
    )
    grid = stock.assign_coords(lon=stock.lon % 360.0)
    # one regridder for the twelve months
    new = fill_nearest(ll_regrid(field, grid)).assign_coords(
        lat=stock.lat, lon=stock.lon
    )
    ocean = ll_regrid(land, grid).assign_coords(lat=stock.lat, lon=stock.lon) < 0.5
    inland = ll_regrid(lake, grid).assign_coords(lat=stock.lat, lon=stock.lon) > 0.5
    if kind == "sic":
        out = (stock + new if perturbation else new).clip(0.0, 1.0)
        out = out.where(ocean & (stock <= LAND_FLAG), stock)
    elif perturbation:
        inside = ocean & (stock >= SST_MIN_K) & (stock <= SST_MAX_K)
        freezing = stock.where(inside).min(("lat", "lon"))
        out = (stock + new).clip(freezing, SST_MAX_K).where(inside, stock)
    else:
        out = new.where(~inland, stock)
    node[name] = native.copy(data=out.values[:, np.newaxis])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)  # never write through a link into fix_src
    save_grib(tree, destination, template)
    return out, stock


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

        # Natural Earth cache of fv3_plot_grid.py
        carto = Path(state.fix_src) / "external" / "CARTOPY"
        if carto.is_dir():
            cartopy.config["pre_existing_data_dir"] = carto
            cartopy.config["data_dir"] = carto

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
    ``CFSR.SEAICE.<family>_<exp>.monthly.clim.grb`` (``<family>_stock_<exp>`` or
    ``<family>_external_<exp>`` for the stock or external baseline, with ``_sst`` or ``_sic`` appended when
    ``pamip_fields`` selects one field); their paths are stored in
    ``state.fntsfc`` and ``state.fnaisc``, which ``update_namsfc`` writes to namsfc.
    ``fix_src`` is only read. With ``pamip_baseline: external`` the perturbation
    is added to the copies in ``FIXED/external`` of the files given by
    ``external_sst``, ``external_sic`` and ``external_mask`` (``stage_surface_files``), and the
    ocean of ``external_mask`` decides which points change. At the cold start the SST and sea ice of the surface
    files in ``state.input`` are also set to the fields of the start date.

    Returns True when files are written, False when no experiment is configured.
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
    # the files the anomaly is added to (stock baseline: the stock files; external: the copies
    # in FIXED/external of the files given by external_sst, external_sic and external_mask)
    base = {
        "sst": am_dir / SST_FILE,
        "sic": am_dir / ICE_FILE,
        "land": am_dir / LAND_FILE,
    }
    if baseline != "external" and any(getattr(state, k) for k in SURFACE_KEYS):
        log.warning(
            f"external_sst, external_sic and external_mask are not the baseline of pamip_baseline {baseline}; "
            "the SST and sea-ice files of the experiment replace the SST and sea-ice files"
        )
    if baseline == "external":  # all three are set and exist (validation)
        for kind, key in (
            ("sst", "fntsfc"),
            ("sic", "fnaisc"),
            ("land", "fnmskh"),
        ):  # staged copies
            base[kind] = Path(state.fix) / Path(getattr(state, key)).relative_to(
                "FIXED"
            )
    # a field that pamip_fields leaves out comes from the control
    specs = {
        kind: (entry if fields in ("both", kind) else control)[kind]
        for kind in ("sst", "sic")
    }
    src = xr.merge(
        [read_pamip(pamip_dir, specs[kind], kind) for kind in ("sst", "sic")],
        join="exact",
    )
    if baseline != "pamip":
        # the PAMIP perturbation (experiment minus control), added to the stock or external files
        src = src - xr.merge(
            [read_pamip(pamip_dir, control[kind], kind) for kind in ("sst", "sic")],
            join="exact",
        )

    # The sea ice and the perturbed baselines change only the ocean of the 0.5 degree land mask
    # (bilinear, threshold 0.5 as in sfcsub.F). One row is added at each pole for tile points
    # beyond the last row.
    node = open_grib(base["land"]).leaves[0]
    land = next(iter(node.data_vars.values())).squeeze(drop=True)
    land = land.rename(latitude="lat", longitude="lon")
    land = land.assign_coords(lon=land.lon % 360.0).sortby(["lat", "lon"]).load()
    labels, count = ndimage.label(land.values < 0.5)
    size = ndimage.sum(labels > 0, labels, np.arange(1, count + 1))
    lake = land.copy(data=np.isin(labels, 1 + np.flatnonzero(size < LAKE_CELLS)))
    land = land.where(~lake, 1.0)  # lakes are land for the changes below
    ocean_src = ll_regrid(land, src) < 0.5
    if baseline == "pamip":  # the PAMIP SST keeps its land values, the sea ice does not
        src["sic"] = src["sic"].where(ocean_src)
    else:
        src = src.where(ocean_src)
    src = src.reindex(lat=[-90.0, *src.lat.values, 90.0], method="nearest")
    land = land.reindex(lat=[-90.0, *land.lat.values, 90.0], method="nearest")
    lake = lake.reindex(lat=land.lat, method="nearest").astype("float32")

    tag = (
        f"{family}_{experiment}"
        if baseline == "pamip"
        else f"{family}_{baseline}_{experiment}"
    )
    if fields != "both":
        tag = f"{tag}_{fields}"
    names = {
        key: file.replace("1982.2012", tag)
        for key, file in (("sst", SST_FILE), ("sic", ICE_FILE))
    }
    perturbation = baseline != "pamip"
    sst_out, sst_stock = write_pamip_grib(
        base["sst"], out_dir / names["sst"], src["sst"], "sst", land, lake, perturbation
    )
    sic_out, sic_stock = write_pamip_grib(
        base["sic"], out_dir / names["sic"], src["sic"], "sic", land, lake, perturbation
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
    if perturbation:
        # the written fields on the 0.5 degree sea-ice grid (SST sampled at its points, land missing)
        sic = sic_out.where(sic_out <= LAND_FLAG)
        sst = sst_out.reindex(lat=sic.lat, lon=sic.lon, method="nearest")
        written = xr.Dataset({"sst": sst.where(sic.notnull()), "sic": sic})
    else:
        written = src
    written = written.sortby("month")
    cycle = xr.concat([written, written.isel(month=[0])], "month")
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
        changed = new | (water & (slmsk == 2))  # new ice and melted ice

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
            water,
            sst_c,
            xr.where(ice, fice * tisfc + (1.0 - fice) * TGICE, ds["tsea"]),
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
    return True
