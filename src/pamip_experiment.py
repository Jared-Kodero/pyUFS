"""PAMIP SST and sea-ice boundary conditions for SHiELD.

The monthly SST and sea-ice climatologies read by the surface cycle (namsfc
FNTSFC and FNAISC) are replaced by the boundary conditions of a PAMIP experiment
(Smith et al., 2019, doi:10.5194/gmd-12-1139-2019, distributed through
input4MIPs). The module is driven only by ``run_config.yaml``::

    pamip_experiment: "1.6"
    pamip_baseline: pamip   # pamip (default), shield or external
    pamip_fields: both      # both (default), sst or sic
    external_sst / external_sic / external_mask   # only with pamip_baseline: external

``pamip_baseline`` says what the experiment is applied to:

* ``pamip``: the PAMIP fields themselves (experiment files, filled values over land
  kept in the SST). Experiments differ only through the files they use.
* ``shield``: the climatologies SHiELD ships in ``<fix_src>/am`` (RTGSST, CFSR.SEAICE,
  seaice_newland.grb) plus the PAMIP perturbation (experiment minus control 1.1).
* ``external``: the same, with the files of ``external_sst``, ``external_sic`` and
  ``external_mask`` as the baseline instead of the SHiELD files.

``pamip_fields`` takes only the SST or only the sea ice from the experiment; the
other field is the control 1.1. The experiment table is ``templates/pamip_experiments.yaml``;
the NetCDF files are read from ``<fix_src>/external/PAMIP/<source_id>/<variable_id>/``
(``tools/get_pamip.py``).

The four functions, called in this order by ``fv3_init_driver`` at the cold start:

1. ``validate_pamip_options``: checks every option before any file is written.
2. ``stage_external_files``: copies ``external_sst``, ``external_sic`` and ``external_mask``
   to ``FIXED/external`` and points namsfc at the copies (the baseline of ``external``).
3. ``write_boundary_files``: regrids the PAMIP fields onto the grid of the baseline
   GRIB files and writes the SST and sea-ice climatologies to ``FIXED/PAMIP``.
4. ``set_cold_start_surface``: sets SST and sea ice of the surface files of every tile
   to the fields of the start date, so the initial state agrees with the forcing.

Only ocean points of the land mask of the baseline (0.5 degree, 1 = land) change in
the sea-ice file and in the ``shield`` and ``external`` baselines; land points keep the
baseline values, because ``sfcsub.F`` weights the climatologies by that mask and never
reads land values at ocean points. The sea-ice file keeps its land flag (values above 1),
because ``sfcsub.F`` takes the sea-ice land mask from the data itself. With the ``pamip``
baseline the SST file is the exception: the PAMIP SST, which holds filled values over land,
is written at every point. Lakes and inland seas (water bodies of the mask below
``LAKE_CELLS`` cells) keep the baseline values.
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
from latlon_to_fv3 import fill_nearest, ll_regrid, load_grid, sfc_filename, to_fv3_grid

SST_FILE = "RTGSST.1982.2012.monthly.clim.grb"
ICE_FILE = "CFSR.SEAICE.1982.2012.monthly.clim.grb"
LAND_FILE = "seaice_newland.grb"  # 0.5 degree land mask, 1 = land
REGISTRY_FILE = (
    Path(__file__).resolve().parent.parent / "templates" / "pamip_experiments.yaml"
)
CONTROL = "1.1"  # the perturbation of the shield and external baselines is experiment minus CONTROL
BASELINES = ("pamip", "shield", "external")
FIELDS = ("both", "sst", "sic")
# run_config option of each file a case provides, and the namsfc entry that points to its copy
EXTERNAL_FILES = {
    "external_sst": "fntsfc",
    "external_sic": "fnaisc",
    "external_mask": "fnmskh",
}

# sfcsub.F: SST outside 271.2..313.0 K over open ocean is replaced by the surface
# cycle, AISLIM = 0.15 is the sea-ice threshold, sea-ice values above 1 are land.
SST_MIN_K, SST_MAX_K = 271.2, 313.0
AISLIM = 0.15
LAND_FLAG = 1.0
LAKE_CELLS = 500  # water bodies of the 0.5 degree mask below this size (lakes, Baltic, Black, Caspian) keep the baseline values

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


def validate_pamip_options() -> None:
    """Check the PAMIP and external-file options of ``run_config.yaml`` before any file is written.

    Raises TypeError for a value of the wrong type, ValueError for an unknown or inconsistent
    option and FileNotFoundError for a missing ``external_*`` file. Sets ``state.pamip_experiment``
    to the stripped experiment number (or None) and an empty ``external_*`` option to None.
    """
    experiment = state.pamip_experiment
    baseline = state.pamip_baseline
    fields = state.pamip_fields

    # 1. pamip_experiment, pamip_baseline and pamip_fields
    if experiment is None or experiment == "":
        state.pamip_experiment = None
        if baseline is not None or fields is not None:
            raise ValueError(
                "`pamip_baseline` and `pamip_fields` only apply to a PAMIP experiment: "
                'set `pamip_experiment` (for example "1.6") or set both options to null'
            )
    else:
        if not isinstance(experiment, str):
            raise TypeError(
                f'`pamip_experiment` must be quoted text such as "1.6" (YAML reads 1.10 as the number 1.1). Got {experiment!r}'
            )
        experiment = experiment.strip()
        if experiment not in PAMIP_EXPERIMENTS:
            raise ValueError(
                f"`pamip_experiment` must be one of {tuple(PAMIP_EXPERIMENTS)}. Got {experiment!r}"
            )
        registry = yaml.safe_load(REGISTRY_FILE.read_text())["experiments"]
        if experiment not in registry:
            raise ValueError(
                f"PAMIP experiment {experiment} ({PAMIP_EXPERIMENTS[experiment]['name']}) has no SST and "
                f"sea-ice files in {REGISTRY_FILE.name}. Experiments with files: {tuple(registry)}"
            )
        if baseline is not None and baseline not in BASELINES:
            raise ValueError(
                f"`pamip_baseline` must be one of {BASELINES} or null. Got {baseline!r}"
            )
        if fields is not None and fields not in FIELDS:
            raise ValueError(
                f"`pamip_fields` must be one of {FIELDS} or null. Got {fields!r}"
            )
        state.pamip_experiment = experiment

    # 2. external_sst, external_sic and external_mask: a path as text, absolute or relative to fix_src
    for key in EXTERNAL_FILES:
        value = getattr(state, key)
        if value is None or value == "":
            setattr(state, key, None)
            continue
        if not isinstance(value, str):
            raise TypeError(f"`{key}` must be a file path as text. Got {value!r}")
        path = (
            Path(state.fix_src) / Path(value).expanduser()
        )  # an absolute path stays as it is
        if not path.is_file():
            raise FileNotFoundError(
                f"`{key}` is {value!r}, which resolves to {path}. That is not a file "
                f"(a relative path is taken from fix_src = {state.fix_src})"
            )
    given = [
        Path(getattr(state, key)).name for key in EXTERNAL_FILES if getattr(state, key)
    ]
    if len(set(given)) != len(given):
        raise ValueError(
            "`external_sst`, `external_sic` and `external_mask` need three different file names, "
            f"because each is copied to FIXED/external under its own name. Got {given}"
        )
    if state.pamip_experiment is not None and baseline == "external":
        missing = [key for key in EXTERNAL_FILES if not getattr(state, key)]
        if missing:
            raise ValueError(
                "`pamip_baseline: external` adds the PAMIP perturbation to the files of `external_sst`, "
                f"`external_sic` and `external_mask`, so all three must be set. Not set: {missing}"
            )
    if (
        state.pamip_experiment is not None
        and baseline != "external"
        and any(getattr(state, key) for key in EXTERNAL_FILES)
    ):
        log.warning(
            "external_sst, external_sic and external_mask are not the baseline with "
            f"pamip_baseline {baseline or 'pamip'}: only pamip_baseline: external adds the "
            "perturbation to them. The SST and sea-ice files of the experiment replace the "
            "SST and sea-ice files they give"
        )


def stage_external_files() -> None:
    """Copy the files of ``external_sst``, ``external_sic`` and ``external_mask`` to ``FIXED/external``.

    Each is a copy, never a link, so the case does not depend on the original and nothing is
    written to it. ``state.fntsfc``, ``state.fnaisc`` and ``state.fnmskh`` (the namsfc entries)
    then hold ``FIXED/external/<name>``, which ``update_namsfc`` writes to namsfc. With
    ``pamip_baseline: external`` ``write_boundary_files`` reads these copies as the baseline.
    A file that is not given stays the SHiELD file of ``<fix_src>/am``.
    """
    for key, entry in EXTERNAL_FILES.items():
        value = getattr(state, key)
        if not value:
            continue
        source = Path(state.fix_src) / Path(value).expanduser()
        destination = Path(state.fix) / "external" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        shutil.copyfile(source, destination)
        setattr(state, entry, f"FIXED/external/{source.name}")


def write_boundary_files() -> xr.Dataset | None:
    """Write the SST and sea-ice climatologies of ``state.pamip_experiment`` to ``FIXED/PAMIP``.

    The files are ``RTGSST.PAMIP_<exp>.monthly.clim.grb`` and ``CFSR.SEAICE.PAMIP_<exp>.monthly.clim.grb``
    (``PAMIP_shield_<exp>`` or ``PAMIP_external_<exp>`` for those baselines, with ``_sst`` or ``_sic``
    appended when ``pamip_fields`` selects one field). Their paths go to ``state.fntsfc`` and
    ``state.fnaisc``, which ``update_namsfc`` writes to namsfc. At the cold start the figures
    ``pamip_<tag>_sst_change.png`` and ``pamip_<tag>_sic_change.png`` (file minus baseline file) go to
    ``state.run_dir``; ensemble members share it, the first draws and the others skip.

    Returns the written fields as a Dataset ``(month, lat, lon)`` of ``sst`` and ``sic`` for
    ``set_cold_start_surface``, or None when no experiment is configured.
    """
    if state.pamip_experiment is None:
        return None
    experiment = state.pamip_experiment
    baseline = state.pamip_baseline or "pamip"
    fields = state.pamip_fields or "both"
    registry = yaml.safe_load(REGISTRY_FILE.read_text())["experiments"]
    entry, control = registry[experiment], registry[CONTROL]
    family = entry[
        "family"
    ]  # PAMIP: directory below FIXED and fix_src/external, and tag of the file names

    # 1. The baseline files: they give the grid, the messages written through and the values kept
    #    at points that do not change.
    if (
        baseline == "external"
    ):  # the copies made by stage_external_files; validation made sure all three are set
        base_files = {
            "sst": Path(state.fix) / Path(state.fntsfc).relative_to("FIXED"),
            "sic": Path(state.fix) / Path(state.fnaisc).relative_to("FIXED"),
            "land": Path(state.fix) / Path(state.fnmskh).relative_to("FIXED"),
        }
    else:  # pamip and shield: the SHiELD files of fix_src/am (pamip only takes the grid from them)
        am_dir = Path(state.fix_src) / "am"
        base_files = {
            "sst": am_dir / SST_FILE,
            "sic": am_dir / ICE_FILE,
            "land": am_dir / LAND_FILE,
        }

    # 2. The PAMIP fields as (month, lat, lon). A field that pamip_fields leaves out comes from the control.
    #    With the shield and external baselines the field is the perturbation, experiment minus control.
    pamip = {}
    for kind in ("sst", "sic"):
        chosen = entry if fields in ("both", kind) else control
        read = {}
        for role, spec in (("experiment", chosen[kind]), ("control", control[kind])):
            variable, source = spec["variable_id"], spec["source_id"]
            path = (
                Path(state.fix_src) / "external" / family / source / variable
                / f"{variable}_input4MIPs_SSTsAndSeaIce_PAMIP_{source}_gn_{spec['period']}.nc"
            )  # fmt: skip
            if not path.is_file():
                raise FileNotFoundError(
                    f"PAMIP file {path} not found. Download it with: python tools/get_pamip.py --root {state.fix_src}"
                )
            with xr.open_dataset(
                path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
            ) as ds:
                da = ds[variable].squeeze(drop=True).load()
            da = da.assign_coords(time=da.time.dt.month, lon=da.lon % 360.0)
            da = da.rename(time="month").sortby(["month", "lat", "lon"])
            da = da.where(abs(da) < 1e10)  # filled values are NaN
            read[role] = da.clip(0.0, 1.0) if kind == "sic" else da
        if baseline == "pamip":
            pamip[kind] = read["experiment"]
        else:
            with xr.set_options(arithmetic_join="exact"):
                pamip[kind] = read["experiment"] - read["control"]

    # 3. The land mask of the baseline, at 0.5 degrees. Lakes and inland seas (small water bodies) count
    #    as land for the changes below and keep the baseline values.
    node = open_grib(base_files["land"]).leaves[0]
    land = next(iter(node.data_vars.values())).squeeze(drop=True)
    land = land.rename(latitude="lat", longitude="lon")
    land = land.assign_coords(lon=land.lon % 360.0).sortby(["lat", "lon"]).load()
    labels, count = ndimage.label(land.values < 0.5)
    size = ndimage.sum(labels > 0, labels, np.arange(1, count + 1))
    lake = land.copy(data=np.isin(labels, 1 + np.flatnonzero(size < LAKE_CELLS)))
    land = land.where(~lake, 1.0)
    ocean_pamip = (
        ll_regrid(land, pamip["sic"]) < 0.5
    )  # ocean on the 1 degree PAMIP grid
    # Only ocean changes, except the PAMIP SST of the pamip baseline, which keeps its filled land values.
    pamip["sic"] = pamip["sic"].where(ocean_pamip)
    if baseline != "pamip":
        pamip["sst"] = pamip["sst"].where(ocean_pamip)
    # One row at each pole, for tile points beyond the last row of the PAMIP grid.
    poles = [-90.0, *pamip["sst"].lat.values, 90.0]
    pamip = {
        kind: da.reindex(lat=poles, method="nearest") for kind, da in pamip.items()
    }
    land = land.reindex(lat=[-90.0, *land.lat.values, 90.0], method="nearest")
    lake = lake.reindex(lat=land.lat, method="nearest").astype("float32")

    # 4. Regrid each field to the grid of its baseline file and write it through the messages of that
    #    file, so headers, packing, month stamps and the sea-ice land flags stay as they are.
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
            # the PAMIP sea ice at ocean points; the land flag (above 1) of the baseline file stays
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
            # the PAMIP SST at every point, land included (lakes keep the baseline values)
            out[kind] = new.where(~inland, base[kind])
        else:
            # perturbed only where the baseline lies in the SST range of the surface cycle, and
            # kept between the freezing value of the baseline of the month and SST_MAX_K
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
        destination.unlink(missing_ok=True)  # never write through a link into fix_src
        save_grib(tree, destination, template)

    # 5. Figures of the change at the cold start: the file written minus the baseline file it is made from.
    if state.restart_no == 0:
        try:
            import cartopy
            import cartopy.crs as ccrs
            import matplotlib

            matplotlib.use("Agg")
            carto = (
                Path(state.fix_src) / "external" / "CARTOPY"
            )  # Natural Earth cache of fv3_plot_grid.py
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
                    continue  # another ensemble member is drawing it
                try:
                    diff = (out[kind] - base[kind]).sortby("month")
                    # blocks of cells are reduced to their value of largest magnitude, so single-cell changes stay visible
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
                        x="lon", y="lat", col="month", col_wrap=4,
                        transform=ccrs.PlateCarree(), subplot_kws={"projection": ccrs.Robinson()},
                        cmap=cmap, vmin=-limit, vmax=limit, size=2.4, aspect=1.55, rasterized=True, add_colorbar=False,
                    )  # fmt: skip
                    for ax in grid.axs.flat:
                        ax.coastlines(linewidth=0.4)
                    grid.fig.set_layout_engine("constrained")
                    grid.fig.colorbar(
                        grid.axs.flat[0].collections[0], ax=list(grid.axs.flat),
                        orientation="horizontal", label=label, shrink=0.5,
                    )  # fmt: skip
                    # name of the file written and of the file it is compared with, without ".monthly.clim.grb"
                    grid.fig.suptitle(
                        f"{names[kind].removesuffix('.monthly.clim.grb')} - "
                        f"{(base_files[kind]).name.removesuffix('.monthly.clim.grb')}: "
                        f"largest change {float(abs(out[kind] - base[kind]).max()):.3g}"
                    )
                    grid.fig.savefig(figure, dpi=300, bbox_inches="tight")
                finally:
                    lock.unlink(missing_ok=True)
        except Exception:
            log.exception("PAMIP change figure failed; continuing")

    # update_namsfc writes these paths to namsfc FNTSFC and FNAISC.
    state.fntsfc = f"FIXED/{family}/{names['sst']}"
    state.fnaisc = f"FIXED/{family}/{names['sic']}"

    # 6. The fields for the cold start, on one grid. pamip baseline: the PAMIP fields (1 degree, with the
    #    pole rows). Others: the files written, on the 0.5 degree sea-ice grid, SST sampled at its points.
    if baseline == "pamip":
        return xr.Dataset({"sst": pamip["sst"], "sic": pamip["sic"]})
    sic = out["sic"].where(out["sic"] <= LAND_FLAG)
    sst = out["sst"].reindex(lat=sic.lat, lon=sic.lon, method="nearest")
    return xr.Dataset({"sst": sst.where(sic.notnull()), "sic": sic})


def set_cold_start_surface(fields: xr.Dataset) -> None:
    """Set SST and sea ice of the surface files of every tile to the fields of the start date.

    ``fields`` is the Dataset returned by ``write_boundary_files``. The fields are interpolated
    between mid-month values as in the surface cycle (sfcsub.F), regridded onto each tile
    (``to_fv3_grid``) and written to slmsk, tsea, fice, hice and tisfc over non-land points, so the initial
    state agrees with the forcing. Rules after sfcsub.F and the CHGRES_CUBE conventions: ice where the
    concentration is at least AISLIM, with slmsk = 2, below it open water (slmsk = 0, fice = hice = 0);
    open water has tsea = tisfc = SST; ice that stays ice keeps its thickness and ice-surface temperature
    (at most the melting point), new ice gets SIHNEW and TGICE; over ice tsea = fice * tisfc +
    (1 - fice) * TGICE, which sfcsub.F inverts to get tisfc. Land points are not touched. As slmsk already
    agrees, newice of sfcsub.F does not fire at points that change between ice and open water, so its
    resets are applied here: snow (sheleg, snwdph) 0, roughness ZORL_NEWICE and soil (sea-ice layer)
    temperature stc TGICE. The original surface file of each tile is kept in ``<ic_data>/perts``.
    """
    date = pd.Timestamp(state.init_datetime)
    rjday = date.dayofyear + date.hour / 24.0
    fields = fields.sortby("month")
    cycle = xr.concat([fields, fields.isel(month=[0])], "month")
    cycle = cycle.assign_coords(day=("month", DAYHF)).swap_dims(month="day")
    now = fill_nearest(cycle.interp(day=rjday + 365.0 * (rjday < DAYHF[0])))

    if state.gtype in ("regional_gfdl", "regional_esg"):
        tiles = [1]
    else:
        tiles = list(range(1, 7 + state.n_nests))
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
