"""PAMIP SST and sea-ice boundary conditions for SHiELD.

The monthly SST and sea-ice climatologies read by the surface cycle (namsfc
FNTSFC and FNAISC) are replaced by the official PAMIP boundary conditions
(Smith et al., 2019, doi:10.5194/gmd-12-1139-2019, distributed through
input4MIPs). Each experiment is a pair of absolute monthly climatologies (``tos``
and ``siconc``) that already holds the SST adjustment at the changed ice edge of
the protocol, so nothing is added to them: experiments differ only through the
files they use, the control (1.1) included.

The PAMIP fields are regridded onto the grids of the native GRIB files with a
mask-normalised bilinear interpolation and written through the native messages
(``grib_io.save_grib``), so headers, packing, month stamps and the land flags of
the sea-ice file stay as they are. As with ``sm_perturbations``, the module is
driven only by ``run_config.yaml``::

    pamip_experiment: "1.6"

The experiment table is ``templates/pamip_experiments.yaml``; the NetCDF files
are read from ``<fix_src>/PAMIP/<source_id>/<variable_id>/``
(``tools/download_pamip_amip.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.ndimage import distance_transform_edt

from fv3_runtime import log
from fv3_state import state
from grib_io import open_grib, save_grib
from latlon_to_fv3 import load_grid, sfc_filename, to_fv3_grid

SST_FILE = "RTGSST.1982.2012.monthly.clim.grb"
ICE_FILE = "CFSR.SEAICE.1982.2012.monthly.clim.grb"
LAND_FILE = "seaice_newland.grb"  # 0.5 degree land mask, 1 = land; only read
REGISTRY_FILE = (
    Path(__file__).resolve().parent.parent / "templates" / "pamip_experiments.yaml"
)

# sfcsub.F: SST outside 271.2..313.0 K over open ocean is replaced by the surface
# cycle, aislim = 0.15 is the sea-ice threshold, sea-ice values above 1 are land.
SST_MIN_K, SST_MAX_K = 271.2, 313.0
AISLIM = 0.15
LAND_FLAG = 1.0
GRIB_TOLERANCE = 0.0051  # half the 0.01 quantum of the native files

# Surface cycle (sfcsub.F): mid-month julian days of the monthly interpolation (dayhf),
# freezing temperature of sea water (tgice), thickness of new ice (sihnew).
DAYHF = (
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
)
TGICE = 271.2
SIHNEW = 0.2
T_MELT = 273.16

# PAMIP experiments of Smith et al. (2019, GMD 12, 1139) and the protocol page
# https://www.cesm.ucar.edu/projects/cmip6/pamip/protocol. "name" is the short name and "namelist" the
# SHiELD namelist settings of the experiment, by namelist group, merged into the model namelist.
# Experiments 1 to 4 and 6 use the radiative forcing of the year 2000 (Appendix B): ictm = yyyy0 makes
# SHiELD read the CO2 table of year yyyy (radiation_gases.f). Set 5 has time-varying forcing and keeps
# the model default. Protocol (tier, start, length, spin-up) is described in templates/pamip_experiments.yaml.
# Only experiments with SST and sea-ice files in that file can be run.
_FORCING_2000 = {"gfs_physics_nml": {"ictm": 20000}}
_FORCING_VARYING: dict[str, dict] = {}
PAMIP_EXPERIMENTS: dict[str, dict] = {
    # Group 1: Atmosphere-only time-slice experiments
    "1.1": {"name": "pdSST-pdSIC", "namelist": _FORCING_2000},
    "1.2": {"name": "piSST-piSIC", "namelist": _FORCING_2000},
    "1.3": {"name": "piSST-pdSIC", "namelist": _FORCING_2000},
    "1.4": {"name": "futSST-pdSIC", "namelist": _FORCING_2000},
    "1.5": {"name": "pdSST-piArcSIC", "namelist": _FORCING_2000},
    "1.6": {"name": "pdSST-futArcSIC", "namelist": _FORCING_2000},
    "1.7": {"name": "pdSST-piAntSIC", "namelist": _FORCING_2000},
    "1.8": {"name": "pdSST-futAntSIC", "namelist": _FORCING_2000},
    "1.9": {"name": "pdSST-pdSICSIT", "namelist": _FORCING_2000},
    "1.10": {"name": "pdSST-futArcSICSIT", "namelist": _FORCING_2000},
    # Group 2: Partially coupled ocean-atmosphere time-slice experiments
    "2.1": {"name": "pa-pdSIC", "namelist": _FORCING_2000},
    "2.2": {"name": "pa-piArcSIC", "namelist": _FORCING_2000},
    "2.3": {"name": "pa-futArcSIC", "namelist": _FORCING_2000},
    "2.4": {"name": "pa-piAntSIC", "namelist": _FORCING_2000},
    "2.5": {"name": "pa-futAntSIC", "namelist": _FORCING_2000},
    # Group 3: Regional sea ice forcing experiments
    "3.1": {"name": "pdSST-futOkhotskSIC", "namelist": _FORCING_2000},
    "3.2": {"name": "pdSST-futBKSeasSIC", "namelist": _FORCING_2000},
    # Group 4: Background SST state experiments
    "4.1": {"name": "modelSST-pdSIC", "namelist": _FORCING_2000},
    "4.2": {"name": "modelSST-futArcSIC", "namelist": _FORCING_2000},
    # Group 5: Atmosphere-only transient experiments (1979-2014)
    "5.1": {"name": "amip-climSST", "namelist": _FORCING_VARYING},
    "5.2": {"name": "amip-climSIC", "namelist": _FORCING_VARYING},
    # Group 6: Extended partially coupled ocean-atmosphere experiments
    "6.1": {"name": "pa-pdSIC-ext", "namelist": _FORCING_2000},
    "6.2": {"name": "pa-futArcSIC-ext", "namelist": _FORCING_2000},
    "6.3": {"name": "pa-futAntSIC-ext", "namelist": _FORCING_2000},
}

_WRITTEN: set[str] = set()  # configurations already written in this process


def validate_pamip_experiment(experiment: object) -> str | None:
    """Validate the ``pamip_experiment`` option before any file is written.

    Parameters
    ----------
    experiment : object
        The option of the run configuration: the experiment number as text, for
        example ``"1.6"``, or None for the native climatologies. The number must
        be quoted, because YAML reads ``1.10`` as the float 1.1.

    Returns
    -------
    str or None
        The experiment number, or None when the option is not set.

    Raises
    ------
    TypeError
        If the option is not text.
    ValueError
        If the experiment is not a PAMIP experiment, or has no SST and sea-ice
        boundary files in ``templates/pamip_experiments.yaml``.
    """
    if experiment is None or experiment == "":
        return None
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
    """SHiELD namelist settings of the configured experiment, by namelist group.

    Returns
    -------
    dict
        For example ``{"gfs_physics_nml": {"ictm": 20000}}``; empty when
        ``pamip_experiment`` is not set.
    """
    experiment = validate_pamip_experiment(state.pamip_experiment)
    return {} if experiment is None else PAMIP_EXPERIMENTS[experiment]["namelist"]


def radiation_year() -> int | None:
    """Year of the CO2 table the experiment selects with ``ictm = yyyy0``, or None."""
    ictm = pamip_namelist().get("gfs_physics_nml", {}).get("ictm")
    return None if ictm is None else ictm // 10


def _read(
    root: Path, spec: dict, kind: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one PAMIP climatology: SST in kelvin or sea ice as a fraction.

    Returns latitude (ascending), longitude (0..360, ascending) and the data
    ``(12, nlat, nlon)`` ordered January to December, NaN over land.
    """
    variable = spec["variable_id"]
    name = f"{variable}_input4MIPs_SSTsAndSeaIce_PAMIP_{spec['source_id']}_gn_{spec['period']}.nc"
    path = Path(root) / spec["source_id"] / variable / name
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
        da = ds[variable].squeeze(drop=True)
        lat_name = next(d for d in da.dims if d in ("lat", "latitude"))
        lon_name = next(d for d in da.dims if d in ("lon", "longitude"))
        da = da.transpose("time", lat_name, lon_name)
        months = ds["time"].dt.month.values
        if da.shape[0] != 12 or sorted(months) != list(range(1, 13)):
            raise ValueError(
                f"{path.name}: a 12-month climatology is required. Got months {sorted(months)}"
            )
        units = str(da.attrs.get("units", "")).strip().lower()
        data = np.asarray(da.values, dtype=float)[np.argsort(months)]
        lat, lon = (
            ds[lat_name].values.astype(float),
            np.mod(ds[lon_name].values.astype(float), 360.0),
        )
    data[~np.isfinite(data) | (np.abs(data) > 1e10)] = np.nan
    if kind == "sst":
        if units in ("degc", "degree_c", "degrees_c", "degree_celsius", "celsius", "c"):
            data += 273.15
        elif units not in ("k", "kelvin"):
            raise ValueError(f"{path.name}: SST units `{units}` are not degC or K")
        low, high = np.nanmin(data), np.nanmax(data)
        if low < 250.0 or high > 320.0:
            raise ValueError(
                f"{path.name}: SST {low:.1f}..{high:.1f} K is not plausible (units?)"
            )
    else:
        if units in ("%", "percent"):
            data /= 100.0
        elif units not in ("1", "", "fraction", "(0-1)"):
            raise ValueError(
                f"{path.name}: sea-ice units `{units}` are not a fraction or percent"
            )
        low, high = np.nanmin(data), np.nanmax(data)
        if low < -1e-3 or high > 1.0 + 1e-3:
            raise ValueError(
                f"{path.name}: sea ice {low:.3f}..{high:.3f} is outside 0..1 (units?)"
            )
        data = np.clip(data, 0.0, 1.0)
    lat_order, lon_order = np.argsort(lat), np.argsort(lon)
    return lat[lat_order], lon[lon_order], data[:, lat_order][:, :, lon_order]


def _ocean(land_file: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Ocean cells of a lat/lon grid from the 0.5 degree land mask (bilinear, threshold 0.5 as in sfcsub.F).

    The PAMIP files hold filled values over land, which must not mix into coastal ocean values.
    """
    node = open_grib(land_file).leaves[0]
    field = node[next(iter(node.data_vars))]
    field = field.isel({d: 0 for d in field.dims if d not in ("latitude", "longitude")})
    land = (
        field.sortby("latitude")
        .sortby("longitude")
        .interp(latitude=np.clip(lat, -89.75, 89.75), longitude=lon)
    )
    return land.values < 0.5


def _regrid(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of a ``(nlat, nlon)`` field with NaN over land.

    Missing source points are left out and the weights renormalised, so land
    never mixes into ocean values; target points with no valid neighbour (the
    coast of a finer grid) take the value of the nearest valid point.
    """
    ok = np.isfinite(field)
    both = xr.DataArray(
        np.stack([np.where(ok, field, 0.0), ok.astype(float)]),
        dims=("v", "lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    both = xr.concat(  # one extra column on each side closes the longitude circle
        [
            both.isel(lon=[-1]).assign_coords(lon=[lon[-1] - 360.0]),
            both,
            both.isel(lon=[0]).assign_coords(lon=[lon[0] + 360.0]),
        ],
        dim="lon",
    )
    out = both.interp(
        lat=np.clip(tgt_lat, lat[0], lat[-1]), lon=np.mod(tgt_lon, 360.0)
    ).values
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(out[1] > 1e-9, out[0] / out[1], np.nan)
    missing = ~np.isfinite(result)
    if missing.any():
        if missing.all():
            raise ValueError("no valid PAMIP point to interpolate from")
        iy, ix = distance_transform_edt(
            missing, return_distances=False, return_indices=True
        )
        result = result[iy, ix]
    return result


def _month_weights(date: pd.Timestamp) -> tuple[int, int, float, float]:
    """Months (0 = January) and weights of the mid-month linear interpolation of sfcsub.F (``clima``)."""
    rjday = date.dayofyear + date.hour / 24.0
    if rjday < DAYHF[0]:
        rjday += 365.0
    for m in range(12):
        if DAYHF[m] <= rjday < DAYHF[m + 1]:
            w2 = (rjday - DAYHF[m]) / (DAYHF[m + 1] - DAYHF[m])
            return m, (m + 1) % 12, 1.0 - w2, w2
    raise ValueError(f"day {rjday} is outside the monthly interpolation range")


def _fill(field: np.ndarray) -> np.ndarray:
    """Replace NaN (land) by the value of the nearest valid point."""
    missing = ~np.isfinite(field)
    if not missing.any():
        return field
    iy, ix = distance_transform_edt(
        missing, return_distances=False, return_indices=True
    )
    return field[iy, ix]


def _tiles() -> list[int]:
    """Tiles with a surface file: six global tiles, then the nests; one file for regional grids."""
    if state.gtype in ("regional_gfdl", "regional_esg"):
        return [1]
    return list(range(1, 7)) + [7 + i for i in range(state.n_nests)]


def _update_surface(sst: tuple, sic: tuple) -> None:
    """Set the SST and sea ice of the cold-start surface files to the PAMIP fields of the start date.

    The PAMIP climatologies (NaN over land) are interpolated in time with the
    mid-month weights of the surface cycle, regridded onto each tile with
    ``latlon_to_fv3.to_fv3_grid`` and written to ``slmsk``, ``tsea``, ``fice``,
    ``hice`` and ``tisfc`` of ``sfc_data`` over non-land points. The surface
    cycle repeats this at the first time step; the edit makes the files agree
    with it. Rules, after sfcsub.F and the CHGRES_CUBE conventions:

    - ice where the concentration is at least ``AISLIM``, with ``slmsk = 2``;
      below it the point is open water (``slmsk = 0``, ``fice = hice = 0``);
    - open water: ``tsea = tisfc =`` SST, limited to the range of the surface cycle;
    - ice that stays ice keeps its thickness and ice-surface temperature
      (at most the melting point); new ice gets ``SIHNEW`` and ``TGICE``;
    - over ice ``tsea = fice * tisfc + (1 - fice) * TGICE``, the relation of the
      CHGRES_CUBE files, which sfcsub.F inverts to get ``tisfc``;
    - land points are not touched.
    """
    lat, lon = sst[0], sst[1]
    if not (np.array_equal(lat, sic[0]) and np.array_equal(lon, sic[1])):
        raise ValueError("PAMIP SST and sea-ice files are on different grids")
    date = pd.Timestamp(state.init_datetime)
    m1, m2, w1, w2 = _month_weights(date)
    source = {}
    for key, field in (("sst", sst[2]), ("sic", sic[2])):
        f = _fill(w1 * field[m1] + w2 * field[m2])
        # one row at each pole, so tile points poleward of the last row are interpolated
        source[key] = (("lat", "lon"), np.vstack([f[:1], f, f[-1:]]))
    src = xr.Dataset(
        source, coords={"lat": np.concatenate([[-90.0], lat, [90.0]]), "lon": lon}
    )
    log.info(
        f"PAMIP surface files: fields of {date:%Y-%m-%d %H}Z from months {m1 + 1} ({w1:.2f}) and {m2 + 1} ({w2:.2f})"
    )

    backup_dir = Path(state.ic_data) / "perts"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for tile in _tiles():
        filename = sfc_filename(tile)
        path = Path(state.input) / filename
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        out = to_fv3_grid(src, load_grid(filename, tile))
        sst_t, sic_t = (
            out[v].isel(Time=0).transpose("yaxis_1", "xaxis_1").values.astype(float)
            for v in ("sst", "sic")
        )

        with xr.open_dataset(path, decode_cf=False, engine="netcdf4") as handle:
            ds = handle.load()
        slmsk = ds["slmsk"].values[0].copy()
        tsea, fice, hice, tisfc = (
            ds[v].values[0].copy() for v in ("tsea", "fice", "hice", "tisfc")
        )
        was_ice = slmsk == 2
        valid = (slmsk != 1) & np.isfinite(sst_t) & np.isfinite(sic_t)
        missing = int(((slmsk != 1) & ~valid).sum())
        if missing:
            log.warning(
                f"tile {tile}: {missing} non-land points have no PAMIP value and keep their state"
            )

        sst_c = np.clip(sst_t, SST_MIN_K, SST_MAX_K)
        sic_c = np.clip(sic_t, 0.0, 1.0)
        ice = valid & (sic_c >= AISLIM)
        water = valid & ~ice
        stay = ice & was_ice
        new = ice & ~was_ice

        tsea[water] = sst_c[water]
        tisfc[water] = sst_c[water]
        fice[water] = 0.0
        hice[water] = 0.0
        slmsk[water] = 0.0

        slmsk[ice] = 2.0
        fice[ice] = sic_c[ice]
        hice[new] = SIHNEW
        hice[stay] = np.where(hice[stay] > 0.0, hice[stay], SIHNEW)
        tisfc[new] = TGICE
        tisfc[stay] = np.minimum(tisfc[stay], T_MELT)
        tsea[ice] = fice[ice] * tisfc[ice] + (1.0 - fice[ice]) * TGICE

        for v, a in (
            ("slmsk", slmsk),
            ("tsea", tsea),
            ("fice", fice),
            ("hice", hice),
            ("tisfc", tisfc),
        ):
            ds[v] = ds[v].copy(data=a[np.newaxis].astype(ds[v].dtype))
            ds[v].attrs.pop(
                "checksum", None
            )  # the data no longer match an FMS checksum
        backup = backup_dir / f"{filename.stem}.pamip.original.nc"
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_name(path.name + ".pamip.tmp")
        ds.to_netcdf(temporary, engine="netcdf4")
        ds.close()
        temporary.replace(path)


def _write(template: Path, destination: Path, source: tuple, kind: str) -> np.ndarray:
    """Write the regridded climatology through the messages of the native file.

    SST replaces every value of the template (missing points stay missing); for
    sea ice the points above 1 (the 1.57 land flags) keep their value.

    Returns
    -------
    numpy.ndarray
        The values written, ``(12, nlat, nlon)`` in the month order of the template.
    """
    tree = open_grib(template)
    node = tree.leaves[0]
    name = next(iter(node.data_vars))
    field = node[name]
    months = field.time.dt.month.values
    if field.shape[:2] != (12, 1) or sorted(months) != list(range(1, 13)):
        raise ValueError(
            f"{template.name}: expected twelve monthly fields on one level"
        )
    lat, lon = field.latitude.values, field.longitude.values
    lat_s, lon_s, data_s = source
    data = field.values.copy()
    for i, month in enumerate(months):
        new = _regrid(data_s[int(month) - 1], lat_s, lon_s, lat, lon)
        old = data[i, 0]
        if kind == "sst":
            data[i, 0] = np.where(np.isnan(old), np.nan, new)
        else:
            data[i, 0] = np.where(
                np.isnan(old) | (old > LAND_FLAG), old, np.clip(new, 0.0, 1.0)
            )
    node[name] = field.copy(data=data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)  # never write through a link into fix_src
    save_grib(tree, destination, template)
    if len(open_grib(destination).attrs["messages"]) != 12:
        raise ValueError(f"{destination.name}: expected twelve messages")
    return data[:, 0]


def apply_pamip_experiment() -> bool:
    """Write the SST and sea-ice climatologies of the PAMIP experiment for the case.

    Uses ``state.pamip_experiment``. At the cold start it also sets the SST and sea ice of the
    surface files in ``state.input`` (``_update_surface``). The files go to
    ``FIXED/<family>`` (``FIXED/PAMIP``) as ``RTGSST.<family>_<exp>.monthly.clim.grb`` and
    ``CFSR.SEAICE.<family>_<exp>.monthly.clim.grb`` and their paths are stored in
    ``state.fntsfc`` and ``state.fnaisc``, which ``update_namsfc`` writes to
    namsfc. ``fix_src`` is only read.

    Returns
    -------
    bool
        True when the files are written (or were already written in this process
        for the same configuration), False when no experiment is configured.

    Raises
    ------
    FileNotFoundError
        If a native or PAMIP file is missing.
    ValueError
        If an input or an output fails its checks.
    """
    experiment = validate_pamip_experiment(state.pamip_experiment)
    if experiment is None:
        return False
    key = f"{state.fix}|{experiment}"
    if key in _WRITTEN:  # update_namsfc runs once per domain; write once
        return True

    entry = yaml.safe_load(REGISTRY_FILE.read_text())["experiments"][experiment]
    family = entry[
        "family"
    ]  # PAMIP: directory below FIXED and fix_src, and tag of the file names
    out_dir = Path(state.fix) / family
    am_dir, pamip_dir = Path(state.fix_src) / "am", Path(state.fix_src) / family
    sst = _read(pamip_dir, entry["sst"], "sst")
    sic = _read(pamip_dir, entry["sic"], "sic")
    ocean = _ocean(am_dir / LAND_FILE, sst[0], sst[1])
    sst[2][:, ~ocean] = np.nan
    sic[2][:, ~ocean] = np.nan
    log.info(
        f"PAMIP experiment {experiment} ({PAMIP_EXPERIMENTS[experiment]['name']}): SST {entry['sst']['source_id']}, SIC {entry['sic']['source_id']}"
    )

    # Ice-edge consistency of the pair (PAMIP sets the SST of changed ice points).
    if sst[2].shape == sic[2].shape:
        ice = sic[2] >= AISLIM
        warm = float((ice & (sst[2] > 275.15)).sum() / max(ice.sum(), 1))
        if warm > 0.02:
            log.warning(
                f"{warm:.1%} of the PAMIP sea-ice points have SST above 2 degC; check that SST and SIC belong together"
            )

    names = {
        "sst": SST_FILE.replace("1982.2012", f"{family}_{experiment}"),
        "sic": ICE_FILE.replace("1982.2012", f"{family}_{experiment}"),
    }
    sst_out = _write(am_dir / SST_FILE, out_dir / names["sst"], sst, "sst")
    sic_out = _write(am_dir / ICE_FILE, out_dir / names["sic"], sic, "sic")

    if (
        np.nanmin(sst_out) < SST_MIN_K - GRIB_TOLERANCE
        or np.nanmax(sst_out) > SST_MAX_K + GRIB_TOLERANCE
    ):
        raise ValueError(
            f"SST {np.nanmin(sst_out):.2f}..{np.nanmax(sst_out):.2f} K leaves the {SST_MIN_K}..{SST_MAX_K} K range of the surface cycle"
        )
    ocean = sic_out[sic_out <= LAND_FLAG]
    if (
        np.isnan(ocean).any()
        or ocean.min() < -GRIB_TOLERANCE
        or ocean.max() > 1.0 + GRIB_TOLERANCE
    ):
        raise ValueError(
            "Sea-ice concentration outside 0..1 or missing in the written file"
        )
    log.info(
        f"Wrote {out_dir / names['sst']} and {out_dir / names['sic']}; SST {np.nanmin(sst_out):.2f}..{np.nanmax(sst_out):.2f} K"
    )

    # update_namsfc writes these paths to namsfc FNTSFC and FNAISC.
    state.fntsfc = f"FIXED/{family}/{names['sst']}"
    state.fnaisc = f"FIXED/{family}/{names['sic']}"
    if (
        state.restart_no == 0
    ):  # later segments continue from the cycled state of the model
        _update_surface(sst, sic)
    _WRITTEN.add(key)
    return True
