"""Apply mean and meridional-gradient SST anomalies to open-water climatology."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

log = logging.getLogger("PLUGINS")

AISLIM = 0.01  # Effectively ice-free, rather than the 15% sea-ice-extent threshold.
SST_MIN_K, SST_MAX_K = 271.2, 313.0
TINY = 1e-9


def _ramp(edge: xr.DataArray, taper: float) -> xr.DataArray:
    """Smoothly rise from 0 at the boundary to 1 a `taper` distance inside."""
    fraction = (edge / max(taper, TINY)).clip(0, 1)
    return xr.where(edge >= 0, 0.5 * (1 - np.cos(np.pi * fraction)), 0.0)


def perturb_sst(
    sst: xr.DataArray,
    sic: xr.DataArray,
    land: xr.DataArray,
    *,
    delta: float,
    months: list[int],
    lat: list[float],
    lon: list[float] | None,
    taper: float,
    gradient_k_per_degree: float = 0.0,
    seasonal_weights: list[float] | None = None,
) -> xr.DataArray:
    """Perturb ice-free SST by a mean anomaly and a zero-area-mean meridional tilt.

    Positive `gradient_k_per_degree` warms the northern part relative to the
    southern part. It therefore weakens an initially negative SST gradient.
    The gradient component has zero cosine-latitude-weighted spatial mean in
    each month, after geographic tapering and masking. `delta` independently
    controls the uniform-in-the-box component. `seasonal_weights`, if supplied,
    gives 12 monthly amplitudes in place of the binary `months` selection.
    """
    weight = _ramp(np.minimum(sst.lat - lat[0], lat[1] - sst.lat), taper)
    if lon is not None and (lon[1] - lon[0]) % 360:
        east = (sst.lon - lon[0]) % 360
        weight = weight * _ramp(np.minimum(east, (lon[1] - lon[0]) % 360 - east), taper)
    weight = weight * ((land < 0.5) & (sic <= AISLIM))

    area = weight * np.cos(np.deg2rad(sst.lat))
    area_total = area.sum(("lat", "lon"))
    mean_lat = xr.where(
        area_total > 0,
        (area * sst.lat).sum(("lat", "lon")) / area_total,
        0.0,
    )
    anomaly = weight * (delta + gradient_k_per_degree * (sst.lat - mean_lat))

    if seasonal_weights is None:
        seasonal = sst.month.isin(months).astype(float)
    else:
        if len(seasonal_weights) != 12:
            raise ValueError("seasonal_weights must contain January through December")
        seasonal = xr.DataArray(
            seasonal_weights,
            dims=("month",),
            coords={"month": np.arange(1, 13)},
        ).sel(month=sst.month)
    anomaly = anomaly * seasonal
    new = sst + anomaly

    out_of_bounds = (abs(anomaly) > TINY) & ((new < SST_MIN_K) | (new > SST_MAX_K))
    if bool(out_of_bounds.any()):
        raise ValueError(
            "The imposed SST anomaly exceeds the prescribed temperature bounds; "
            "reduce its amplitude or change the forced region. Clipping would "
            "change the gradient and violate zero-mean compensation."
        )
    return new.transpose(*sst.dims)


def _monthly(open_grib, path: Path):
    tree = open_grib(path)
    node = tree.leaves[0]
    variable = next(iter(node.data_vars))
    native = node[variable]
    field = (
        native.squeeze(drop=True)
        .assign_coords(month=native.time.dt.month)
        .swap_dims(time="month")
        .rename(latitude="lat", longitude="lon")
        .load()
    )
    return tree, node, variable, native, field


def _on_grid(field: xr.DataArray, grid: xr.DataArray, ll_regrid) -> xr.DataArray:
    """field on the lat/lon grid of `grid` (which has longitudes in 0 to 360)."""
    lon = field.lon % 360.0
    same = all(
        a.size == b.size and np.allclose(a, b, atol=1e-3)
        for a, b in ((field.lat, grid.lat), (lon, grid.lon))
    )
    if not same:
        field = ll_regrid(field.assign_coords(lon=lon).sortby(["lat", "lon"]), grid)
    return field.assign_coords(lat=grid.lat, lon=grid.lon)


def write_perturbed_sst(
    config: dict,
    *,
    state: Any,
    namsfc: dict,
    open_grib: Callable,
    save_grib: Callable,
    ll_regrid: Callable,
) -> str:
    """Write a new SST climatology without changing the original source."""
    case = Path(state.case_dir)
    source = case / namsfc["fntsfc"]
    tree, node, variable, native, sst = _monthly(open_grib, source)
    sst = sst.assign_coords(lon=sst.lon % 360.0)
    sic = _on_grid(_monthly(open_grib, case / namsfc["fnaisc"])[-1], sst, ll_regrid)
    mask = open_grib(case / namsfc["fnmskh"]).leaves[0]
    land = next(iter(mask.data_vars.values())).squeeze(drop=True)
    land = _on_grid(land.rename(latitude="lat", longitude="lon").load(), sst, ll_regrid)

    settings = {k: config[k] for k in ("delta", "months", "lat", "lon", "taper")}
    settings["gradient_k_per_degree"] = config.get("gradient_k_per_degree", 0.0)
    settings["seasonal_weights"] = config.get("seasonal_weights")
    new = perturb_sst(sst, sic, land, **settings)
    node[variable] = native.copy(data=new.values[:, np.newaxis])

    name = Path(namsfc["fntsfc"]).with_suffix("").name + ".sstpert.grb"
    destination = Path(state.fix) / "sst_perturbation" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    save_grib(tree, destination, source)
    log.info(
        "SST climatology perturbed, largest change %.2f K: %s",
        float(abs(new - sst).max()),
        destination,
    )
    return f"FIXED/sst_perturbation/{name}"
