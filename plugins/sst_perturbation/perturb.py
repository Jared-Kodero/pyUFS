"""Add an SST anomaly to the open-water points of a monthly SST climatology."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from grib_io import open_grib, save_grib
from latlon_to_fv3 import ll_regrid

log = logging.getLogger("PLUGINS")

AISLIM = 0.15
SST_MIN_K, SST_MAX_K = 271.2, 313.0
TINY = 1e-9


def _ramp(edge: xr.DataArray, taper: float) -> xr.DataArray:
    """0 outside the box, rising linearly to 1 at `taper` degrees inside it."""
    return xr.where(edge >= 0, (edge / max(taper, TINY)).clip(0, 1), 0.0)


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
) -> xr.DataArray:
    """sst + delta * weight, with the weight 1 in the box and months, tapered at the box
    edge, and zero at land (land mask >= 0.5) and at ice (sea-ice fraction > AISLIM)."""
    weight = _ramp(np.minimum(sst.lat - lat[0], lat[1] - sst.lat), taper)
    if lon is not None and (lon[1] - lon[0]) % 360:
        east = (sst.lon - lon[0]) % 360
        weight = weight * _ramp(np.minimum(east, (lon[1] - lon[0]) % 360 - east), taper)
    weight = weight * ((land < 0.5) & (sic <= AISLIM)) * sst.month.isin(months)
    new = (sst + delta * weight).clip(SST_MIN_K, SST_MAX_K)
    return xr.where(weight > 0, new, sst).transpose(*sst.dims)


def _monthly(path: Path):
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


def _on_grid(field: xr.DataArray, grid: xr.DataArray) -> xr.DataArray:
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
) -> str:
    """Write the perturbed SST climatology to FIXED/sst_perturbation and return the
    namsfc path of the new file. The source is never modified."""
    case = Path(state.case_dir)
    source = case / namsfc["fntsfc"]
    tree, node, variable, native, sst = _monthly(source)
    sst = sst.assign_coords(lon=sst.lon % 360.0)
    sic = _on_grid(_monthly(case / namsfc["fnaisc"])[-1], sst)
    mask = open_grib(case / namsfc["fnmskh"]).leaves[0]
    land = next(iter(mask.data_vars.values())).squeeze(drop=True)
    land = _on_grid(land.rename(latitude="lat", longitude="lon").load(), sst)

    settings = {k: config[k] for k in ("delta", "months", "lat", "lon", "taper")}
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
