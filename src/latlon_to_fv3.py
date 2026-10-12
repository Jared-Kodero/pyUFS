"""Regridding of regular lat/lon fields onto the tiles of the FV3 surface files.

Shared by ``sm_perturbations`` (soil moisture climatologies) and
``pamip_experiment`` (SST and sea ice). The target grid is the ``geolat`` and
``geolon`` of a tile of the cold-start surface files in ``state.input``
(global ``sfc_data.tileN.nc``, nests ``sfc_data.nestNN.tileN.nc``, regional
``sfc_data.nc``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
import xesmf as xe

from fv3_state import state


def sfc_filename(tile: int) -> Path:
    """Surface file of a tile: global sfc_data.tileN.nc, nest
    sfc_data.nestNN.tileN.nc, regional sfc_data.nc (single tile, no suffix)."""
    if state.gtype in ("regional_gfdl", "regional_esg"):
        return Path("sfc_data.nc")
    nest_idx = f"nest{(tile - 5):02d}." if tile > 6 else ""
    return Path(f"sfc_data.{nest_idx}tile{tile}.nc")


def _is_global(lon: np.ndarray) -> bool:
    """True when the 1-D longitudes of a regular grid close the circle."""
    lon = np.sort(np.asarray(lon, dtype=float))
    if lon.size < 2:
        return False
    return float(lon[-1] - lon[0] + np.median(np.diff(lon))) > 359.0


def to_fv3_grid(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray,
    method: Literal["bilinear", "conservative"] = "bilinear",
) -> xr.Dataset:
    """
    Remap ll grid to c-grid using xesmf
    interpolation. Regridding weights are recomputed on every call.

    The longitude circle is closed when the source grid is global, so tile
    points between the last and the first source column are interpolated
    instead of left missing.
    """

    ll_grid = xr.Dataset(
        {
            "lat": grid_in["lat"],
            "lon": grid_in["lon"],
        }
    )

    c_grid = xr.Dataset(
        {
            "lat": grid_out["geolat"],
            "lon": grid_out["geolon"],
        }
    )

    regridder = xe.Regridder(
        ll_grid,
        c_grid,
        method=method,
        periodic=_is_global(grid_in["lon"].values),
    )

    # init dims ('Time', 'yaxis_1', 'xaxis_1', 'zaxis_1')
    # restart dims ('Time', 'yaxis_1', 'xaxis_1', 'zaxis_1', 'zaxis_2', 'zaxis_3')

    # skipna renormalizes the weights over valid source points; a target
    # point with less than half of its weight on valid points becomes NaN and
    # keeps its original value in the caller.
    out = regridder(grid_in, skipna=True, na_thres=0.5)

    if "Time" not in out.coords:
        out = out.expand_dims("Time", axis=0)

    out.attrs = grid_out.attrs
    out_coords = set(out.coords)
    for c in out_coords:
        if c in grid_out.coords:
            out[c].attrs = grid_out[c].attrs
        else:
            out = out.drop_vars(c)

    out["Time"] = grid_out["Time"]

    new_dims = [d for d in grid_out.dims if d in out.dims]
    out = out.transpose(*new_dims)

    out.attrs = grid_out.attrs

    return out


def load_grid(filename: Path, tile: int) -> xr.Dataset:
    """Coordinates (geolat, geolon and the dimension coordinates) of a tile.

    Read from the surface file once and cached in ``<ic_data>/perts``.
    """
    grid_file = Path(state.ic_data) / "perts" / f"tile.{tile}.grid.nc"
    if not grid_file.exists():
        if state.restart_no == 0:
            path = Path(state.input) / filename
        else:
            path = Path(state.ic_data) / "INPUT" / filename
        with xr.open_dataset(path, decode_cf=False, engine="netcdf4") as ds:
            # Select coordinate variables only. `ds.dims` returns dimension names,
            # which are not necessarily variables and raise KeyError on selection.
            keep = ["geolat", "geolon"] + list(ds.coords)
            keep = [k for k in dict.fromkeys(keep) if k in ds.variables]
            ds = ds[keep].load()
            grid_file.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(grid_file, engine="netcdf4")

    ds = xr.open_dataset(grid_file, decode_cf=False, engine="netcdf4").load()
    return ds
