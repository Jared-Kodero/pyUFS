"""Regridding of regular lat/lon fields onto the tiles of the FV3 surface files.

Shared by ``sm_perturbations`` (soil moisture climatologies) and
``pamip_experiment`` (SST and sea ice): ``to_fv3_grid`` regrids lat/lon fields onto
the tiles and ``ll_regrid`` lat/lon fields onto other lat/lon grids. The tile grid is the ``geolat`` and
``geolon`` of a tile of the cold-start surface files in ``state.input``
(global ``sfc_data.tileN.nc``, nests ``sfc_data.nestNN.tileN.nc``, regional
``sfc_data.nc``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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


def ll_regrid(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray,
    periodic: bool = False,
    na_thres: float = 0.5,
) -> xr.Dataset | xr.DataArray:
    """Bilinear regridding of a regular lat/lon field onto another regular lat/lon grid.

    ``grid_out`` has 1-D ``lat`` and ``lon``. Missing source points (NaN) are left
    out and the weights renormalised; a target point whose NaN share of weight
    exceeds ``na_thres`` is NaN. ``periodic=True`` closes the longitude circle of a
    global source grid. It uses ``xarray`` interpolation, not xESMF, which runs out
    of memory on fine regular grids such as the 0.083 degree SST file (9.3 million
    points); for tiles use ``to_fv3_grid``.
    """
    if periodic:
        first, last = float(grid_in["lon"][0]), float(grid_in["lon"][-1])
        grid_in = xr.concat(
            [
                grid_in.isel(lon=[-1]).assign_coords(lon=[last - 360.0]),
                grid_in,
                grid_in.isel(lon=[0]).assign_coords(lon=[first + 360.0]),
            ],
            dim="lon",
        )
    target = {"lat": grid_out["lat"], "lon": grid_out["lon"]}
    weight = grid_in.notnull().astype(float).interp(**target)
    value = grid_in.fillna(0.0).interp(**target)
    out = value / weight.where(
        weight < 1.0 - 1e-9, 1.0
    )  # full weight: no rounding of the ratio
    return out.where((weight >= 1.0 - na_thres) & (weight > 0.0))


def to_fv3_grid(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray,
    method: Literal["bilinear", "conservative"] = "bilinear",
    periodic: bool = False,
    na_thres: float = 0.5,
) -> xr.Dataset | xr.DataArray:
    """Regrid a regular lat/lon field onto a tile of the surface files (``geolat``/``geolon``) with xESMF.

    The result is formatted like the tile, with a ``Time`` axis. Missing source
    points (NaN) are left out and the weights renormalised; a target point whose
    NaN share of weight exceeds ``na_thres`` is NaN. ``periodic=True`` closes the
    longitude circle of a global source grid; weights are recomputed on every call.
    """
    regridder = xe.Regridder(
        xr.Dataset({"lat": grid_in["lat"], "lon": grid_in["lon"]}),
        xr.Dataset({"lat": grid_out["geolat"], "lon": grid_out["geolon"]}),
        method=method,
        periodic=periodic,
    )
    out = regridder(grid_in, skipna=True, na_thres=na_thres)

    # init dims ('Time', 'yaxis_1', 'xaxis_1', 'zaxis_1')
    # restart dims ('Time', 'yaxis_1', 'xaxis_1', 'zaxis_1', 'zaxis_2', 'zaxis_3')
    if "Time" not in out.coords:
        out = out.expand_dims("Time", axis=0)
    for c in set(out.coords):
        if c in grid_out.coords:
            out[c].attrs = grid_out[c].attrs
        else:
            out = out.drop_vars(c)
    out["Time"] = grid_out["Time"]
    out = out.transpose(*[d for d in grid_out.dims if d in out.dims])
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
