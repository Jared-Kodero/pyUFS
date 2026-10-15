"""Regridding of regular lat/lon fields onto the tiles of the FV3 surface files.

Shared by ``sm_perturbations`` (soil moisture climatologies) and
``pamip_experiment`` (SST and sea ice): ``to_fv3_grid`` regrids lat/lon fields onto
the tiles and ``ll_regrid`` (xESMF) lat/lon fields onto other lat/lon grids. The tile grid is the ``geolat`` and
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


def sfc_filename(tile: int) -> Path:
    """Surface file of a tile: global sfc_data.tileN.nc, nest
    sfc_data.nestNN.tileN.nc, regional sfc_data.nc (single tile, no suffix)."""

    from fv3_state import state

    if state.gtype in ("regional_gfdl", "regional_esg"):
        return Path("sfc_data.nc")
    nest_idx = f"nest{(tile - 5):02d}." if tile > 6 else ""
    return Path(f"sfc_data.{nest_idx}tile{tile}.nc")


def ll_regrid(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray | None = None,
    x_cood: str = "lon",
    y_cood: str = "lat",
    method: Literal[
        "bilinear",
        "conservative",
        "conservative_normed",
        "patch",
        "nearest_s2d",
        "nearest_d2s",
    ] = "bilinear",
    unmapped_to_nan: bool = True,
) -> xr.Dataset | xr.DataArray:
    """Regrid source data using xESMF with NaN-aware interpolation."""

    import xesmf as xe

    if grid_out is None:
        return grid_in

    for grid in (grid_in, grid_out):
        for coord in (y_cood, x_cood):
            if coord not in grid.dims:
                raise ValueError(f"Grid must contain {coord!r} dimension.")

    ll_in = xr.Dataset(
        coords={
            "lat": (y_cood, grid_in[y_cood].values),
            "lon": (x_cood, grid_in[x_cood].values),
        }
    )

    ll_out = xr.Dataset(
        coords={
            "lat": (y_cood, grid_out[y_cood].values),
            "lon": (x_cood, grid_out[x_cood].values),
        }
    )

    if isinstance(grid_in, xr.DataArray):
        chunked = grid_in if grid_in.chunks is not None else None
    else:
        chunked = next(
            (
                var
                for var in grid_in.data_vars.values()
                if var.chunks is not None and y_cood in var.dims and x_cood in var.dims
            ),
            None,
        )

    output_chunks = None
    if chunked is not None:
        output_chunks = {
            coord: min(chunked.chunksizes[coord][0], grid_out.sizes[coord])
            for coord in (y_cood, x_cood)
        }

    regridder = xe.Regridder(
        ll_in,
        ll_out,
        method=method,
        periodic=True,
        unmapped_to_nan=unmapped_to_nan,
        input_dims=(y_cood, x_cood),
        output_dims=(y_cood, x_cood),
    )

    return regridder(
        grid_in,
        skipna=True,
        keep_attrs=True,
        output_chunks=output_chunks,
    )


def fill_nearest(da: xr.DataArray, limit: float = float("inf")) -> xr.DataArray:
    """Fill NaN with the nearest valid cell (in cells) of the same lat/lon field, up to ``limit`` cells."""

    from scipy.ndimage import distance_transform_edt

    def fill(a: np.ndarray) -> np.ndarray:
        dist, (iy, ix) = distance_transform_edt(np.isnan(a), return_indices=True)
        return np.where(dist <= limit, a[iy, ix], np.nan)

    return xr.apply_ufunc(
        fill,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
    )


def to_fv3_grid(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray,
    method: Literal["bilinear", "conservative"] = "bilinear",
    periodic: bool = False,
) -> xr.Dataset:
    """
    Remap ll grid to c-grid using xesmf
    interpolation. Regridding weights are recomputed on every call.
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
        periodic=periodic,
    )

    # init dims ('Time', 'yaxis_1', 'xaxis_1', 'zaxis_1')
    # restart dims ('Time', 'yaxis_1', 'xaxis_1', 'zaxis_1', 'zaxis_2', 'zaxis_3')

    out = regridder(grid_in, skipna=True)

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
    from fv3_state import state

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
