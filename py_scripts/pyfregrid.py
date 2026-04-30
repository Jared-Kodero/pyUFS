"""Python reimplementation of fregrid using xarray, dask, and xESMF.

This script mirrors the original fregrid command-line interface as closely as
possible while using Python-native data handling and interpolation.
"""

from __future__ import annotations

import inspect
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import xesmf as xe

warnings.filterwarnings("ignore", message=".*F_CONTIGUOUS.*", module="xesmf.backend")

_tiles_type: Literal["global", "nest"] = None

log = logging.getLogger("REGRIDING")


@dataclass
class MosaicTile:
    tile_name: str
    grid_path: Path
    nx: int
    ny: int
    lon_t: np.ndarray
    lat_t: np.ndarray
    lon_c: np.ndarray
    lat_c: np.ndarray
    lon1d_t: np.ndarray
    lat1d_t: np.ndarray
    lon1d_c: np.ndarray
    lat1d_c: np.ndarray


def g_fargs(func) -> dict:
    """
    Get the signature of a function as a dictionary.
    """
    sig = inspect.signature(func)
    params = {}

    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            params[name] = None
        else:
            params[name] = param.default

    return params


def _clip_lat(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values), -90.0, 90.0)
    return clipped


def _is_rectilinear_latlon(tile: MosaicTile, atol: float = 1e-10) -> bool:
    lon_ok = np.allclose(tile.lon_t, tile.lon_t[0:1, :], rtol=0.0, atol=atol)
    lat_ok = np.allclose(tile.lat_t, tile.lat_t[:, 0:1], rtol=0.0, atol=atol)
    return bool(lon_ok and lat_ok)


def _nc_base(name: str) -> str:
    return name[:-3] if name.endswith(".nc") else name


def _ensure_nc(path: str) -> str:
    return path if path.endswith(".nc") else f"{path}.nc"


def _char_array_to_list(da: xr.DataArray) -> List[str]:
    arr = da.values
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    items: List[str] = []
    for row in arr:
        if row.dtype.kind in {"S", "U"}:
            text = b"".join(
                part
                if isinstance(part, (bytes, bytearray))
                else str(part).encode("ascii", "ignore")
                for part in row
            ).decode("ascii", "ignore")
        else:
            text = "".join(chr(int(v)) for v in row)
        items.append(text.strip().rstrip("\x00"))
    return items


def _infer_axis(ds: xr.Dataset, dim: str) -> str:
    if dim in ds and "cartesian_axis" in ds[dim].attrs:
        return str(ds[dim].attrs["cartesian_axis"]).upper()
    d = dim.lower()
    if "time" in d or d in {"t", "time", "times"}:
        return "T"
    if d.startswith("z") or "lev" in d or "depth" in d:
        return "Z"
    if d.startswith("x") or "lon" in d:
        return "X"
    if d.startswith("y") or "lat" in d:
        return "Y"
    if d.startswith("n"):
        return "N"
    return "?"


def _find_axis_dim(ds: xr.Dataset, da: xr.DataArray, axis: str) -> Optional[str]:
    for dim in da.dims:
        if _infer_axis(ds, dim) == axis:
            return dim
    return None


def _load_mosaic_tiles(mosaic_path: Path) -> List[MosaicTile]:
    with xr.open_dataset(mosaic_path, decode_cf=False) as ds:
        if "gridfiles" not in ds:
            raise ValueError(f"mosaic file {mosaic_path} does not contain gridfiles")
        gridfiles = _char_array_to_list(ds["gridfiles"])
        if "gridtiles" in ds:
            tile_names = _char_array_to_list(ds["gridtiles"])
        else:
            tile_names = [f"tile{i + 1}" for i in range(len(gridfiles))]

    if len(tile_names) != len(gridfiles):
        raise ValueError("mosaic gridtiles and gridfiles lengths do not match")

    tiles: List[MosaicTile] = []
    for tile_name, grid_rel in zip(tile_names, gridfiles):
        grid_path = (mosaic_path.parent / grid_rel).resolve()
        with xr.open_dataset(grid_path, decode_cf=False) as gds:
            if "x" not in gds or "y" not in gds:
                raise ValueError(f"grid file {grid_path} must contain x and y")
            x = np.asarray(gds["x"].values)
            y = np.asarray(gds["y"].values)
            if x.ndim != 2 or y.ndim != 2:
                raise ValueError(f"grid file {grid_path}: x and y must be 2-D")
            ny_super, nx_super = x.shape
            nx = (nx_super - 1) // 2
            ny = (ny_super - 1) // 2
            if 2 * nx + 1 != nx_super or 2 * ny + 1 != ny_super:
                raise ValueError(
                    f"grid file {grid_path}: supergrid shape must be odd in both dimensions"
                )

            lon_c = x[0::2, 0::2]
            lat_c = _clip_lat(y[0::2, 0::2])
            lon_t = x[1::2, 1::2]
            lat_t = _clip_lat(y[1::2, 1::2])

            lon1d_t = x[1, 1::2]
            lat1d_t = _clip_lat(y[1::2, 1])
            lon1d_c = x[0, 0::2]
            lat1d_c = _clip_lat(y[0::2, 0])

        tiles.append(
            MosaicTile(
                tile_name=tile_name,
                grid_path=grid_path,
                nx=nx,
                ny=ny,
                lon_t=lon_t,
                lat_t=lat_t,
                lon_c=lon_c,
                lat_c=lat_c,
                lon1d_t=lon1d_t,
                lat1d_t=lat1d_t,
                lon1d_c=lon1d_c,
                lat1d_c=lat1d_c,
            )
        )

    return tiles


def _regular_latlon_tile(
    lon_begin: float,
    lon_end: float,
    lat_begin: float,
    lat_end: float,
    nlon: int,
    nlat: int,
    center_y: bool,
) -> MosaicTile:
    if nlon <= 0 or nlat <= 0:
        raise ValueError(
            "nlon and nlat must be positive when output_mosaic is not supplied"
        )
    lat_begin = float(np.clip(lat_begin, -90.0, 90.0))
    lat_end = float(np.clip(lat_end, -90.0, 90.0))

    if lon_end <= lon_begin or lat_end <= lat_begin:
        raise ValueError("lonEnd must be > lonBegin and latEnd must be > latBegin")

    dlon = (lon_end - lon_begin) / nlon
    lon1d_t = lon_begin + (np.arange(nlon) + 0.5) * dlon
    lon1d_c = lon_begin + np.arange(nlon + 1) * dlon

    if center_y:
        dlat = (lat_end - lat_begin) / nlat
        lat1d_t = lat_begin + (np.arange(nlat) + 0.5) * dlat
        lat1d_c = lat_begin + np.arange(nlat + 1) * dlat
    else:
        if nlat == 1:
            raise ValueError("nlat must be > 1 when center_y is not set")
        dlat = (lat_end - lat_begin) / (nlat - 1)
        lat1d_t = lat_begin + np.arange(nlat) * dlat
        lat1d_c = lat_begin + (np.arange(nlat + 1) - 0.5) * dlat

    lat1d_t = _clip_lat(lat1d_t)
    lat1d_c = _clip_lat(lat1d_c)

    lon_t, lat_t = np.meshgrid(lon1d_t, lat1d_t)
    lon_c, lat_c = np.meshgrid(lon1d_c, lat1d_c)

    return MosaicTile(
        tile_name="tile1",
        grid_path=Path("<generated_latlon_grid>"),
        nx=nlon,
        ny=nlat,
        lon_t=lon_t,
        lat_t=lat_t,
        lon_c=lon_c,
        lat_c=lat_c,
        lon1d_t=lon1d_t,
        lat1d_t=lat1d_t,
        lon1d_c=lon1d_c,
        lat1d_c=lat1d_c,
    )


def _tile_to_regridder_grid(tile: MosaicTile) -> xr.Dataset:
    return xr.Dataset(
        {
            "lon": (("y", "x"), tile.lon_t),
            "lat": (("y", "x"), np.clip(tile.lat_t, -90.0, 90.0)),
            "lon_b": (("y_b", "x_b"), tile.lon_c),
            "lat_b": (("y_b", "x_b"), np.clip(tile.lat_c, -90.0, 90.0)),
        }
    )


def _tile_file_paths(
    base_name: str, directory: Path, tile_names: Sequence[str]
) -> List[Path]:
    base = _nc_base(str(base_name))
    if len(tile_names) == 1:
        names = [f"{base}.nc"]
    else:
        names = [f"{base}.{tile}.nc" for tile in tile_names]
    return [(directory / n).resolve() for n in names]


def _mosaic_grid_paths(mosaic_path: Path) -> List[Path]:
    with xr.open_dataset(mosaic_path, decode_cf=False) as ds:
        if "gridfiles" not in ds:
            raise ValueError(f"mosaic file {mosaic_path} does not contain gridfiles")
        gridfiles = _char_array_to_list(ds["gridfiles"])
    return [(mosaic_path.parent / grid_rel).resolve() for grid_rel in gridfiles]


def _read_mosaic_ncontacts(mosaic_path: Path) -> int:
    with xr.open_dataset(mosaic_path, decode_cf=False) as ds:
        if "contacts" in ds:
            var = ds["contacts"]
            if var.ndim == 0:
                return 0
            return int(var.shape[0])
        if "ncontact" in ds.sizes:
            return int(ds.sizes["ncontact"])
    return 0


def _coerce_int_flag(value: object) -> int:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("ascii", "ignore")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "false", "f", "no", "n"):
            return 0
        if text in ("true", "t", "yes", "y"):
            return 1
        try:
            return int(float(text))
        except ValueError:
            return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _grid_great_circle_algorithm(grid_path: Path) -> int:
    with xr.open_dataset(grid_path, decode_cf=False) as ds:
        if "tile" in ds and "great_circle_algorithm" in ds["tile"].attrs:
            return _coerce_int_flag(ds["tile"].attrs.get("great_circle_algorithm"))
        if "great_circle_algorithm" in ds.attrs:
            return _coerce_int_flag(ds.attrs.get("great_circle_algorithm"))
        if "great_circle_algorithm" in ds:
            val = ds["great_circle_algorithm"].values
            arr = np.asarray(val)
            if arr.size == 0:
                return 0
            return _coerce_int_flag(arr.reshape(-1)[0])
    return 0


def _mosaic_great_circle_algorithm(
    mosaic_path, in_gca=None, interp_method=None, method=None, u_field=None
) -> int:

    if mosaic_path:
        grid_paths = _mosaic_grid_paths(mosaic_path)
        if not grid_paths:
            return 0
        flags = [_grid_great_circle_algorithm(path) for path in grid_paths]
        first = flags[0]
        if any(flag != first for flag in flags[1:]):
            raise ValueError(
                f"inconsistent great_circle_algorithm values across grid tiles in {mosaic_path}"
            )
    elif not mosaic_path and in_gca is not None:
        first = 0

    if in_gca is not None:
        if in_gca or first:
            if interp_method != "conserve_order1":
                raise ValueError(
                    "when great_circle_algorithm is active, interp_method must be conserve_order1"
                )
            if method == "bilinear":
                raise ValueError(
                    "bilinear interpolation is not supported when great_circle_algorithm is active"
                )
            if u_field:
                raise ValueError(
                    "vector interpolation is not supported when great_circle_algorithm is active"
                )

    return first


def _select_and_slice(
    da: xr.DataArray,
    ds: xr.Dataset,
    KlevelBegin: int,
    KlevelEnd: int,
    LstepBegin: int,
    LstepEnd: int,
) -> xr.DataArray:
    zdim = _find_axis_dim(ds, da, "Z")
    if (KlevelBegin is not None or KlevelEnd is not None) and zdim is not None:
        k0 = 0 if KlevelBegin is None else KlevelBegin - 1
        k1 = da.sizes[zdim] if KlevelEnd is None else KlevelEnd
        if k0 < 0 or k1 <= k0:
            raise ValueError("invalid KlevelBegin/KlevelEnd range")
        da = da.isel({zdim: slice(k0, k1)})

    tdim = _find_axis_dim(ds, da, "T")
    if (LstepBegin is not None or LstepEnd is not None) and tdim is not None:
        l0 = 0 if LstepBegin is None else LstepBegin - 1
        l1 = da.sizes[tdim] if LstepEnd is None else LstepEnd
        if l0 < 0 or l1 <= l0:
            raise ValueError("invalid LstepBegin/LstepEnd range")
        da = da.isel({tdim: slice(l0, l1)})

    return da


def _apply_extrapolate(da: xr.DataArray, ydim: str, xdim: str) -> xr.DataArray:
    # Approximate fregrid extrapolation by directional nearest-fill on both axes.
    return da.ffill(xdim).bfill(xdim).ffill(ydim).bfill(ydim)


def _smooth_for_finer_step(
    da: xr.DataArray, ydim: str, xdim: str, steps: int
) -> xr.DataArray:
    if steps <= 0:
        return da
    window = 2**steps
    return (
        da.rolling({ydim: window, xdim: window}, center=True, min_periods=1)
        .mean()
        .astype(da.dtype)
    )


def _basis_vectors_from_lonlat(
    lon_deg: np.ndarray, lat_deg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.deg2rad(np.asarray(lon_deg))
    lat = np.deg2rad(np.asarray(lat_deg))

    e_lon = np.stack(
        (
            -np.sin(lon),
            np.cos(lon),
            np.zeros_like(lon),
        ),
        axis=0,
    )
    e_lat = np.stack(
        (
            -np.cos(lon) * np.sin(lat),
            -np.sin(lon) * np.sin(lat),
            np.cos(lat),
        ),
        axis=0,
    )
    return e_lon, e_lat


def _read_dst_vgrid_centers(vgrid_file: Path) -> np.ndarray:
    with xr.open_dataset(vgrid_file, decode_cf=False) as ds:
        if "zeta" not in ds:
            raise ValueError(f"{vgrid_file} must contain zeta")
        zeta = np.asarray(ds["zeta"].values)
    if zeta.ndim != 1 or (zeta.size - 1) % 2 != 0:
        raise ValueError("destination vgrid zeta must be 1-D with length 2*nz+1")
    return zeta[1::2]


def _vertical_interp(
    da: xr.DataArray,
    src_ds: xr.Dataset,
    dst_z: np.ndarray,
) -> xr.DataArray:
    zdim = _find_axis_dim(src_ds, da, "Z")
    if zdim is None:
        return da

    if zdim in src_ds:
        src_z = np.asarray(src_ds[zdim].values)
    else:
        src_z = np.arange(da.sizes[zdim], dtype=float)
        da = da.assign_coords({zdim: src_z})

    result = da.interp({zdim: dst_z}, kwargs={"fill_value": "extrapolate"})
    low = da.isel({zdim: 0}).broadcast_like(result)
    high = da.isel({zdim: -1}).broadcast_like(result)
    result = xr.where(result[zdim] < src_z.min(), low, result)
    result = xr.where(result[zdim] > src_z.max(), high, result)
    return result


def _xesmf_method(
    interp_method,
    monotonic,
    finer_step,
    output_mosaic,
    u_field,
    grid_type,
    src_tiles,
    input_mosaic_path,
    extrapolate,
) -> str:
    if interp_method == "bilinear":
        method = "bilinear"
    elif interp_method in {
        "conserve_order1",
        "conserve_order2",
        "conserve_order2_monotonic",
    }:
        method = "conservative"
    else:
        raise ValueError(
            "interp_method must be one of conserve_order1, conserve_order2, conserve_order2_monotonic, bilinear"
        )

    if finer_step and method != "bilinear":
        raise ValueError("finer_step is only valid when interp_method bilinear")

    if method == "bilinear" and output_mosaic:
        raise ValueError(
            "bilinear mode requires nlon/nlat regular lat-lon output and does not support output_mosaic"
        )

    if u_field:
        if method != "bilinear":
            raise ValueError("vector fields require interp_method bilinear")
        if grid_type != "AGRID":
            raise ValueError("vector fields currently support grid_type AGRID only")

    if interp_method in {"conserve_order2", "conserve_order2_monotonic"}:
        if len(src_tiles) != 6:
            raise ValueError(
                "conserve_order2 modes require a 6-tile cubed-sphere input mosaic"
            )

    if method == "bilinear" and len(src_tiles) != 6:
        raise ValueError("bilinear mode requires a 6-tile cubed-sphere input mosaic")

    if method == "bilinear":
        ncontact = _read_mosaic_ncontacts(input_mosaic_path)
        if ncontact != 12:
            raise ValueError(
                "bilinear mode requires a 12-contact cubed-sphere input mosaic"
            )

    if extrapolate:
        if len(src_tiles) != 1:
            raise ValueError("extrapolate is limited to single-tile input mosaics")
        if not _is_rectilinear_latlon(src_tiles[0]):
            raise ValueError(
                "extrapolate is limited to rectilinear lat-lon input grids"
            )

    return method


def _weight_filename(
    remap_file: Optional[str], src_idx: int, dst_idx: int, nsrc: int, ndst: int
) -> Optional[str]:
    if not remap_file:
        return None
    base = _ensure_nc(remap_file)
    if nsrc == 1 and ndst == 1:
        return base
    stem = base[:-3]
    return f"{stem}.src{src_idx + 1}.dst{dst_idx + 1}.nc"


def _build_regridders(
    src_tiles: Sequence[MosaicTile],
    dst_tiles: Sequence[MosaicTile],
    method: str,
    remap_file: Optional[str],
) -> Dict[Tuple[int, int], xe.Regridder]:

    global _tiles_type

    regridders: Dict[Tuple[int, int], xe.Regridder] = {}
    for si, src in enumerate(src_tiles):
        src_grid = _tile_to_regridder_grid(src)
        for di, dst in enumerate(dst_tiles):
            dst_grid = _tile_to_regridder_grid(dst)
            wfile = _weight_filename(remap_file, si, di, len(src_tiles), len(dst_tiles))
            kwargs = {}
            if wfile:
                kwargs["filename"] = wfile
                kwargs["reuse_weights"] = Path(wfile).exists()
            if method == "conservative":
                kwargs["ignore_degenerate"] = True
                if _tiles_type == "nest":
                    method = "conservative_normed"
                    kwargs["unmapped_to_nan"] = True

            regridders[(si, di)] = xe.Regridder(
                src_grid, dst_grid, method=method, **kwargs
            )
    return regridders


def _combine_regridded(
    pieces: Sequence[xr.DataArray],
    method: str,
) -> xr.DataArray:
    if not pieces:
        raise ValueError("no regridded pieces to combine")
    if len(pieces) == 1:
        return pieces[0]

    if method == "bilinear":
        out = pieces[0]
        for piece in pieces[1:]:
            out = out.combine_first(piece)
        return out

    total = xr.zeros_like(pieces[0]).fillna(0)
    valid = xr.zeros_like(pieces[0]).fillna(0)
    for piece in pieces:
        mask = piece.notnull()
        total = total + piece.fillna(0)
        valid = valid + mask.astype(total.dtype)
    return xr.where(valid > 0, total, np.nan)


def _regrid_scalar_field(
    src_data: Sequence[xr.DataArray],
    src_weights: Optional[Sequence[xr.DataArray]],
    src_tiles: Sequence[MosaicTile],
    dst_tiles: Sequence[MosaicTile],
    regridders: Dict[Tuple[int, int], xe.Regridder],
    method: str,
    apply_fill_missing: bool,
    finer_step: int,
) -> List[xr.DataArray]:
    outputs: List[xr.DataArray] = []
    for di, _ in enumerate(dst_tiles):
        pieces: List[xr.DataArray] = []
        for si, _src in enumerate(src_tiles):
            da = src_data[si]
            ydim, xdim = da.dims[-2], da.dims[-1]
            work = da.rename({ydim: "y", xdim: "x"})

            if src_weights is not None:
                w = src_weights[si].rename(
                    {src_weights[si].dims[-2]: "y", src_weights[si].dims[-1]: "x"}
                )
                w_b = xr.broadcast(w, work)[0]
                regrid_num = regridders[(si, di)](work * w_b)
                regrid_den = regridders[(si, di)](w_b)
                piece = regrid_num / xr.where(regrid_den > 0, regrid_den, np.nan)
            else:
                piece = regridders[(si, di)](work)

            piece = piece.rename({"y": ydim, "x": xdim})
            pieces.append(piece)

        combined = _combine_regridded(pieces, method)
        ydim, xdim = combined.dims[-2], combined.dims[-1]
        if method == "bilinear" and finer_step > 0:
            combined = _smooth_for_finer_step(combined, ydim, xdim, finer_step)
        if apply_fill_missing:
            combined = combined.ffill(xdim).bfill(xdim).ffill(ydim).bfill(ydim)
        outputs.append(combined)

    return outputs


def _regrid_vector_bilinear_field(
    u_src: Sequence[xr.DataArray],
    v_src: Sequence[xr.DataArray],
    src_tiles: Sequence[MosaicTile],
    dst_tiles: Sequence[MosaicTile],
    regridders: Dict[Tuple[int, int], xe.Regridder],
    apply_fill_missing: bool,
    finer_step: int,
) -> List[Tuple[xr.DataArray, xr.DataArray]]:
    if len(u_src) != len(v_src) or len(u_src) != len(src_tiles):
        raise ValueError(
            "u/v source fields and source tiles must have matching lengths"
        )

    x_src: List[xr.DataArray] = []
    y_src: List[xr.DataArray] = []
    z_src: List[xr.DataArray] = []

    for si, src_tile in enumerate(src_tiles):
        u_da = u_src[si]
        v_da = v_src[si]
        uy, ux = u_da.dims[-2], u_da.dims[-1]
        vy, vx = v_da.dims[-2], v_da.dims[-1]
        if (uy, ux) != (vy, vx):
            v_da = v_da.rename({vy: uy, vx: ux})

        e_lon, e_lat = _basis_vectors_from_lonlat(src_tile.lon_t, src_tile.lat_t)
        ex = xr.DataArray(e_lon[0], dims=(uy, ux))
        ey = xr.DataArray(e_lon[1], dims=(uy, ux))
        ez = xr.DataArray(e_lon[2], dims=(uy, ux))
        nx = xr.DataArray(e_lat[0], dims=(uy, ux))
        ny = xr.DataArray(e_lat[1], dims=(uy, ux))
        nz = xr.DataArray(e_lat[2], dims=(uy, ux))

        x_src.append(u_da * ex + v_da * nx)
        y_src.append(u_da * ey + v_da * ny)
        z_src.append(u_da * ez + v_da * nz)

    x_out = _regrid_scalar_field(
        x_src,
        None,
        src_tiles,
        dst_tiles,
        regridders,
        "bilinear",
        apply_fill_missing=apply_fill_missing,
        finer_step=finer_step,
    )
    y_out = _regrid_scalar_field(
        y_src,
        None,
        src_tiles,
        dst_tiles,
        regridders,
        "bilinear",
        apply_fill_missing=apply_fill_missing,
        finer_step=finer_step,
    )
    z_out = _regrid_scalar_field(
        z_src,
        None,
        src_tiles,
        dst_tiles,
        regridders,
        "bilinear",
        apply_fill_missing=apply_fill_missing,
        finer_step=finer_step,
    )

    uv_out: List[Tuple[xr.DataArray, xr.DataArray]] = []
    for di, dst_tile in enumerate(dst_tiles):
        x_da = x_out[di]
        y_da = y_out[di]
        z_da = z_out[di]
        ydim, xdim = x_da.dims[-2], x_da.dims[-1]

        dlon, dlat = _basis_vectors_from_lonlat(dst_tile.lon_t, dst_tile.lat_t)
        dex = xr.DataArray(dlon[0], dims=(ydim, xdim))
        dey = xr.DataArray(dlon[1], dims=(ydim, xdim))
        dez = xr.DataArray(dlon[2], dims=(ydim, xdim))
        dnx = xr.DataArray(dlat[0], dims=(ydim, xdim))
        dny = xr.DataArray(dlat[1], dims=(ydim, xdim))
        dnz = xr.DataArray(dlat[2], dims=(ydim, xdim))

        u_out = x_da * dex + y_da * dey + z_da * dez
        v_out = x_da * dnx + y_da * dny + z_da * dnz
        uv_out.append((u_out, v_out))

    return uv_out


def _attach_xy_coords(
    da: xr.DataArray,
    dst_tile: MosaicTile,
    standard_dimension: bool,
) -> xr.DataArray:
    ydim_old, xdim_old = da.dims[-2], da.dims[-1]
    ydim_new = "lat" if standard_dimension else ydim_old
    xdim_new = "lon" if standard_dimension else xdim_old

    if (ydim_new, xdim_new) != (ydim_old, xdim_old):
        da = da.rename({ydim_old: ydim_new, xdim_old: xdim_new})

    da = da.assign_coords(
        {
            xdim_new: dst_tile.lon1d_t,
            ydim_new: dst_tile.lat1d_t,
        }
    )
    return da


def _area_sum(da: xr.DataArray, tile: MosaicTile) -> xr.DataArray:
    grid = _tile_to_regridder_grid(tile)
    area = xe.util.cell_area(grid)
    ydim, xdim = da.dims[-2], da.dims[-1]
    field = da.rename({ydim: "y", xdim: "x"})
    return (field * area).sum(dim=("y", "x"), skipna=True)


def _to_netcdf(
    ds: xr.Dataset,
    out_path: Path,
    fmt: Optional[str],
    deflation: int,
    shuffle: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    front = [d for d in ("time",) if d in ds.dims]
    middle = [d for d in ds.dims if d not in {"time", "lat", "lon"}]
    back = [d for d in ("lat", "lon") if d in ds.dims]
    order = front + middle + back
    if tuple(order) != tuple(ds.dims):
        ds = ds.transpose(*order)

    ds.to_netcdf(out_path)


def validate_inputs(
    shuffle,
    deflation,
    grid_type,
    input_file,
    scalar_field,
    u_field,
    v_field,
    remap_file,
    output_file,
    output_mosaic,
    nlon,
    nlat,
    weight_field,
    dst_vgrid,
    extrapolate,
    symmetry,
    target_grid,
    associated_file_dir,
    stop_crit,
):

    if shuffle < -1 or shuffle > 1:
        raise ValueError("shuffle must be 0, 1, or omitted")
    if deflation < -1 or deflation > 9:
        raise ValueError("deflation must be between 0 and 9, or omitted")

    if grid_type not in ("AGRID", "BGRID"):
        raise ValueError("grid_type must be AGRID or BGRID")

    if len(input_file) == 0:
        if scalar_field or u_field or v_field:
            raise ValueError(
                "when input_file is not specified, scalar_field/u_field/v_field must not be specified"
            )
        if not remap_file:
            raise ValueError(
                "when input_file is not specified, remap_file must be specified"
            )
        save_weight_only = True
    elif len(input_file) in {1, 2}:
        save_weight_only = False

        if u_field and v_field:
            if len(u_field) != len(v_field):
                raise ValueError(
                    "number of u_field entries must equal number of v_field entries"
                )
        if not scalar_field and not u_field:
            raise ValueError(
                "at least one scalar_field or paired u_field/v_field is required"
            )
        if scalar_field and len(input_file) != 1:
            raise ValueError(
                "when scalar_field is specified, number of input files must be 1"
            )
        if len(input_file) == 2 and (scalar_field or not u_field):
            raise ValueError(
                "two input files are only supported for paired vector regridding (u in file1, v in file2)"
            )
    else:
        raise ValueError("number of input files must be 0, 1, or 2")

    if output_file and len(output_file) != len(input_file):
        raise ValueError("number of output files must match number of input files")
    if not output_file and input_file:
        raise ValueError("output_file is required when input_file is specified")

    if output_mosaic:
        if nlon or nlat:
            raise ValueError("do not specify nlon/nlat when output_mosaic is provided")
    else:
        if nlon <= 0 or nlat <= 0:
            raise ValueError(
                "nlon and nlat are required when output_mosaic is not provided"
            )

    if weight_field and u_field:
        raise ValueError("weight_field is not supported for vector interpolation")

    if dst_vgrid:
        extrapolate = True

    if dst_vgrid and u_field:
        raise ValueError("dst_vgrid is not supported for vector fields")

    if extrapolate and u_field:
        raise ValueError("extrapolate is not supported for vector fields")

    if symmetry:
        ...
    if target_grid:
        ...
    if associated_file_dir:
        ...
    if stop_crit != 0.005:
        ...

    return save_weight_only


def _to_list(value: object) -> list:

    if not isinstance(value, list):
        if value is None:
            return []
        return [value]
    return value


def fregrid(
    input_mosaic: str,
    input_file: list | Path = None,
    output_mosaic: Path = None,
    output_file: list | Path = None,
    input_dir: Path = None,
    output_dir: Path = None,
    scalar_field: list = None,
    u_field: list = None,
    v_field: list = None,
    remap_file: Path = None,
    interp_method: str = "conserve_order1",
    grid_type: str = "AGRID",
    symmetry: bool = False,
    target_grid: bool = False,
    finer_step: int = 0,
    center_y: bool = False,
    check_conserve: bool = False,
    monotonic: bool = False,
    lonBegin: float = -180.0,
    lonEnd: float = 180.0,
    latBegin: float = -90.0,
    latEnd: float = 90.0,
    nlon: int = 0,
    nlat: int = 0,
    KlevelBegin: int = None,
    KlevelEnd: int = None,
    LstepBegin: int = None,
    LstepEnd: int = None,
    weight_file: Path = None,
    weight_field: str = None,
    dst_vgrid: str = None,
    extrapolate: bool = False,
    stop_crit: float = 0.005,
    standard_dimension: bool = False,
    associated_file_dir: Path = None,
    fill_missing: bool = True,
    format: str = None,
    deflation: int = -1,
    shuffle: int = -1,
    tiles_type: str = None,
):

    global _tiles_type
    _tiles_type = tiles_type

    input_file = _to_list(input_file)
    output_file = _to_list(output_file)
    scalar_field = _to_list(scalar_field)
    u_field = _to_list(u_field)
    v_field = _to_list(v_field)

    save_weight_only = validate_inputs(
        **{k: v for k, v in locals().items() if k in g_fargs(validate_inputs)}
    )

    input_mosaic_path = Path(input_mosaic).resolve()
    src_tiles = _load_mosaic_tiles(input_mosaic_path)

    method = _xesmf_method(
        **{k: v for k, v in locals().items() if k in g_fargs(_xesmf_method)}
    )

    if output_mosaic:
        output_mosaic_path = Path(output_mosaic).resolve()
        dst_tiles = _load_mosaic_tiles(output_mosaic_path)
    else:
        output_mosaic_path = None
        center_y = center_y if method == "bilinear" else True

        dst_tiles = [
            _regular_latlon_tile(
                lon_begin=lonBegin,
                lon_end=lonEnd,
                lat_begin=latBegin,
                lat_end=latEnd,
                nlon=nlon,
                nlat=nlat,
                center_y=center_y,
            )
        ]

        in_gca = _mosaic_great_circle_algorithm(
            input_mosaic_path,
            in_gca=None,
            interp_method=interp_method,
            method=method,
            u_field=u_field,
        )
        out_gca = _mosaic_great_circle_algorithm(
            output_mosaic_path,
            in_gca=in_gca,
            interp_method=interp_method,
            method=method,
            u_field=u_field,
        )

        regridders = _build_regridders(src_tiles, dst_tiles, method, remap_file)

        if save_weight_only:
            for si in range(len(src_tiles)):
                for di in range(len(dst_tiles)):
                    wf = _weight_filename(
                        remap_file, si, di, len(src_tiles), len(dst_tiles)
                    )

            return 0

        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()

        src_tile_names = [t.tile_name for t in src_tiles]
        dst_tile_names = [t.tile_name for t in dst_tiles]

        file1_in_paths = _tile_file_paths(input_file[0], input_dir, src_tile_names)
        file1_out_paths = _tile_file_paths(output_file[0], output_dir, dst_tile_names)

        file2_in_paths: List[Path] = []
        file2_out_paths: List[Path] = []
        if len(input_file) == 2:
            file2_in_paths = _tile_file_paths(input_file[1], input_dir, src_tile_names)
            file2_out_paths = _tile_file_paths(
                output_file[1], output_dir, dst_tile_names
            )

        ds1_tiles = [xr.open_dataset(path) for path in file1_in_paths]
        ds2_tiles = (
            [xr.open_dataset(path) for path in file2_in_paths] if file2_in_paths else []
        )

        if weight_field:
            weight_base = weight_file if weight_file else input_file[0]
            weight_paths = _tile_file_paths(weight_base, input_dir, src_tile_names)
            dsw_tiles = [xr.open_dataset(path) for path in weight_paths]
            weight_tiles = []
            for dsw in dsw_tiles:
                if weight_field not in dsw:
                    raise ValueError(
                        f"weight field {weight_field} not found in {dsw.encoding.get('source', '?')}"
                    )
                weight_tiles.append(dsw[weight_field])
        else:
            dsw_tiles = []
            weight_tiles = None

        dst_vgrid = (
            _read_dst_vgrid_centers(Path(dst_vgrid).resolve()) if dst_vgrid else None
        )

        out_file1_tiles = [xr.Dataset() for _ in dst_tiles]
        out_file2_tiles = [xr.Dataset() for _ in dst_tiles]

        skipped_none_interp: List[str] = []

        for field in scalar_field:
            src_field_data: List[xr.DataArray] = []
            for ds in ds1_tiles:
                if field not in ds:
                    raise ValueError(
                        f"scalar field {field} missing in {ds.encoding.get('source', '?')}"
                    )
                da = ds[field]
                attr_interp = str(da.attrs.get("interp_method", "")).lower()
                if attr_interp == "none":
                    skipped_none_interp.append(field)
                    src_field_data = []
                    break
                da = _select_and_slice(
                    da, ds, KlevelBegin, KlevelEnd, LstepBegin, LstepEnd
                )
                ydim, xdim = da.dims[-2], da.dims[-1]
                if extrapolate:
                    da = _apply_extrapolate(da, ydim, xdim)
                src_field_data.append(da)

            if not src_field_data:
                continue

            field_weights: Optional[List[xr.DataArray]] = None
            if weight_tiles is not None:
                field_weights = []
                for wt in weight_tiles:
                    w = wt
                    if w.dims[-2:] != src_field_data[0].dims[-2:]:
                        w = w.rename(
                            {
                                w.dims[-2]: src_field_data[0].dims[-2],
                                w.dims[-1]: src_field_data[0].dims[-1],
                            }
                        )
                    field_weights.append(w)

            out_pieces = _regrid_scalar_field(
                src_field_data,
                field_weights,
                src_tiles,
                dst_tiles,
                regridders,
                method,
                apply_fill_missing=fill_missing,
                finer_step=finer_step,
            )

            for di, out_da in enumerate(out_pieces):
                if dst_vgrid is not None:
                    out_da = _vertical_interp(out_da, ds1_tiles[0], dst_vgrid)
                out_da = _attach_xy_coords(out_da, dst_tiles[di], standard_dimension)
                out_da.attrs.update(ds1_tiles[0][field].attrs)
                out_file1_tiles[di][field] = out_da

            if check_conserve:
                src_sum = sum(
                    _area_sum(src_field_data[i], src_tiles[i])
                    for i in range(len(src_tiles))
                )
                dst_sum = sum(
                    _area_sum(out_pieces[i], dst_tiles[i])
                    for i in range(len(dst_tiles))
                )
                src_val = float(src_sum.compute())
                dst_val = float(dst_sum.compute())
                diff = 0.0 if src_val == 0 else abs(dst_val - src_val) / abs(src_val)

        for uf, vf in zip(u_field, v_field):
            u_src: List[xr.DataArray] = []
            v_src: List[xr.DataArray] = []
            for i, ds in enumerate(ds1_tiles):
                if uf not in ds:
                    raise ValueError(
                        f"u field {uf} missing in {ds.encoding.get('source', '?')}"
                    )
                u_da = _select_and_slice(
                    ds[uf], ds, KlevelBegin, KlevelEnd, LstepBegin, LstepEnd
                )
                if extrapolate:
                    u_da = _apply_extrapolate(u_da, u_da.dims[-2], u_da.dims[-1])
                u_src.append(u_da)

                if len(ds2_tiles) == len(ds1_tiles):
                    vds = ds2_tiles[i]
                else:
                    vds = ds
                if vf not in vds:
                    raise ValueError(
                        f"v field {vf} missing in {vds.encoding.get('source', '?')}"
                    )
                v_da = _select_and_slice(
                    vds[vf], vds, KlevelBegin, KlevelEnd, LstepBegin, LstepEnd
                )
                if extrapolate:
                    v_da = _apply_extrapolate(v_da, v_da.dims[-2], v_da.dims[-1])
                v_src.append(v_da)

            uv_out = _regrid_vector_bilinear_field(
                u_src,
                v_src,
                src_tiles,
                dst_tiles,
                regridders,
                apply_fill_missing=fill_missing,
                finer_step=finer_step,
            )

            for di in range(len(dst_tiles)):
                u_regridded, v_regridded = uv_out[di]
                out_u = _attach_xy_coords(
                    u_regridded, dst_tiles[di], standard_dimension
                )
                out_v = _attach_xy_coords(
                    v_regridded, dst_tiles[di], standard_dimension
                )
                out_u.attrs.update(ds1_tiles[0][uf].attrs)
                if len(ds2_tiles) == len(ds1_tiles):
                    out_v.attrs.update(ds2_tiles[0][vf].attrs)
                else:
                    out_v.attrs.update(ds1_tiles[0][vf].attrs)

                out_file1_tiles[di][uf] = out_u
                if len(ds2_tiles) == len(ds1_tiles):
                    out_file2_tiles[di][vf] = out_v
                else:
                    out_file1_tiles[di][vf] = out_v

        for di, ds_out in enumerate(out_file1_tiles):
            src_ref = ds1_tiles[0]
            ds_out.attrs.update(src_ref.attrs)
            hist = str(ds_out.attrs.get("history", ""))
            cmd = " ".join(sys.argv)
            ds_out.attrs["history"] = (f"{hist}\n{cmd}").strip()

            _to_netcdf(
                ds_out,
                file1_out_paths[di],
                format,
                deflation,
                shuffle,
            )

        if file2_out_paths:
            for di, ds_out in enumerate(out_file2_tiles):
                src_ref = ds2_tiles[0]
                ds_out.attrs.update(src_ref.attrs)
                hist = str(ds_out.attrs.get("history", ""))
                cmd = " ".join(sys.argv)
                ds_out.attrs["history"] = (f"{hist}\n{cmd}").strip()
                _to_netcdf(
                    ds_out,
                    file2_out_paths[di],
                    format,
                    deflation,
                    shuffle,
                )

        for ds in ds1_tiles + ds2_tiles + dsw_tiles:
            ds.close()

        if skipped_none_interp:
            skipped_none_interp = sorted(set(skipped_none_interp))
    return skipped_none_interp
