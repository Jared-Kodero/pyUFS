"""Python reimplementation of fregrid using xarray, dask, and xESMF.

This module mirrors the fregrid command-line interface (UFS_UTILS
sorc/fre-nctools.fd/tools/fregrid) while using Python-native data handling
and xESMF for interpolation.

Deviations from the C reference, all intentional:
  - Interpolation weights come from xESMF (ESMF regridding), not from FRE's
    exact clip / great-circle cell-intersection algorithm. Results are close
    but not bit-identical to the C tool.
  - `lonBegin`/`lonEnd` default to -180/180 here; the C tool defaults to 0/360.
  - `fill_missing` defaults to True here; the C tool defaults to off.
  - `finer_step` is approximated by a post-interpolation rolling mean rather
    than by refining the target grid before bilinear interpolation.
  - `format`, `deflation`, and `shuffle` are accepted for CLI parity but are
    not yet applied to the output NetCDF encoding.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from contextlib import ExitStack, suppress
from functools import reduce
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

warnings.filterwarnings("ignore", message=".*F_CONTIGUOUS.*", module="xesmf.backend")
log = logging.getLogger("REGRIDDER")

XESMF_METHODS = {
    "conserve_order1": "conservative",
    "conserve_order2": "conservative",
    "conserve_order2_monotonic": "conservative",
    "bilinear": "bilinear",
}
GCA = "great_circle_algorithm"


# --------------------------------------------------------------------------- #
# grids (C: get_input_grid, get_output_grid_from_mosaic, get_output_grid_by_size)
# --------------------------------------------------------------------------- #
def supergrid_to_grid(x: np.ndarray, y: np.ndarray, **attrs: object) -> xr.Dataset:
    """xESMF grid from an FMS supergrid: odd indices are centres, even are corners."""
    y = np.clip(y, -90.0, 90.0)
    return xr.Dataset(
        {
            "lon": (("y", "x"), x[1::2, 1::2]),
            "lat": (("y", "x"), y[1::2, 1::2]),
            "lon_b": (("y_b", "x_b"), x[::2, ::2]),
            "lat_b": (("y_b", "x_b"), y[::2, ::2]),
        },
        attrs=attrs,
    )


def char_array_to_list(da: xr.DataArray) -> list[str]:
    arr = da.values
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    items: list[str] = []
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


def read_mosaic(path: Path) -> tuple[list[xr.Dataset], int]:
    """One grid per mosaic tile (attrs: name, gca) and the mosaic contact count.

    A grid (supergrid) file is accepted in place of a mosaic and read as a
    one-tile mosaic, for example the shaved compute-domain grid of a regional
    run, for which no mosaic is written.
    """
    with xr.open_dataset(path, decode_cf=False) as ds:
        if "gridfiles" not in ds and "x" in ds and "y" in ds:
            grid = supergrid_to_grid(
                ds["x"].values, ds["y"].values, name="tile1", gca=0
            )
            return [grid], 0
        if "gridfiles" not in ds:
            raise ValueError(f"mosaic file {path} does not contain gridfiles")
        files = char_array_to_list(ds["gridfiles"])
        names = (
            char_array_to_list(ds["gridtiles"])
            if "gridtiles" in ds
            else [f"tile{i + 1}" for i in range(len(files))]
        )
        if "contacts" in ds:
            ncontact = ds["contacts"].shape[0] if ds["contacts"].ndim else 0
        else:
            ncontact = ds.sizes.get("ncontact", 0)
    if len(names) != len(files):
        raise ValueError("mosaic gridtiles and gridfiles lengths do not match")

    grids = []
    for name, file in zip(names, files):
        grid_path = (path.parent / file).resolve()
        with xr.open_dataset(grid_path, decode_cf=False) as gds:
            if "x" not in gds or "y" not in gds:
                raise ValueError(f"grid file {grid_path} must contain x and y")
            x, y = gds["x"].values, gds["y"].values
            flag = (gds["tile"].attrs if "tile" in gds else {}).get(
                GCA, gds.attrs.get(GCA)
            )
            if flag is None and GCA in gds and gds[GCA].size:
                flag = gds[GCA].values.flat[0]
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"grid file {grid_path}: x and y must be 2-D")
        if x.shape[0] % 2 == 0 or x.shape[1] % 2 == 0:
            raise ValueError(
                f"grid file {grid_path}: supergrid shape must be odd in both dimensions"
            )
        text = flag.decode("ascii", "ignore") if isinstance(flag, bytes) else str(flag)
        text, gca = text.strip().lower(), 0
        with suppress(ValueError):
            gca = 1 if text in ("true", "t", "yes", "y") else int(float(text))
        grids.append(supergrid_to_grid(x, y, name=name, gca=gca))
    return grids, ncontact


def latlon_grid(
    lon_begin: float,
    lon_end: float,
    lat_begin: float,
    lat_end: float,
    nlon: int,
    nlat: int,
    center_y: bool,
) -> xr.Dataset:
    """Regular lat-lon grid (C get_output_grid_by_size).

    With center_y, cells tile [lat_begin, lat_end]; otherwise the first and
    last cell centres sit on lat_begin and lat_end.
    """
    lat_begin, lat_end = np.clip([lat_begin, lat_end], -90.0, 90.0)
    if lon_end <= lon_begin or lat_end <= lat_begin:
        raise ValueError("lonEnd must be > lonBegin and latEnd must be > latBegin")
    if not center_y and nlat == 1:
        raise ValueError("nlat must be > 1 when center_y is not set")
    dlon = (lon_end - lon_begin) / nlon
    dlat = (lat_end - lat_begin) / (nlat if center_y else nlat - 1)
    shift = 0.5 if center_y else 0.0
    # Interleave corners (even) and centres (odd) into a supergrid.
    xs, ys = np.empty(2 * nlon + 1), np.empty(2 * nlat + 1)
    xs[0::2] = lon_begin + np.arange(nlon + 1) * dlon
    xs[1::2] = lon_begin + (np.arange(nlon) + 0.5) * dlon
    ys[0::2] = lat_begin + (np.arange(nlat + 1) + shift - 0.5) * dlat
    ys[1::2] = lat_begin + (np.arange(nlat) + shift) * dlat
    return supergrid_to_grid(*np.meshgrid(xs, ys), name="tile1", gca=0)


def data_paths(base: str | Path, directory: Path, names: Sequence[str]) -> list[Path]:
    """Per-tile file names (C set_mosaic_data_file): base.nc or base.<tile>.nc."""
    path = Path(directory, base)
    stem = path.name.removesuffix(".nc")
    suffixes = [".nc"] if len(names) == 1 else [f".{name}.nc" for name in names]
    return [path.with_name(stem + suffix).resolve() for suffix in suffixes]


# --------------------------------------------------------------------------- #
# fields (C: get_input_data, do_scalar_*_interp, do_vector_bilinear_interp)
# --------------------------------------------------------------------------- #
def axis_dim(ds: xr.Dataset, da: xr.DataArray, axis: str) -> str | None:
    """First dim of `da` on axis X/Y/Z/T, from `cartesian_axis` or the dim name."""
    for dim in da.dims:
        name = dim.lower()
        if dim in ds and "cartesian_axis" in ds[dim].attrs:
            kind = str(ds[dim].attrs["cartesian_axis"]).upper()
        elif "time" in name or name == "t":
            kind = "T"
        elif name.startswith("z") or "lev" in name or "depth" in name:
            kind = "Z"
        elif name.startswith("x") or "lon" in name:
            kind = "X"
        elif name.startswith("y") or "lat" in name:
            kind = "Y"
        else:
            kind = "?"
        if kind == axis:
            return dim
    return None


def read_field(
    ds: xr.Dataset, name: str, kind: str, ranges: dict, extrapolate: bool
) -> xr.DataArray:
    """Field limited to the 1-based inclusive K/L ranges, optionally nearest-filled."""
    if name not in ds:
        raise ValueError(
            f"{kind} field {name} missing in {ds.encoding.get('source', '?')}"
        )
    da = ds[name]
    for axis, (begin, end, label) in ranges.items():
        dim = axis_dim(ds, da, axis)
        if dim is None or (begin is None and end is None):
            continue
        start = 0 if begin is None else begin - 1
        stop = da.sizes[dim] if end is None else end
        if start < 0 or stop <= start:
            raise ValueError(f"invalid {label} range")
        da = da.isel({dim: slice(start, stop)})
    if extrapolate:  # approximates C extrapolation on lat-lon input
        ydim, xdim = da.dims[-2:]
        da = da.ffill(xdim).bfill(xdim).ffill(ydim).bfill(ydim)
    return da


def unit_vectors(
    grid: xr.Dataset, dims: Sequence[str]
) -> tuple[xr.DataArray, xr.DataArray]:
    """Cartesian (xyz) components of local east and north unit vectors (C unit_vect_latlon)."""
    lon = np.deg2rad(xr.DataArray(grid["lon"].values, dims=dims))
    lat = np.deg2rad(xr.DataArray(grid["lat"].values, dims=dims))
    east = xr.concat([-np.sin(lon), np.cos(lon), xr.zeros_like(lon)], "xyz")
    north = xr.concat(
        [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)], "xyz"
    )
    return east, north


def interp(
    src: Sequence[xr.DataArray],
    weights: Sequence[xr.DataArray] | None,
    regridders: list[list[xe.Regridder]],
    method: str,
    fill_missing: bool,
    finer_step: int,
) -> list[xr.DataArray]:
    """Remap per-tile source fields to every destination tile and merge the tiles.

    Conservative remapping is the ratio of two remapped sums over all source
    tiles, R(w f) / R(w), where w is the weight (1 by default) on valid
    source cells and 0 on missing ones. A destination cell therefore takes
    the area-weighted mean of the valid source area that overlaps it: missing
    values (for example pressure levels below the surface) are excluded
    instead of counted as zero, and a cell shared by two tiles combines both.
    """
    outputs = []
    for row in regridders:
        if method != "bilinear":
            num = den = None
            for si, (da, regridder) in enumerate(zip(src, row)):
                ydim, xdim = da.dims[-2:]
                work = da.rename({ydim: "y", xdim: "x"})
                w = 1.0
                if weights is not None:
                    w = weights[si].rename(dict(zip(weights[si].dims[-2:], ("y", "x"))))
                    w = xr.broadcast(w, work)[0]
                valid = work.notnull()
                n_part = regridder(xr.where(valid, work * w, 0.0))
                d_part = regridder(xr.where(valid, w, 0.0).astype(work.dtype))
                num = n_part if num is None else num + n_part
                den = d_part if den is None else den + d_part
            out = (num / den.where(den > 0)).astype(src[0].dtype)
            out = out.rename({"y": ydim, "x": xdim})
        else:
            pieces = []
            for da, regridder in zip(src, row):
                ydim, xdim = da.dims[-2:]
                work = da.rename({ydim: "y", xdim: "x"})
                pieces.append(regridder(work).rename({"y": ydim, "x": xdim}))
            out = reduce(xr.DataArray.combine_first, pieces)

        ydim, xdim = out.dims[-2:]
        if finer_step > 0:
            window = {ydim: 2**finer_step, xdim: 2**finer_step}
            out = (
                out.rolling(window, center=True, min_periods=1).mean().astype(out.dtype)
            )
        if fill_missing:
            out = out.ffill(xdim).bfill(xdim).ffill(ydim).bfill(ydim)
        outputs.append(out)
    return outputs


def to_output(
    da: xr.DataArray, grid: xr.Dataset, standard_dimension: bool, attrs: dict
) -> xr.DataArray:
    """Attach 1-D destination lon/lat coordinates and the source attributes."""
    if standard_dimension:
        da = da.rename(dict(zip(da.dims[-2:], ("lat", "lon"))))
    ydim, xdim = da.dims[-2:]
    da = da.assign_coords({xdim: grid["lon"].values[0], ydim: grid["lat"].values[:, 0]})
    da.attrs.update(attrs)
    return da


# --------------------------------------------------------------------------- #
# driver (C: main)
# --------------------------------------------------------------------------- #
def fregrid(
    input_mosaic: str,
    input_file: list | Path | None = None,
    output_mosaic: Path | None = None,
    output_file: list | Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    scalar_field: list | None = None,
    u_field: list | None = None,
    v_field: list | None = None,
    remap_file: Path | None = None,
    interp_method: str = "conserve_order1",
    grid_type: str = "AGRID",
    symmetry: bool = False,
    target_grid: bool = False,
    finer_step: int = 0,
    center_y: bool = False,
    check_conserve: bool = False,
    monotonic: bool = False,
    lonBegin: float = 0,
    lonEnd: float = 360,
    latBegin: float = -90,
    latEnd: float = 90,
    nlon: int = 0,
    nlat: int = 0,
    KlevelBegin: int | None = None,
    KlevelEnd: int | None = None,
    LstepBegin: int | None = None,
    LstepEnd: int | None = None,
    weight_file: Path | None = None,
    weight_field: str | None = None,
    dst_vgrid: str | None = None,
    extrapolate: bool = False,
    stop_crit: float = 0.005,
    standard_dimension: bool = False,
    associated_file_dir: Path | None = None,
    fill_missing: bool = True,  # C default is off; enabled here
    format: str | None = None,
    deflation: int = -1,
    shuffle: int = -1,
    tiles_type: str | None = None,
) -> None:
    """Remap scalar and/or vector fields from input_mosaic onto the target grid.

    Scalar fields whose interp_method attribute is "none" are skipped and logged.
    """
    input_file, output_file, scalar_field, u_field, v_field = (
        [] if v is None else v if isinstance(v, list) else [v]
        for v in (input_file, output_file, scalar_field, u_field, v_field)
    )
    n = len(input_file)
    errors = {
        "shuffle must be 0, 1, or omitted": not -1 <= shuffle <= 1,
        "deflation must be between 0 and 9, or omitted": not -1 <= deflation <= 9,
        "grid_type must be AGRID or BGRID": grid_type not in ("AGRID", "BGRID"),
        "number of input files must be 0, 1, or 2": n > 2,
        "when input_file is not specified, scalar_field/u_field/v_field must not be specified": (
            n == 0 and (scalar_field or u_field or v_field)
        ),
        "when input_file is not specified, remap_file must be specified": (
            n == 0 and not remap_file
        ),
        "number of u_field entries must equal number of v_field entries": (
            n and u_field and v_field and len(u_field) != len(v_field)
        ),
        "at least one scalar_field or paired u_field/v_field is required": (
            n and not scalar_field and not u_field
        ),
        "when scalar_field is specified, number of input files must be 1": (
            n == 2 and scalar_field
        ),
        "two input files are only supported for paired vector regridding (u in file1, v in file2)": (
            n == 2 and not u_field
        ),
        "number of output files must match number of input files": (
            output_file and len(output_file) != n
        ),
        "output_file is required when input_file is specified": n and not output_file,
        "do not specify nlon/nlat when output_mosaic is provided": (
            output_mosaic and (nlon or nlat)
        ),
        "nlon and nlat are required when output_mosaic is not provided": (
            not output_mosaic and (nlon <= 0 or nlat <= 0)
        ),
        "weight_field is not supported for vector interpolation": (
            weight_field and u_field
        ),
        "dst_vgrid is not supported for vector fields": dst_vgrid and u_field,
        "extrapolate is not supported for vector fields": extrapolate and u_field,
    }
    if message := next((m for m, failed in errors.items() if failed), None):
        raise ValueError(message)

    # ---- input grid and interpolation method ------------------------------ #
    input_mosaic = Path(input_mosaic).resolve()
    src_grids, ncontact = read_mosaic(input_mosaic)
    if interp_method not in XESMF_METHODS:
        raise ValueError(f"interp_method must be one of {', '.join(XESMF_METHODS)}")
    method = XESMF_METHODS[interp_method]
    bilinear, nsrc = method == "bilinear", len(src_grids)
    lon, lat = src_grids[0]["lon"].values, src_grids[0]["lat"].values
    errors = {
        "finer_step is only valid when interp_method bilinear": finer_step
        and not bilinear,
        "bilinear mode requires nlon/nlat regular lat-lon output and does not support output_mosaic": (
            bilinear and output_mosaic
        ),
        "vector fields require interp_method bilinear": u_field and not bilinear,
        "weight_field requires a conservative interp_method": weight_field and bilinear,
        "vector fields currently support grid_type AGRID only": (
            u_field and grid_type != "AGRID"
        ),
        "conserve_order2 modes require a 6-tile cubed-sphere input mosaic": (
            interp_method.startswith("conserve_order2") and nsrc != 6
        ),
        "bilinear mode requires a 6-tile cubed-sphere input mosaic": bilinear
        and nsrc != 6,
        "bilinear mode requires a 12-contact cubed-sphere input mosaic": (
            bilinear and ncontact != 12
        ),
        "extrapolate is limited to single-tile input mosaics": extrapolate
        and nsrc != 1,
        "extrapolate is limited to rectilinear lat-lon input grids": (
            extrapolate
            and nsrc == 1
            and not (
                np.allclose(lon, lon[:1], rtol=0.0, atol=1e-10)
                and np.allclose(lat, lat[:, :1], rtol=0.0, atol=1e-10)
            )
        ),
    }
    if message := next((m for m, failed in errors.items() if failed), None):
        raise ValueError(message)

    # ---- output grid and great-circle consistency ------------------------- #
    if output_mosaic:
        output_mosaic = Path(output_mosaic).resolve()
        dst_grids, _ = read_mosaic(output_mosaic)
    else:
        dst_grids = [
            latlon_grid(
                lonBegin, lonEnd, latBegin, latEnd, nlon, nlat, center_y or not bilinear
            )
        ]
    for path, grids in ((input_mosaic, src_grids), (output_mosaic, dst_grids)):
        if len({grid.attrs["gca"] for grid in grids}) > 1:
            raise ValueError(
                f"inconsistent great_circle_algorithm values across grid tiles in {path}"
            )
    gca = src_grids[0].attrs["gca"] or dst_grids[0].attrs["gca"]
    if gca and interp_method != "conserve_order1":
        raise ValueError(
            "when great_circle_algorithm is active, interp_method must be conserve_order1"
        )

    # ---- regridders [dst][src], weights cached in remap_file -------------- #
    options: dict = {"method": method}
    if method == "conservative":
        options["ignore_degenerate"] = True
        if tiles_type == "nest":
            options |= {"method": "conservative_normed", "unmapped_to_nan": True}
    one_to_one = nsrc == len(dst_grids) == 1
    regridders = []
    for di, dst in enumerate(dst_grids):
        row = []
        for si, src in enumerate(src_grids):
            if remap_file:
                suffix = ".nc" if one_to_one else f".src{si + 1}.dst{di + 1}.nc"
                remap = Path(remap_file)
                remap = remap.with_name(remap.name.removesuffix(".nc") + suffix)
                options |= {"filename": remap, "reuse_weights": remap.exists()}
            row.append(xe.Regridder(src, dst, **options))
        regridders.append(row)
    if n == 0:  # weights only
        return

    # ---- data files ------------------------------------------------------- #
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    src_names = [grid.attrs["name"] for grid in src_grids]
    dst_names = [grid.attrs["name"] for grid in dst_grids]
    ranges = {
        "Z": (KlevelBegin, KlevelEnd, "KlevelBegin/KlevelEnd"),
        "T": (LstepBegin, LstepEnd, "LstepBegin/LstepEnd"),
    }
    out1 = [xr.Dataset() for _ in dst_grids]
    out2 = [xr.Dataset() for _ in dst_grids]
    skipped = []

    with ExitStack() as stack:
        ds1, ds2, dsw = (
            [
                stack.enter_context(xr.open_dataset(path))
                for path in data_paths(base, input_dir, src_names)
            ]
            if base
            else []
            for base in (
                input_file[0],
                n == 2 and input_file[1],
                weight_field and (weight_file or input_file[0]),
            )
        )
        weights = None
        if weight_field:
            for ds in dsw:
                if weight_field not in ds:
                    source = ds.encoding.get("source", "?")
                    raise ValueError(
                        f"weight field {weight_field} not found in {source}"
                    )
            weights = [ds[weight_field] for ds in dsw]

        dst_z = None
        if dst_vgrid:  # cell centres of a 2*nz+1 vertical supergrid
            with xr.open_dataset(Path(dst_vgrid).resolve(), decode_cf=False) as ds:
                if "zeta" not in ds:
                    raise ValueError(f"{dst_vgrid} must contain zeta")
                zeta = ds["zeta"].values
            if zeta.ndim != 1 or zeta.size % 2 == 0:
                raise ValueError(
                    "destination vgrid zeta must be 1-D with length 2*nz+1"
                )
            dst_z = zeta[1::2]

        # ---- scalar fields ------------------------------------------------ #
        for field in scalar_field:
            src = []
            for ds in ds1:
                attrs = ds[field].attrs if field in ds else {}
                if str(attrs.get("interp_method", "")).lower() == "none":
                    break
                src.append(read_field(ds, field, "scalar", ranges, extrapolate))
            if len(src) < len(ds1):
                skipped.append(field)
                continue

            out = interp(src, weights, regridders, method, fill_missing, finer_step)
            for di, (da, grid) in enumerate(zip(out, dst_grids)):
                zdim = axis_dim(ds1[0], da, "Z") if dst_z is not None else None
                if zdim is not None:  # linear in z, end values held outside the range
                    if zdim not in ds1[0]:
                        da = da.assign_coords(
                            {zdim: np.arange(da.sizes[zdim], dtype=float)}
                        )
                    src_z = (ds1[0] if zdim in ds1[0] else da)[zdim].values
                    low, high = da.isel({zdim: 0}), da.isel({zdim: -1})
                    da = da.interp({zdim: dst_z}, kwargs={"fill_value": "extrapolate"})
                    da = xr.where(da[zdim] < src_z.min(), low.broadcast_like(da), da)
                    da = xr.where(da[zdim] > src_z.max(), high.broadcast_like(da), da)
                attrs = ds1[0][field].attrs
                out1[di][field] = to_output(da, grid, standard_dimension, attrs)

            if check_conserve:
                src_sum, dst_sum = (
                    sum(
                        (
                            d.rename(dict(zip(d.dims[-2:], ("y", "x"))))
                            * xe.util.cell_area(g)
                        ).sum(("y", "x"))
                        for d, g in zip(fields, grids)
                    )
                    for fields, grids in ((src, src_grids), (out, dst_grids))
                )
                log.debug(
                    "%s area-weighted sum: source %g, destination %g",
                    field,
                    float(src_sum),
                    float(dst_sum),
                )

        # ---- vector fields: rotate to Cartesian, remap, rotate back ------- #
        paired = len(ds2) == len(ds1)
        for uf, vf in zip(u_field, v_field):
            xyz = []
            for ds, vds, grid in zip(ds1, ds2 if paired else ds1, src_grids):
                u = read_field(ds, uf, "u", ranges, extrapolate)
                v = read_field(vds, vf, "v", ranges, extrapolate)
                v = v.rename(dict(zip(v.dims[-2:], u.dims[-2:])))
                east, north = unit_vectors(grid, u.dims[-2:])
                xyz.append((u * east + v * north).transpose("xyz", ...))

            out = interp(xyz, None, regridders, method, fill_missing, finer_step)
            for di, (comp, grid) in enumerate(zip(out, dst_grids)):
                east, north = unit_vectors(grid, comp.dims[-2:])
                u_out = (comp * east).sum("xyz", skipna=False)
                v_out = (comp * north).sum("xyz", skipna=False)
                v_attrs = (ds2 if paired else ds1)[0][vf].attrs
                out1[di][uf] = to_output(
                    u_out, grid, standard_dimension, ds1[0][uf].attrs
                )
                (out2 if paired else out1)[di][vf] = to_output(
                    v_out, grid, standard_dimension, v_attrs
                )

        # ---- write: dims ordered (time, ..., lat, lon) ---------------------- #
        for outs, ds_in, base in zip((out1, out2), (ds1, ds2), output_file):
            for ds, path in zip(outs, data_paths(base, output_dir, dst_names)):
                ds.attrs.update(ds_in[0].attrs)
                path.parent.mkdir(parents=True, exist_ok=True)
                order = ("time", ..., "lat", "lon")
                ds.transpose(*order, missing_dims="ignore").to_netcdf(path)

    if skipped:
        log.warning(
            "Skipped fields with interp_method 'none': %s",
            ", ".join(sorted(set(skipped))),
        )
