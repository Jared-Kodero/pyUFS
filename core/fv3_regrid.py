from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import uuid
import warnings
from contextlib import suppress
from itertools import batched
from pathlib import Path

import dask
import numpy as np
import xarray as xr
import xesmf as xe

import fv3_plugins
from derived_variables import derive
from fv3_diag_files import load_variables, variables
from fv3_runtime import exit_code
from fv3_state import load_fv3_state, save_fv3_state, setup_logging, update_walltime
from fv3_utils import cres_to_deg

warnings.filterwarnings("ignore")
state = load_fv3_state()

log = logging.getLogger("REGRIDDER")

py_ncpus = len(os.sched_getaffinity(0))


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
GCA = "great_circle_algorithm"

# Static fields are attached once to each final file by add_static_fields(), so
# per-restart files and merge inputs never carry them.
STATIC_FIELDS = ("HGTsfc", "pk", "bk", "hyam", "hybm")


# --------------------------------------------------------------------------- #
# grids (C: get_input_grid, get_output_grid_by_size)
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


def read_mosaic(path: Path) -> list[xr.Dataset]:
    """One grid per mosaic tile (attrs: name, gca).

    A grid (supergrid) file is accepted in place of a mosaic and read as a
    one-tile mosaic, for example the shaved compute-domain grid of a regional
    run, for which no mosaic is written.
    """
    with xr.open_dataset(path, decode_cf=False) as ds:
        if "gridfiles" not in ds and "x" in ds and "y" in ds:
            grid = supergrid_to_grid(
                ds["x"].values, ds["y"].values, name="tile1", gca=0
            )
            return [grid]
        if "gridfiles" not in ds:
            raise ValueError(f"mosaic file {path} does not contain gridfiles")
        files = char_array_to_list(ds["gridfiles"])
        names = (
            char_array_to_list(ds["gridtiles"])
            if "gridtiles" in ds
            else [f"tile{i + 1}" for i in range(len(files))]
        )
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
    return grids


def latlon_grid(
    lon_begin: float,
    lon_end: float,
    lat_begin: float,
    lat_end: float,
    nlon: int,
    nlat: int,
) -> xr.Dataset:
    """Regular lat-lon grid (C get_output_grid_by_size) whose cells tile [lat_begin, lat_end]."""
    lat_begin, lat_end = np.clip([lat_begin, lat_end], -90.0, 90.0)
    if lon_end <= lon_begin or lat_end <= lat_begin:
        raise ValueError("lonEnd must be > lonBegin and latEnd must be > latBegin")
    dlon = (lon_end - lon_begin) / nlon
    dlat = (lat_end - lat_begin) / nlat
    # Interleave corners (even) and centres (odd) into a supergrid.
    xs, ys = np.empty(2 * nlon + 1), np.empty(2 * nlat + 1)
    xs[0::2] = lon_begin + np.arange(nlon + 1) * dlon
    xs[1::2] = lon_begin + (np.arange(nlon) + 0.5) * dlon
    ys[0::2] = lat_begin + np.arange(nlat + 1) * dlat
    ys[1::2] = lat_begin + (np.arange(nlat) + 0.5) * dlat
    return supergrid_to_grid(*np.meshgrid(xs, ys), name="tile1", gca=0)


def bounding_box(grid_file: Path) -> tuple[float, float, float, float]:
    """Longitude and latitude range of a one-tile grid (cell corners).

    Longitudes are measured from the domain centre, so a domain that crosses
    0 or 180 degrees gives a continuous range (lon_end may exceed 180).
    """
    with xr.open_dataset(grid_file) as ds:
        x, y = ds["x"].values, ds["y"].values
    centre = float(x[x.shape[0] // 2, x.shape[1] // 2])
    offset = (x - centre + 180.0) % 360.0 - 180.0
    lon_begin = centre + float(offset.min())
    lon_begin = (lon_begin + 180.0) % 360.0 - 180.0
    return lon_begin, lon_begin + float(np.ptp(offset)), float(y.min()), float(y.max())


# --------------------------------------------------------------------------- #
# scalar conservative regridding onto a lat-lon grid (xarray + xESMF)
# --------------------------------------------------------------------------- #


def xregrid(
    input_mosaic: str,
    input_file: str | None = None,
    output_file: str | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    scalar_field: list | None = None,
    remap_file: Path | None = None,
    lonBegin: float = 0,
    lonEnd: float = 360,
    latBegin: float = -90,
    latEnd: float = 90,
    nlon: int = 0,
    nlat: int = 0,
    standard_dimension: bool = False,
    fill_missing: bool = True,  # C default is off; enabled here
    tiles_type: str | None = None,
) -> None:
    """Remap scalar fields from input_mosaic onto a regular lat-lon grid.

    Scalar fields whose interp_method attribute is "none" are skipped and logged.
    Without input_file, only the weights are computed and saved to remap_file.

    Conservative remapping is the ratio of two remapped sums over all source
    tiles, R(f) / R(1), where 1 marks valid source cells and 0 missing ones. A
    destination cell therefore takes the area-weighted mean of the valid source
    area that overlaps it: missing values (for example pressure levels below the
    surface) are excluded instead of counted as zero, and a cell shared by two
    tiles combines both.

    Deviations from the C fregrid: weights come from xESMF (ESMF regridding),
    not from FRE's exact cell-intersection algorithm, so results are close but
    not bit-identical; fill_missing defaults to True here (off in C).
    """
    scalar_field = scalar_field or []

    # ---- grids ------------------------------------------------------------ #
    input_mosaic = Path(input_mosaic).resolve()
    src_grids = read_mosaic(input_mosaic)
    if len({grid.attrs["gca"] for grid in src_grids}) > 1:
        raise ValueError(
            f"inconsistent great_circle_algorithm values across grid tiles in {input_mosaic}"
        )
    dst_grid = latlon_grid(lonBegin, lonEnd, latBegin, latEnd, nlon, nlat)
    lon, lat = dst_grid["lon"].values[0], dst_grid["lat"].values[:, 0]

    # ---- one regridder per source tile, weights cached in remap_file ------ #
    options: dict = {"method": "conservative", "ignore_degenerate": True}
    if tiles_type == "nest":
        options |= {"method": "conservative_normed", "unmapped_to_nan": True}
    regridders = []
    for i, src_grid in enumerate(src_grids, start=1):
        if remap_file:
            suffix = ".nc" if len(src_grids) == 1 else f".src{i}.dst1.nc"
            remap = Path(remap_file)
            remap = remap.with_name(remap.name.removesuffix(".nc") + suffix)
            options |= {"filename": remap, "reuse_weights": remap.exists()}
        regridders.append(xe.Regridder(src_grid, dst_grid, **options))
    if not input_file:  # weights only
        return

    # ---- fields ----------------------------------------------------------- #
    # One input file per source tile: base.nc for one tile, else base.<tile>.nc.
    stem = Path(input_dir, input_file.removesuffix(".nc"))
    tile_suffixes = (
        [""] if len(src_grids) == 1 else [f".{g.attrs['name']}" for g in src_grids]
    )
    datasets = [xr.open_dataset(f"{stem}{suffix}.nc") for suffix in tile_suffixes]
    out = xr.Dataset()
    skipped = []
    for field in scalar_field:
        if any(
            str(ds[field].attrs.get("interp_method", "")).lower() == "none"
            for ds in datasets
            if field in ds
        ):
            skipped.append(field)
            continue
        for ds in datasets:
            if field not in ds:
                raise ValueError(
                    f"scalar field {field} missing in {ds.encoding.get('source', '?')}"
                )

        # Sum R(f) and R(1) over the source tiles.
        tiles = [ds[field] for ds in datasets]
        ydim, xdim = tiles[0].dims[-2:]
        num = den = 0
        for tile, regridder in zip(tiles, regridders):
            tile = tile.rename({ydim: "y", xdim: "x"})
            num += regridder(tile.fillna(0.0))
            den += regridder(tile.notnull().astype(tile.dtype))
        da = (num / den.where(den > 0)).astype(tiles[0].dtype)
        da = da.rename({"y": ydim, "x": xdim})
        if fill_missing:
            da = da.ffill(xdim).bfill(xdim).ffill(ydim).bfill(ydim)
        if standard_dimension:
            da = da.rename({ydim: "lat", xdim: "lon"})
            ydim, xdim = "lat", "lon"
        out[field] = da.assign_coords({xdim: lon, ydim: lat})
        out[field].attrs.update(tiles[0].attrs)

    # ---- write: dims ordered (time, ..., lat, lon) ------------------------ #
    out.attrs.update(datasets[0].attrs)
    path = Path(output_dir, output_file.removesuffix(".nc") + ".nc").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.transpose("time", ..., "lat", "lon", missing_dims="ignore").to_netcdf(path)
    for ds in datasets:
        ds.close()

    if skipped:
        log.warning(
            "Skipped fields with interp_method 'none': %s", ", ".join(sorted(skipped))
        )


# --------------------------------------------------------------------------- #
# output encoding and post-processing
# --------------------------------------------------------------------------- #


def pack_encoding(ds: xr.Dataset, pack: bool) -> dict[str, dict]:
    """Build per-variable NetCDF encoding.

    When packing is enabled, floating-point gridded fields with at least
    three dimensions are packed as integers using scale_factor/add_offset.

    float32 -> int16
    float64 -> int32

    The lowest two integer codes are excluded from valid packed data:
        min       : unused
        min + 1   : _FillValue / missing_value
        min + 2   : minimum valid packed value

    Non-gridded variables and all-missing/non-finite fields remain unpacked.
    Existing variable encodings are cleared when packing is requested.
    """
    encoding: dict[str, dict] = {}

    for name, da in ds.data_vars.items():
        encoding[name] = {}

        if not pack:
            continue

        # Prevent inherited packing metadata from being reused.
        da.encoding = {}

        if da.ndim < 3 or da.dtype.kind != "f":
            continue

        target_dtype = np.int32 if da.dtype == np.float64 else np.int16
        iinfo = np.iinfo(target_dtype)

        fill_value = iinfo.min + 1
        packed_min = iinfo.min + 2
        packed_max = iinfo.max

        vmin = float(da.min(skipna=True))
        vmax = float(da.max(skipna=True))

        # Leave all-missing or otherwise non-finite fields unpacked.
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            continue

        if vmax > vmin:
            scale = (vmax - vmin) / (packed_max - packed_min)
            offset = vmin - packed_min * scale
        else:
            # Constant field: map the value to packed integer zero.
            scale = 1.0
            offset = vmin

        for attr in ("valid_range", "valid_min", "valid_max"):
            da.attrs.pop(attr, None)

        encoding[name] = {
            "dtype": target_dtype,
            "scale_factor": scale,
            "add_offset": offset,
            "_FillValue": fill_value,
            "missing_value": fill_value,
        }

    return encoding


def without_static(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars(STATIC_FIELDS, errors="ignore")


def horizontal_fields(ds: xr.Dataset) -> tuple[list[str], list[str]]:
    """Split data variables into fields on the model grid and the rest.

    Time-mean streams also carry average_T1, average_T2, average_DT and
    time_bnds, which have no horizontal dimensions; they are copied to the
    regridded file unchanged.
    """
    on_grid, other = [], []
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        horizontal = {"grid_xt", "grid_yt"} <= dims or {"xaxis_1", "yaxis_1"} <= dims
        (on_grid if horizontal else other).append(name)
    return on_grid, other


def post_process(ds: xr.Dataset, data_attrs: dict, dim_attrs: dict) -> xr.Dataset:

    if "plev" in ds.dims:  # surface-only streams carry no pressure axis
        ds = ds.sortby("plev", ascending=False)

    # Precipitation rates are clipped at zero, found by their native diag_table
    # field so the output_name chosen there does not matter.
    precip = {variables["totprcpb_ave"], variables["cnvprcpb_ave"]}

    for var in ds.data_vars:
        ds[var].attrs.update(data_attrs.get(var, {}))

        if var in precip:
            ds[var] = ds[var].clip(min=0, keep_attrs=True)

    ds = ds[sorted(ds.data_vars)]
    for func in derive:
        ds = func(ds, state=state, variables=variables)
    ds = fv3_plugins.apply("regrid", ds, variables)

    for dim in ds.dims:
        ds[dim].attrs.update(dim_attrs.get(dim, {}))

    ds["lat"].attrs = {
        "axis": "Y",
        "standard_name": "latitude",
        "units": "degrees_north",
    }
    ds["lon"].attrs = {
        "axis": "X",
        "standard_name": "longitude",
        "units": "degrees_east",
    }
    if "plev" in ds.coords:
        ds["plev"].attrs = {
            "axis": "Z",
            "standard_name": "pressure_level",
            "units": "hPa",
        }

    ds = ds.transpose(..., "lat", "lon")

    try:
        ds["time"] = ds.indexes["time"].to_datetimeindex(time_unit="ns")
        ds["time"].encoding["units"] = "seconds since 1970-01-01 00:00:00"
        ds["time"].encoding["dtype"] = "int64"

    except (AttributeError, KeyError, TypeError, ValueError):
        # Keep the decoded time when it cannot be expressed as datetime64.
        pass

    return ds


# --------------------------------------------------------------------------- #
# static fields
# --------------------------------------------------------------------------- #


def static_source(suffix: str) -> tuple[Path, str] | None:
    """Locate the atmos_static file of a group: (directory, input_file stem).

    ``suffix`` is ".tile6" (global), ".nest02.tile7" (nest) or "" (regional).
    The file is in STATIC once a restart has finalized it, else in HIST as
    atmos_static.<nn><suffix>.nc.
    """
    pattern = re.compile(rf"atmos_static(\.\d+)?{re.escape(suffix)}\.nc")
    for folder in (Path(state.case_dir) / "STATIC", Path(state.hist)):
        for path in sorted(folder.glob("atmos_static*.nc")):
            if pattern.fullmatch(path.name):
                return folder, path.name[: -len(".nc") - len(suffix)]
    return None


def static_cache(alias: str) -> Path:
    """Regridded static fields of one output group (global, nest02, regional)."""
    return Path(state.case_dir) / "STATIC" / f"regridded.{alias}.nc"


def build_static_cache(
    base_cmd: dict, suffix: str, tiles_type: str, alias: str, tmp_dir: Path
) -> None:
    """Regrid the static fields of a group once and keep them in STATIC.

    HGTsfc (model orography) is regridded with the weights of the group. The
    vertical coefficients pk, bk, hyam and hybm are 1-D and copied unchanged.
    Later streams and segments reuse the cache; add_static_fields() attaches
    it to the final output files.
    """
    cache = static_cache(alias)
    if cache.exists():
        return

    source = static_source(suffix)
    if source is None:
        log.warning(f"atmos_static not found for {alias}; static fields not added")
        return
    folder, stem = source

    out = Path(tmp_dir) / "static"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    xregrid(
        **{
            **base_cmd,
            "input_file": stem if tiles_type == "global" else f"{stem}{suffix}",
            "input_dir": folder,
            "scalar_field": ["HGTsfc"],
            "output_file": "HGTsfc.nc",
            "output_dir": out,
        }
    )
    with xr.open_dataset(out / "HGTsfc.nc") as hgt:
        da = hgt["HGTsfc"].squeeze(drop=True).load()
    # Attributes as in tables/diag_field.csv (gfs_sfc, HGTsfc).
    attrs = {"long_name": "surface geopotential height", "units": "gpm"}
    static = xr.Dataset({"HGTsfc": (("lat", "lon"), da.values, attrs)})

    name = f"{stem}.tile6.nc" if tiles_type == "global" else f"{stem}{suffix}.nc"
    with xr.open_dataset(folder / name) as st:
        for var in STATIC_FIELDS[1:]:
            if var in st:
                v = st[var].squeeze(drop=True).reset_coords(drop=True).load()
                static[var] = (v.dims, v.values, {**v.attrs})

    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(f".{cache.name}.partial")
    static.to_netcdf(partial)
    os.replace(partial, cache)  # a cache is never left half written
    shutil.rmtree(out)


def add_static_fields(path: Path, alias: str) -> None:
    """Append the static fields of a group to a final output file, in place.

    HGTsfc is added to every file. The vertical coefficients are added only to
    files on model levels (those with a pfull dimension). Files that already
    hold the fields are left unchanged, so a rerun is harmless.
    """
    cache = static_cache(alias)
    if not cache.exists():
        log.warning(f"{path.name}: no static fields for {alias}; not added")
        return

    with xr.open_dataset(cache) as static, xr.open_dataset(path) as ds:
        wanted = ["HGTsfc"]
        if "pfull" in ds.dims:
            wanted += [v for v in STATIC_FIELDS[1:] if v in static]
        missing = [v for v in wanted if v not in ds]
        if not missing:
            return
        extra = static[missing].load()

    # The appended file keeps lat and lon as they are: only new variables are written.
    extra.to_netcdf(path, mode="a")


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #


def merge_files(inputs: list[Path], target: Path) -> None:
    """Concatenate inputs along time into target.

    The merged file is written to a scratch path and moved into place only
    on success, so target is never left truncated if the concatenation
    fails.
    """
    ds = xr.open_mfdataset(
        [str(p) for p in inputs],
        combine="nested",
        concat_dim="time",
        coords="minimal",
        data_vars="minimal",
        compat="override",
        join="exact",
        decode_times=True,
        preprocess=without_static,
    )

    tmp = target.with_name(f".{target.name}.merge")

    try:
        out = ds.sortby("time")
        pack = pack_encoding(out, bool(state.get("pack_output")))
        encoding = {
            var: {"zlib": True, "complevel": 4, **pack[var]} for var in out.data_vars
        }
        out.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
    finally:
        ds.close()  # sortby returns a new object, so close the opened files here

    os.replace(tmp, target)


def merge_outputs() -> None:
    """Merge the per-restart files of OUTPUT into chunks of merge_freq restarts.

    Called on the final restart. The whole run is one chunk if merge_freq < 0.
    The per-restart files are moved to OUTPUT/.tmp, merged there and written back
    to OUTPUT as <handle>.<alias>.<nn>.nc, with nn counting chunks from 1 and
    zero-padded to at least two digits (.01, ..., .99, .001, ...); a single chunk
    has no suffix. Chunks are recorded in state.merged_files before they are
    written, so a rerun finishes only the missing ones.
    """
    size = state.total_restarts if state.merge_freq < 0 else state.merge_freq
    width = max(2, len(str(-(-state.total_restarts // size))))
    tmp = state.output / ".tmp"
    tmp.mkdir(exist_ok=True)

    for path in sorted(state.output.glob("*.nc")):
        is_part = re.fullmatch(r".+\.(global|regional|nest\d+)\.\d+", path.stem)
        if is_part and path.name not in state.merged_files:
            path.replace(tmp / path.name)

    chunks: dict[tuple[str, int], list[Path]] = {}
    for path in sorted(tmp.glob("*.nc")):
        prefix, n = path.stem.rsplit(".", 1)
        chunks.setdefault((prefix, (int(n) - 1) // size + 1), []).append(path)

    for (prefix, chunk), inputs in chunks.items():
        tag = "" if size >= state.total_restarts else f".{chunk:0{width}d}"
        target = state.output / f"{prefix}{tag}.nc"

        if target.name not in state.merged_files:
            state.merged_files.append(target.name)
            save_fv3_state()
        if not target.exists():
            merge_files(inputs, target)
        add_static_fields(target, prefix.rsplit(".", 1)[1])
        for path in inputs:
            path.unlink()

    shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
# driver (C: main)
# --------------------------------------------------------------------------- #


def regrid_stream(
    input_mosaic: Path,
    nx: int,
    ny: int,
    handle: str,
    step: float,
    lon_begin: float,
    lon_end: float,
    lat_begin: float,
    lat_end: float,
    name: str = "",
    static_suffix: str = "",
    alias: str = "",
):

    if not Path(input_mosaic).exists():
        raise FileNotFoundError(f"Input mosaic file {input_mosaic} does not exist.")

    stream = f"{handle}.{state.restart_no:02d}"
    tiles_type = "global" if name == "GLOBAL" else "nest"
    input_file = stream if name == "GLOBAL" else f"{stream}{static_suffix}"
    hist_ds_file = state.hist / f"{stream}{static_suffix}.nc"

    # Per-restart files stay in OUTPUT; merge_outputs replaces them at the end.
    width = max(2, len(str(state.total_restarts)))
    output_file = state.output / f"{handle}.{alias}.{state.restart_no + 1:0{width}d}.nc"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(hist_ds_file) as ds:
        data_vars, other_vars = horizontal_fields(ds)
        extras = ds[other_vars].load() if other_vars else None

    if not data_vars:
        log.warning(f"{hist_ds_file.name}: no fields on the model grid; not regridded")
        return

    fregrid_out = state.tmp / "fregrid" / "out"
    shutil.rmtree(fregrid_out, ignore_errors=True)
    fregrid_out.mkdir(parents=True, exist_ok=True)

    # Weights are computed once per source and target grid and reused by
    # every chunk and stream (regrid() removes them at the end).
    weights_dir = state.tmp / "regrid_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    remap_file = weights_dir / f"{Path(input_mosaic).stem}.{nx}x{ny}.nc"

    cmd = {
        "input_mosaic": input_mosaic,
        "nlon": nx,
        "nlat": ny,
        "input_file": input_file,
        "input_dir": state.hist,
        "standard_dimension": True,
        "lonBegin": lon_begin,
        "lonEnd": lon_end,
        "latBegin": lat_begin,
        "latEnd": lat_end,
        "tiles_type": tiles_type,
        "remap_file": remap_file,
        # Missing values (pressure levels below the surface, points outside a
        # nest or regional domain) stay missing instead of being filled from
        # their neighbours.
        "fill_missing": False,
    }
    weight_cmd = {k: v for k, v in cmd.items() if k not in ("input_file", "input_dir")}
    xregrid(**weight_cmd)  # weights only, before the parallel chunks read them
    build_static_cache(weight_cmd, static_suffix, tiles_type, alias, state.tmp)

    chunk_size = 5  # number of variables to process in parallel
    tasks = [
        dask.delayed(xregrid)(
            **cmd,
            scalar_field=list(chunk),
            output_file=f"{uuid.uuid4().hex}.nc",
            output_dir=fregrid_out,
        )
        for chunk in batched(data_vars, chunk_size)
    ]
    dask.compute(
        *tasks,
        scheduler=state.preprocess_dask_scheduler,
        num_workers=min(len(tasks), py_ncpus),
        chunksize=1,
    )

    files = sorted(fregrid_out.glob("*.nc"))

    with xr.open_mfdataset(
        files,
        combine="by_coords",
        compat="override",
    ) as ds:
        if extras is not None:
            ds = ds.merge(extras, compat="override", join="left")
        data_attrs = {var: {**ds[var].attrs} for var in ds.data_vars}
        dim_attrs = {dim: {**ds[dim].attrs} for dim in ds.dims}

        ds = without_static(post_process(ds, data_attrs, dim_attrs))

        case = state.get("case_description") or ""
        description = state.get("description") or ""

        ds.attrs = {
            "tile_type": name,
            "resolution": f"{step:.2f} degrees",
            "case": case,
            "description": description,
        }

        # Segments are final files only when merging is disabled.
        pack = state.pack_output and state.merge_freq == 0
        ds.to_netcdf(output_file, encoding=pack_encoding(ds, True) if pack else None)

    if state.merge_freq == 0:
        add_static_fields(output_file, alias)

    shutil.rmtree(state.tmp / "fregrid")


def regrid_global_tiles(handles: list, c_res: int):
    if state.gtype == "nest":
        g_input_mosaic = state.case_dir / "GRID" / f"C{c_res}_coarse_mosaic.nc"
    else:
        g_input_mosaic = state.case_dir / "GRID" / f"C{c_res}_mosaic.nc"

    step = cres_to_deg(state.c_res).deg

    lon_begin = 0.0
    lon_end = 360.0
    lat_begin = -90.0
    lat_end = 90.0

    nx = int(360 / step)
    ny = int(180 / step)

    for handle in handles:
        regrid_stream(
            g_input_mosaic,
            nx,
            ny,
            handle,
            step,
            lon_begin,
            lon_end,
            lat_begin,
            lat_end,
            "GLOBAL",
            ".tile6",
            "global",
        )


def regrid_nest_tiles(handles: list, c_res: int):
    if state.gtype != "nest":
        return

    refine_ratio = state.refine_ratio
    for i in range(len(refine_ratio)):
        nest = i + 1
        nest_idx = i + 2
        tile = 6 + nest

        if state.nest_type == "telescoping":
            n_step = cres_to_deg(c_res * np.prod(refine_ratio[: i + 1])).deg
        else:
            n_step = cres_to_deg(c_res * refine_ratio[i]).deg

        lon_min = state.lon_min[i] % 360.0
        lon_max = lon_min + (state.lon_max[i] - state.lon_min[i]) % 360.0
        lat_min = state.lat_min[i]
        lat_max = state.lat_max[i]

        nx = max(1, int(np.round((lon_max - lon_min) / n_step)))
        ny = max(1, int(np.round((lat_max - lat_min) / n_step)))

        input_mosaic = (
            state.case_dir / "GRID" / f"C{c_res}_nested{nest_idx:02d}_mosaic.nc"
        )

        if not input_mosaic.exists():
            log.error(f"Input mosaic file {input_mosaic} does not exist. Aborting!")
            sys.exit(1)

        for handle in handles:
            regrid_stream(
                input_mosaic,
                nx,
                ny,
                handle,
                n_step,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                "NEST",
                f".nest{nest_idx:02d}.tile{tile}",
                f"nest{nest_idx:02d}",
            )


def regrid_regional_tile(handles: list, c_res: int):
    """Regrid a standalone regional domain onto its lat-lon bounding box.

    The history is written on the compute domain, the halo-0 grid of
    fv3_shave; points of the box outside the domain are missing.
    """
    grid_file = state.case_dir / "GRID" / f"C{c_res}_grid.tile7.halo0.nc"
    if not grid_file.exists():
        raise FileNotFoundError(f"Regional compute-domain grid not found: {grid_file}")

    step = cres_to_deg(c_res).deg
    lon_begin, lon_end, lat_begin, lat_end = bounding_box(grid_file)
    nx = max(1, int(np.round((lon_end - lon_begin) / step)))
    ny = max(1, int(np.round((lat_end - lat_begin) / step)))

    for handle in handles:
        regrid_stream(
            grid_file,
            nx,
            ny,
            handle,
            step,
            lon_begin,
            lon_end,
            lat_begin,
            lat_end,
            "REGIONAL",
            "",
            "regional",
        )


def regrid():
    load_variables()  # output names of the staged diag_table
    # remove spec and static files from the streams
    handles = [h for h in state.diag_handles if "spec" not in h and "static" not in h]
    if state.gtype in ("regional_gfdl", "regional_esg"):
        regrid_regional_tile(handles, state.c_res)
    else:
        regrid_global_tiles(handles, state.c_res)
        regrid_nest_tiles(handles, state.c_res)
    shutil.rmtree(state.tmp / "regrid_weights", ignore_errors=True)

    last = state.restart_no == state.total_restarts - 1
    if last and state.merge_freq != 0:
        merge_outputs()

    static_path = Path(state.case_dir) / "STATIC"

    static_path.mkdir(parents=True, exist_ok=True)

    for f in list(Path(state.hist).glob("*")):
        if "spec" in f.name or "static" in f.name:
            dest = static_path / f.name.replace(f".{state.restart_no:02d}", "")
            if dest.exists():
                continue
            f.rename(dest)

    if last:  # bookkeeping of the regridder is not archived with the state
        state.merged_files = []
        state.diag_handles = []

    state.regrid_progress = {"last": state.restart_no, "total": state.total_restarts}
    update_walltime()
    save_fv3_state()


if __name__ == "__main__":
    setup_logging()
    try:
        regrid()

    except Exception:
        exit_code(1)
        raise
