from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import uuid
import warnings
from pathlib import Path

import dask
import numpy as np
import xarray as xr

import fv3_plugins
from fv3_diag_files import load_variables, stream_handles, variables
from fv3_state import load_fv3_state, save_fv3_state, setup_logging
from fv3_utils import cres_to_deg, env_setup, exit_code
from pyfregrid import fregrid

warnings.filterwarnings("ignore")
state = load_fv3_state()

log = logging.getLogger("REGRIDDER")

py_ncpus = len(os.sched_getaffinity(0))


def stream_family(stream: str) -> str:
    """Return the diag_table file handle stripped of path and restart tag."""
    return str(Path(stream).name).split(".")[0]


def segment_index(stream: str) -> int:
    """Restart index carried by the diag_table stream name, HIST/<handle>.<nn>."""
    tag = str(Path(stream).name).split(".")[-1]
    if tag.isdigit():
        return int(tag)
    return int(state.restart_no or 0)


def segment_name(handle: str, seg: int, group: str) -> str:
    """Name of a per-restart regridded file, before merging.

    The seg tag marks these as intermediates and keeps them outside the
    grammar of the merged names, so a merge can never select its own
    target as an input.
    """
    return f"{handle}.seg{seg:02d}.{group}.nc"


def parse_segment(path: Path, handle: str) -> tuple[int, str] | None:
    """Identify a per-restart regridded file.

    Returns (restart index, tile group) for names of the form
    <handle>.seg<nn>.global.nc, <handle>.seg<nn>.regional.nc or
    <handle>.seg<nn>.tile<N>.nc, and None for anything else, including
    merged files.
    """
    pattern = rf"^{re.escape(handle)}\.seg(\d+)\.(global|regional|tile\d+)\.nc$"
    match = re.match(pattern, path.name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def group_alias(group: str, n_nests: int) -> str:
    """Map a tile group onto its output name: global, regional, nest02, ..."""
    if group in ("global", "regional"):
        return group
    nest = int(group.removeprefix("tile")) - 6
    if 1 <= nest <= n_nests:
        return f"nest{nest + 1:02d}"
    return group


def get_merge_freq() -> int:
    """Read and validate merge_freq from the run state.

    -1 (default) merges the whole run, 0 disables merging, n > 0 merges
    every n restart segments.
    """
    merge_freq = state.get("merge_freq")

    if merge_freq is None:
        return -1

    if isinstance(merge_freq, bool) or not isinstance(merge_freq, int):
        raise TypeError(f"merge_freq must be an integer, got {merge_freq!r}")

    if merge_freq < -1:
        raise ValueError(
            f"merge_freq must be -1, 0 or a positive integer, got {merge_freq}"
        )

    return merge_freq


def merge_window(
    restart_no: int, total_restarts: int, merge_freq: int
) -> tuple[int, int] | None:
    """Return the inclusive restart range to merge at this restart, or None.

    merge_freq  < 0 : merge the complete run once, on the final restart
    merge_freq == 0 : never merge, publish one file per restart segment
    merge_freq  > 0 : merge every merge_freq restarts, flushing any
                      remainder on the final restart
    """
    last = total_restarts - 1

    if merge_freq == 0:
        return None

    if merge_freq < 0:
        return (0, last) if restart_no == last else None

    if (restart_no + 1) % merge_freq != 0 and restart_no != last:
        return None

    first = (restart_no // merge_freq) * merge_freq
    return first, restart_no


def merged_name(
    handle: str,
    alias: str,
    first: int,
    total_restarts: int,
    merge_freq: int,
) -> str:
    """Name of the merged file.

    A single merge covering the whole run is untagged. Chunked merges are
    numbered sequentially from 01, padded so that the names sort in time
    order.
    """
    if merge_freq <= 0 or merge_freq >= total_restarts:
        return f"{handle}.{alias}.nc"

    n_chunks = -(-total_restarts // merge_freq)
    width = max(2, len(str(n_chunks)))
    chunk = first // merge_freq + 1

    return f"{handle}.{chunk:0{width}d}.{alias}.nc"


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


# Static fields are attached once to each final file by add_static_fields(), so
# seg files and merge inputs never carry them.
STATIC_FIELDS = ("HGTsfc", "pk", "bk", "hyam", "hybm")


def without_static(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars(STATIC_FIELDS, errors="ignore")


def merge_files(inputs: list[Path], target: Path) -> None:
    """Concatenate inputs along time into target.

    The merged file is written to a scratch path and moved into place only
    on success, so target is never left truncated or removed if the
    concatenation fails. An existing target is treated as an additional
    input. All time records are retained and sorted chronologically.
    """
    sources = [target] + inputs if target.exists() else list(inputs)

    if len(sources) > 1:
        ds = xr.open_mfdataset(
            [str(p) for p in sources],
            combine="nested",
            concat_dim="time",
            coords="minimal",
            data_vars="minimal",
            compat="override",
            join="exact",
            decode_times=True,
            preprocess=without_static,
        )
    else:
        ds = without_static(xr.open_dataset(sources[0]))

    tmp = target.with_name(f".{target.name}.merge")

    try:
        if "time" in ds.dims:
            ds = ds.sortby("time")
        pack = pack_encoding(ds, bool(state.get("pack_output")))
        encoding = {
            var: {"zlib": True, "complevel": 4, **pack[var]} for var in ds.data_vars
        }
        ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
    finally:
        ds.close()

    os.replace(tmp, target)

    for p in inputs:
        if p != target:
            p.unlink(missing_ok=True)


def promote_segment_outputs(
    output_dir: Path, streams: list, n_nests: int
) -> list[tuple[Path, str]]:
    """Move the current restart's unmerged segment files into output_dir.

    Returns the promoted files as (path, group alias).

    Segment indices are taken from the stream names so each restart is promoted
    independently. Existing targets are atomically replaced, which makes a
    rerun of the same restart deterministic without affecting other restarts.
    """
    output_dir = Path(output_dir)
    segment_dir = output_dir / "seg"
    current = sorted(
        {(stream_family(stream), segment_index(stream)) for stream in streams}
    )
    promoted: list[tuple[Path, str]] = []

    for handle, seg in current:
        for path in segment_dir.glob(f"{handle}.seg{seg:02d}.*.nc"):
            parsed = parse_segment(path, handle)
            if parsed is None:
                continue
            idx, group = parsed
            if idx != seg:
                continue
            alias = group_alias(group, n_nests)
            target = output_dir / f"{handle}.seg{seg:02d}.{alias}.nc"
            os.replace(path, target)
            promoted.append((target, alias))

    return promoted


def merge_outputs(
    output_dir: Path,
    streams: list,
    n_nests: int,
    restart_no: int,
    total_restarts: int,
    merge_freq: int,
) -> list[tuple[Path, str]]:
    """Finalize per-restart regridded files according to merge_freq.

    Returns the final files written by this call as (path, group alias).
    """
    if merge_freq == 0:
        return promote_segment_outputs(output_dir, streams, n_nests)

    window = merge_window(restart_no, total_restarts, merge_freq)

    if window is None:
        return []

    first, last = window
    output_dir = Path(output_dir)
    segment_dir = output_dir / "seg"
    handles = list(dict.fromkeys(stream_family(s) for s in streams))
    merged: list[tuple[Path, str]] = []

    for handle in handles:
        segments: dict[str, list[tuple[int, Path]]] = {}

        for path in segment_dir.glob(f"{handle}.*.nc"):
            parsed = parse_segment(path, handle)
            if parsed is None:
                continue  # merged output or unrelated file
            idx, group = parsed
            if not first <= idx <= last:
                continue
            segments.setdefault(group, []).append((idx, path))

        for group, items in sorted(segments.items()):
            inputs = [path for _, path in sorted(items)]
            alias = group_alias(group, n_nests)
            target = output_dir / merged_name(
                handle, alias, first, total_restarts, merge_freq
            )

            merge_files(inputs, target)
            merged.append((target, alias))

    return merged


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

    ds = fv3_plugins.add_derived_variables(ds)
    ds = ds[sorted(ds.data_vars)]

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
    fregrid(
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


def _run_fregrid(base_cmd: dict, data_vars: list, fregrid_out: Path):
    tmp_name = f"{uuid.uuid4().hex}.nc"

    cmd = {
        **base_cmd,
        "scalar_field": data_vars,
        "output_file": tmp_name,
        "output_dir": fregrid_out,
    }
    fregrid(**cmd)


def call_fregrid(
    input_mosaic: Path,
    nx: int,
    ny: int,
    stream: str,
    output_file: str,
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

    if not Path(output_file).parent.exists():
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    stream_path = Path(stream)

    if name == "GLOBAL":
        tiles_type = "global"
        input_file = stream_path.name
        hist_ds_file = state.hist / f"{input_file}.tile6.nc"

    else:  # NEST or REGIONAL: one tile, no tile suffix for a regional domain
        tiles_type = "nest"
        input_file = stream_path.stem  # removes .nc
        hist_ds_file = state.hist / f"{input_file}.nc"

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
        "interp_method": "conserve_order1",
        "standard_dimension": True,
        "lonBegin": lon_begin,
        "lonEnd": lon_end,
        "latBegin": lat_begin,
        "latEnd": lat_end,
        "format": "netcdf4",
        "tiles_type": tiles_type,
        "remap_file": remap_file,
        # Missing values (pressure levels below the surface, points outside a
        # nest or regional domain) stay missing instead of being filled from
        # their neighbours.
        "fill_missing": False,
    }
    weight_cmd = {k: v for k, v in cmd.items() if k not in ("input_file", "input_dir")}
    fregrid(**weight_cmd)  # weights only, before the parallel chunks read them
    build_static_cache(weight_cmd, static_suffix, tiles_type, alias, state.tmp)

    chunk_size = 5  # number of variables to process in parallel
    tasks = []
    for idx, i in enumerate(range(0, len(data_vars), chunk_size)):
        chunk = data_vars[i : i + chunk_size]
        tasks.append((cmd, chunk, fregrid_out))

    delayed = [dask.delayed(_run_fregrid)(*t) for t in tasks]
    dask.compute(
        *delayed,
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
        pack = state.pack_output and get_merge_freq() == 0
        ds.to_netcdf(output_file, encoding=pack_encoding(ds, True) if pack else None)

    shutil.rmtree(state.tmp / "fregrid")


def regrid_global_tiles(streams: list, c_res: int):
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

    for stream in streams:
        input_file = stream
        output_file = (
            state.output
            / "seg"
            / segment_name(stream_family(stream), segment_index(stream), "global")
        )

        call_fregrid(
            g_input_mosaic,
            nx,
            ny,
            input_file,
            output_file,
            step,
            lon_begin,
            lon_end,
            lat_begin,
            lat_end,
            "GLOBAL",
            ".tile6",
            "global",
        )


def regrid_nest_tiles(streams: list, c_res: int):
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

        for stream in streams:
            input_file = f"{stream}.nest{nest_idx:02d}.tile{tile}.nc"
            output_file = (
                state.output
                / "seg"
                / segment_name(
                    stream_family(stream), segment_index(stream), f"tile{tile}"
                )
            )

            call_fregrid(
                input_mosaic,
                nx,
                ny,
                input_file,
                output_file,
                n_step,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                "NEST",
                f".nest{nest_idx:02d}.tile{tile}",
                f"nest{nest_idx:02d}",
            )


def regrid_regional_tile(streams: list, c_res: int):
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

    for stream in streams:
        output_file = (
            state.output
            / "seg"
            / segment_name(stream_family(stream), segment_index(stream), "regional")
        )
        call_fregrid(
            grid_file,
            nx,
            ny,
            f"{stream}.nc",
            output_file,
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
    env_setup()
    load_variables()  # output names of the staged diag_table
    streams = stream_handles()
    # remove spec and static files from streams
    streams = [s for s in streams if "spec" not in s and "static" not in s]
    if state.gtype in ("regional_gfdl", "regional_esg"):
        regrid_regional_tile(streams, state.c_res)
    else:
        regrid_global_tiles(streams, state.c_res)
        regrid_nest_tiles(streams, state.c_res)
    shutil.rmtree(state.tmp / "regrid_weights", ignore_errors=True)

    merge_freq = get_merge_freq()
    final_files = merge_outputs(
        state.output,
        streams,
        state.n_nests,
        state.restart_no,
        state.total_restarts,
        merge_freq,
    )
    for path, alias in final_files:
        add_static_fields(path, alias)

    static_path = Path(state.case_dir) / "STATIC"

    static_path.mkdir(parents=True, exist_ok=True)

    for f in list(Path(state.hist).glob("*")):
        if "spec" in f.name or "static" in f.name:
            dest = static_path / f.name.replace(f".{state.restart_no:02d}", "")
            if dest.exists():
                continue
            f.rename(dest)

    if state.resubmit_idx == state.resubmit:
        os.system(f"rm -rf {Path(state.output) / 'seg'}")

    state.regrid_progress = {"last": state.restart_no, "total": state.total_restarts}
    save_fv3_state()


if __name__ == "__main__":
    setup_logging()
    try:
        regrid()

    except Exception:
        exit_code(1)
        raise
