from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import uuid
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
from derived_vars import calc_derived_vars
from fv3_runtime import exit_code, get_stream_handles
from fv3_state import load_fv3_state, state
from fv3_utils import cres_to_deg, env_setup
from pyfregrid import fregrid

warnings.filterwarnings("ignore")
load_fv3_state()  # ensure pstate is populated before any function calls

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
    <handle>.seg<nn>.global.nc or <handle>.seg<nn>.tile<N>.nc, and None
    for anything else, including merged files.
    """
    pattern = rf"^{re.escape(handle)}\.seg(\d+)\.(global|tile\d+)\.nc$"
    match = re.match(pattern, path.name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def group_alias(group: str, n_nests: int) -> str:
    """Map a tile group onto its output name: global, nest02, nest03, ..."""
    if group == "global":
        return "global"
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
        raise ValueError(f"merge_freq must be an integer, got {merge_freq!r}")

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
    merge_freq == 0 : never merge, retain one file per restart segment
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


def merge_files(inputs: list[Path], target: Path) -> None:
    """Concatenate inputs along time into target.

    The merged file is written to a scratch path and moved into place only
    on success, so target is never left truncated or removed if the
    concatenation fails. An existing target is treated as an additional
    input, and duplicated time records are resolved in favour of the newer
    segment.
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
        )
    else:
        ds = xr.open_dataset(sources[0])

    tmp = target.with_name(f".{target.name}.merge")

    try:
        if "time" in ds.dims:
            ds = ds.sortby("time")
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
        ds.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
    finally:
        ds.close()

    os.replace(tmp, target)

    for p in inputs:
        if p != target:
            p.unlink(missing_ok=True)


def merge_outputs(
    output_dir: Path,
    streams: list,
    n_nests: int,
    restart_no: int,
    total_restarts: int,
    merge_freq: int,
) -> None:
    """Merge per-restart regridded files according to merge_freq."""
    window = merge_window(restart_no, total_restarts, merge_freq)

    if window is None:
        log.info(f"No merge at restart {restart_no} (merge_freq = {merge_freq})")
        return

    first, last = window
    output_dir = Path(output_dir)
    handles = list(dict.fromkeys(stream_family(s) for s in streams))

    for handle in handles:
        segments: dict[str, list[tuple[int, Path]]] = {}

        for path in output_dir.glob(f"{handle}.*.nc"):
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


def post_process(ds: xr.Dataset, data_attrs: dict, dim_attrs: dict) -> xr.Dataset:

    ds = ds.sortby("plev", ascending=False)

    for var in ds.data_vars:
        ds[var].attrs.update(data_attrs.get(var, {}))

        if var in ["pr", "prc", "cnvprcpb_ave", "totprcpb_ave"]:
            ds[var] = ds[var] * 3600.0  # convert from m/s to mm/hr
            ds[var] = ds[var].clip(min=0, keep_attrs=True)
            ds[var].attrs["units"] = "mm/hr"

    ds = calc_derived_vars(ds)
    ds = ds[sorted(list(ds.data_vars))]

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

    except Exception:
        ...

    return ds


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

    else:
        tiles_type = "nest"
        input_file = stream_path.stem  # removes .nc
        hist_ds_file = state.hist / f"{input_file}.nc"

    with xr.open_dataset(hist_ds_file) as ds:
        data_vars = list(ds.data_vars)

    fregrid_out = state.tmp / "fregrid" / "out"
    fregrid_out.mkdir(parents=True, exist_ok=True)

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
    }

    chunk_size = 5  # number of variables to process in parallel
    tasks = []
    for idx, i in enumerate(range(0, len(data_vars), chunk_size)):
        chunk = data_vars[i : i + chunk_size]
        tasks.append((cmd, chunk, fregrid_out))

    with Pool(processes=min(len(tasks), py_ncpus)) as pool:
        pool.starmap(_run_fregrid, tasks)

    files = sorted(fregrid_out.glob("*.nc"))

    with xr.open_mfdataset(
        files,
        combine="by_coords",
        compat="override",
    ) as ds:
        data_attrs = {var: {**ds[var].attrs} for var in ds.data_vars}
        dim_attrs = {dim: {**ds[dim].attrs} for dim in ds.dims}

        ds = post_process(ds, data_attrs, dim_attrs)

        case = state.get("case_description") or ""
        description = state.get("description") or ""

        ds.attrs = {
            "tile_type": name,
            "resolution": f"{step:.2f} degrees",
            "case": case,
            "description": description,
        }

        ds.to_netcdf(output_file)

    shutil.rmtree(state.tmp / "fregrid")


def regrid_global_tiles(streams: list, c_res: int):
    if state.gtype == "nest":
        g_input_mosaic = state.work_dir / "GRID" / f"C{c_res}_coarse_mosaic.nc"
    else:
        g_input_mosaic = state.work_dir / "GRID" / f"C{c_res}_mosaic.nc"

    step = cres_to_deg(state.c_res).deg

    lon_begin = -180.0
    lon_end = 180.0
    lat_begin = -90.0
    lat_end = 90.0

    nx = int(360 / step)
    ny = int(180 / step)

    for stream in streams:
        input_file = stream
        output_file = state.output / segment_name(
            stream_family(stream), segment_index(stream), "global"
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

        lon_min = state.lon_min[i]
        lon_max = state.lon_max[i]
        lat_min = state.lat_min[i]
        lat_max = state.lat_max[i]

        nx = int(np.round(abs(lon_max - lon_min) / n_step))
        ny = int(np.round(abs(lat_max - lat_min) / n_step))

        input_mosaic = (
            state.work_dir / "GRID" / f"C{c_res}_nested{nest_idx:02d}_mosaic.nc"
        )

        if not input_mosaic.exists():
            log.error(f"Input mosaic file {input_mosaic} does not exist. Aborting!")
            sys.exit(1)

        for stream in streams:
            input_file = f"{stream}.nest{nest_idx:02d}.tile{tile}.nc"
            output_file = state.output / segment_name(
                stream_family(stream), segment_index(stream), f"tile{tile}"
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
            )


def regrid():
    env_setup()
    streams = get_stream_handles()
    # remove spec and static files from streams
    streams = [s for s in streams if "spec" not in s and "static" not in s]
    regrid_global_tiles(streams, state.c_res)
    regrid_nest_tiles(streams, state.c_res)

    merge_outputs(
        state.output,
        streams,
        state.n_nests,
        state.restart_no,
        state.total_restarts,
        get_merge_freq(),
    )

    static_path = Path(state.work_dir) / "STATIC"

    static_path.mkdir(parents=True, exist_ok=True)

    for f in Path(state.hist).glob("*"):
        if "spec" in f.name or "static" in f.name:
            dest = static_path / f.name.replace(f".{state.restart_no:02d}", "")
            if dest.exists():
                continue
            f.rename(dest)

    shutil.rmtree(state.tmp)


if __name__ == "__main__":
    try:
        regrid()

    except Exception as e:
        exit_code(-1)
        raise e
