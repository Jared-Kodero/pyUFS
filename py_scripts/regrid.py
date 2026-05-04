from __future__ import annotations

import logging
import shutil
import sys
import uuid
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
from fv3gfs_runtime import exit_code
from fv3gfs_setup import logger
from fv3gfs_state import load_state
from fv3gfs_state import prev_state as state
from fv3gfs_utils import cres_to_deg, env_setup
from merge_ouputs import merge_outputs
from pyfregrid import fregrid

warnings.filterwarnings("ignore")
load_state()
logger(state.debug)

log = logging.getLogger("REGRIDDING")


def get_stream_handles() -> list[str]:
    path = Path(state.home) / "diag_table"
    handles = []

    with open(path) as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) == 8 and not ("static" in parts[3] or "spec" in parts[3]):
                handles.append(parts[3])

    return list(dict.fromkeys(handles))


def post_process(ds: xr.Dataset, data_attrs: dict, dim_attrs: dict) -> xr.Dataset:
    # rename plev to level if it exists
    if "plev" in ds.coords or "plev" in ds.dims:
        ds = ds.rename({"plev": "level"})
        ds["level"].attrs = {
            "units": "hPa",
            "standard_name": "pressure_level",
        }

        ds = ds.sortby("level", ascending=False)

    for var in ds.data_vars:
        ds[var].attrs.update(data_attrs.get(var, {}))

        if var in ["pr", "prc", "cnvprcpb_ave", "totprcpb_ave"]:
            ds[var] = ds[var] * 3600.0  # convert from m/s to mm/hr
            ds[var] = ds[var].clip(min=0, keep_attrs=True)
            ds[var].attrs["units"] = "mm/hr"

    ds = ds[sorted(list(ds.data_vars))]

    for dim in ds.dims:
        ds[dim].attrs.update(dim_attrs.get(dim, {}))

    ds["lat"].attrs = {
        "standard_name": "latitude",
        "units": "degrees_north",
    }
    ds["lon"].attrs = {
        "standard_name": "longitude",
        "units": "degrees_east",
    }

    ds = ds.transpose(..., "lat", "lon")

    try:
        ds["time"] = ds.indexes["time"].to_datetimeindex(time_unit="ns")
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

    chunk_size = 10

    tasks = []
    for idx, i in enumerate(range(0, len(data_vars), chunk_size)):
        chunk = data_vars[i : i + chunk_size]
        tasks.append((cmd, chunk, fregrid_out))

    with Pool(processes=min(len(tasks), 10)) as pool:
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

        ds.attrs = {
            "case": state.get("case_description", ""),
            "tile_type": name,
            "resolution": f"{step:.2f} degrees",
            "description": state.description,
        }

        ds.to_netcdf(output_file)

    shutil.rmtree(state.tmp / "fregrid")


def regrid_global_tiles(streams: list, c_res: int):
    if state.gtype == "nest":
        g_input_mosaic = state.home / "GRID" / f"C{c_res}_coarse_mosaic.nc"
    else:
        g_input_mosaic = state.home / "GRID" / f"C{c_res}_mosaic.nc"

    step = cres_to_deg(state.res).deg

    lon_begin = -180.0
    lon_end = 180.0
    lat_begin = -90.0
    lat_end = 90.0

    nx = int(360 / step)
    ny = int(180 / step)

    for stream in streams:
        input_file = stream
        output_file = state.output / Path(f"{stream}.global.nc").name

        if state.restart_no == 0 and not state.continue_run:
            output_file = Path(str(output_file).replace(f"_{state.restart_no:03d}", ""))

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

        input_mosaic = state.home / "GRID" / f"C{c_res}_nested{nest_idx:02d}_mosaic.nc"

        if not input_mosaic.exists():
            log.error(f"Input mosaic file {input_mosaic} does not exist. Aborting!")
            sys.exit(1)

        for stream in streams:
            input_file = f"{stream}.nest{nest_idx:02d}.tile{tile}.nc"
            output_file = state.output / Path(f"{stream}.tile{tile}.nc").name

            if state.restart_no == 0 and not state.continue_run:
                output_file = Path(
                    str(output_file).replace(f"_{state.restart_no:03d}", "")
                )

            call_fregrid(
                input_mosaic,
                nx,
                ny,
                input_file,  # should be the file HIST/fv3_hist_r000.nest02.tile7.nc
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
    log.info("Regridding FV3 hist files to regular lat-lon grid")
    streams = get_stream_handles()
    regrid_global_tiles(streams, state.res)
    regrid_nest_tiles(streams, state.res)

    if state.resubmit == 0:
        merge_outputs(
            state.output, streams, state.n_nests, state.run_nhours, state.total_restarts
        )
        log.info("Run Completed Successfully!")

    for f in state.hist.glob("*"):
        if "spec" in f.name or "static" in f.name:
            continue
        f.unlink(missing_ok=True)
    shutil.rmtree(state.tmp)


if __name__ == "__main__":
    try:
        regrid()

    except Exception as e:
        exit_code(-1)
        raise e
