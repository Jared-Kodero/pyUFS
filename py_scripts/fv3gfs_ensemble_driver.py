import hashlib
from pathlib import Path

import numpy as np
import xarray as xr

from fv3gfs_runtime import log
from fv3gfs_state import state

ensemble_stds = {}
ensemble_amp = 0.01  # 1% perturbation


def _get_stds(in_file, target_vars):
    out = {}
    with xr.open_dataset(in_file) as ds:
        for v in ds.data_vars:
            if v not in target_vars:
                continue
            da = ds[v]
            out[v] = {}
            for z in range(len(da.lev)):
                layer = da.isel(lev=z)
                out[v][z] = float(layer.std(skipna=True).values)
    return out


def _get_delta(
    scale: float, rng, shape=None, dims=None, coords=None, dx=None
) -> xr.DataArray:

    delta = rng.normal(0.0, scale, size=shape)
    delta = delta - delta.mean()
    da = xr.DataArray(
        delta,
        dims=dims,
        coords=coords,
    )
    return da


def _gen_ensemble(stds, in_file, out_file, target_vars, rng, dx):
    """
    Generate ensemble members by adding small perturbations to the input data. for GFDL SHiELD, 3km convective run
    """
    with xr.open_dataset(in_file) as ds:
        ds = ds.load()

        for v in ds.data_vars:
            if v not in target_vars:
                continue

            da = ds[v]  # get the data array for the variable

            for z in range(len(da.lev)):
                layer = da.isel(lev=z)
                coord_val = da.lev.values[z]
                delta = _get_delta(
                    stds[v][z] * ensemble_amp,
                    rng,
                    shape=layer.shape,
                    dims=layer.dims,
                    coords=layer.coords,
                    dx=dx,
                )
                new_layer = layer + delta
                da.loc[{"lev": coord_val}] = new_layer

            ds[v] = da

        ds.to_netcdf(out_file)

    if Path(out_file).exists():
        Path(in_file).unlink()


# dont touch this function,
def ensemble_config():

    if not state.ensemble_run:
        return

    if state.restart_no != 0:
        return

    if state.ensemble_id == 1:
        log.info(f"Skipping ensemble generation for ensemble  {state.ensemble_id}")
        return  # 1 member is the control, so no need to perturb

    log.info(f"Generating ensemble member for ensemble {state.ensemble_id}")

    seed_string = f"{state.init_datetime}_{state.ensemble_id}_{state.res}"
    seed = int(hashlib.sha256(seed_string.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    target_vars = {"t"}  # only perturb temperature enough for div of ensemble
    atm_files = list(Path(state.input).glob("gfs_data*.nc"))

    file_stds = {}
    for f in atm_files:
        file_stds[str(f)] = _get_stds(f, target_vars)

    for f in atm_files:
        tmp_f = f.with_suffix(".tmp")
        f.rename(tmp_f)

        if "nest" in f.name:
            tile_num = int(f.stem.split("tile")[-1])
            tile_idx = tile_num - 7
            dx = state.nest_res_km[tile_idx]
        else:
            dx = state.global_res_km

        _gen_ensemble(file_stds[str(f)], tmp_f, f, target_vars, rng, dx)
