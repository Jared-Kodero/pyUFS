from pathlib import Path

import numpy as np
import xarray as xr
from fv3_runtime import log
from fv3_state import compute_checksum, state

ENSEMBLE_AMP = 1e-3


def _get_stds(in_file: Path, target_vars: set) -> dict:
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
    scale: float,
    rng: np.random.Generator,
    shape: tuple = None,
    dims: tuple = None,
    coords: tuple = None,
    dx: float | None = None,
) -> xr.DataArray:

    delta = rng.normal(0.0, scale, size=shape)
    delta = delta - delta.mean()
    da = xr.DataArray(
        delta,
        dims=dims,
        coords=coords,
    )
    return da


def _gen_ensemble(
    stds: dict,
    in_file: Path,
    out_file: Path,
    target_vars: set,
    rng: np.random.Generator,
    dx: float,  # nest resolution in km,
):
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
                    stds[v][z] * ENSEMBLE_AMP,
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
        return  # 1 member is the control, so no need to perturb

    log.info(f"Generating ensemble member for ensemble {state.ensemble_id}")

    checksum = compute_checksum(
        state,
        hash_keys=[
            "ensemble_id",
            "n_ensembles",
            "k_split",
            "n_split",
            "dt_atmos",
            "dt_ocean",
        ],
    )

    seed = int(checksum, 16) % (2**32)
    rng = np.random.default_rng(seed)

    target_vars = {"t"}  # only perturb temperature enough for div of ensemble
    atm_files = sorted(Path(state.input).glob("gfs_data*.nc"))

    file_stds = {}
    for f in atm_files:
        file_stds[str(f)] = _get_stds(f, target_vars)

    for f in atm_files:
        tmp_f = f.with_suffix(".tmp")
        f.rename(tmp_f)

        if "nest" in f.name:
            tile_num = int(f.stem.split("tile")[-1])
            tile_idx = tile_num - 6
            dx = state.res_km[tile_idx]
        else:
            dx = state.res_km[0]

        _gen_ensemble(file_stds[str(f)], tmp_f, f, target_vars, rng, dx)
