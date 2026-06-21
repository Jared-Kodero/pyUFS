from __future__ import annotations

from pathlib import Path
from typing import Literal

import xarray as xr
import xesmf as xe
from fv3_runtime import log
from fv3_state import state
from fv3_utils import cp

# Physical bounds for valid volumetric soil moisture (m3 m-3). Used for both
# the validity mask and the clip so the two are guaranteed consistent.
SM_MIN = 0.01
SM_MAX = 0.99


def load_climo(path: Path, data_var: str) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Climatology file not found: {path}")

    cdate = state.init_datetime
    ds = xr.open_dataset(path, engine="netcdf4")

    if data_var not in ds.data_vars:
        raise KeyError(f"Variable {data_var} not found in climatology file {path}")

    ds = ds.sel(time=ds.time.dt.month == cdate.month)

    if ds.sizes["time"] == 0:
        msg = f"No climatology data found for month {cdate.month} in file {path}"
        raise ValueError(msg)

    ds = ds.sortby(["lat", "lon"])
    ds = ds.where((ds[data_var] >= SM_MIN) & (ds[data_var] <= SM_MAX), other=1.0)

    return ds


def to_fv3_grid(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray,
    method: Literal["bilinear", "conservative"] = "bilinear",
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
    )

    # init dims ('Time', 'yaxis_1', 'xaxis_1', 'zaxis_1')
    # restart dims ('Time', 'yaxis_1', 'xaxis_1', 'zaxis_1', 'zaxis_2', 'zaxis_3')

    out = regridder(grid_in)

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
            ds.to_netcdf(grid_file, engine="netcdf4")

    ds = xr.open_dataset(grid_file, decode_cf=False, engine="netcdf4").load()
    return ds


def do_hold(p: dict, backup_dir: Path, restart_no: int):

    if restart_no == 0:
        return

    prev_restart = restart_no - 1

    log.info(f"`do_hold` is set to true: using sm state from restart {prev_restart}")

    for tile in p["tiles"]:
        nest_idx = f"nest{(tile - 5):02d}." if tile > 6 else ""
        filename = Path(f"sfc_data.{nest_idx}tile{tile}.nc")

        in_path = Path(state.input) / filename
        prev_path = backup_dir / f"{filename.stem}.r{prev_restart:03d}.perturbed.nc"
        orig_path = backup_dir / f"{filename.stem}.r{restart_no:03d}.original.nc"

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        if not prev_path.exists():
            raise FileNotFoundError(f"Previous perturbed file not found: {prev_path}")

        cp(in_path, orig_path)
        in_path.unlink()

        with xr.open_dataset(prev_path, decode_cf=False, engine="netcdf4") as ds:
            ds = ds.load()
            for v in ds.data_vars:
                ds[v] = ds[v].drop_attrs(deep=True).drop_encoding()
            ds.to_netcdf(in_path)


def do_nudge_soil_moisture(
    p: dict, backup_dir: Path, restart_no: int, sm_clim_path: Path
):

    tau_hours = p.get("tau_hours", 24)
    dt_hours = state.run_nhours
    use_climo = p.get("use_climo", False)

    alpha = dt_hours / tau_hours
    if alpha > 1.0:
        log.info(
            f"dt_hours ({dt_hours}) >= tau_hours ({tau_hours}): alpha clipped to 1.0; nudge reduces to full replacement"
        )
    alpha = min(max(alpha, 0.0), 1.0)

    for tile in p["tiles"]:
        nest_idx = f"nest{(tile - 5):02d}." if tile > 6 else ""
        filename = Path(f"sfc_data.{nest_idx}tile{tile}.nc")

        in_path = Path(state.input) / filename
        backup_path = backup_dir / f"{filename.stem}.r{restart_no:03d}.perturbed.nc"
        orig_path = backup_dir / f"{filename.stem}.r{restart_no:03d}.original.nc"

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        cp(in_path, orig_path)

        grid = load_grid(filename, tile)

        # Read fully into memory and release the handle before writing back.
        ds = xr.open_dataset(in_path, decode_cf=False, engine="netcdf4")
        ds = ds.load()

        if use_climo:
            log.info("Nudging soil moisture towards climatological mean")
            ds_ref = load_climo(sm_clim_path, p["target_var"])
            ds_ref = ds_ref.mean(dim="time", skipna=True)
            ds_ref = ds_ref.squeeze(drop=True)
            ds_ref = to_fv3_grid(ds_ref, grid)
            ds_ref = ds_ref.load()
        else:
            log.info("Nudging soil moisture towards state from last restart")
            ref_path = (
                backup_dir / f"{filename.stem}.r{restart_no - 1:03d}.perturbed.nc"
            )
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Previous perturbed file not found: {ref_path}"
                )
            with xr.open_dataset(ref_path, decode_cf=False, engine="netcdf4") as ds_ref:
                ds_ref = ds_ref.load()

        ice = None
        if "smc" in ds and "slc" in ds:
            ice = ds["smc"] - ds["slc"]

        for z in p["soil_layers"]:
            v = p["target_var"]
            if v not in ds.data_vars:
                continue

            layer = ds[v].isel(zaxis_1=z)
            ref_layer = ds_ref[v].isel(zaxis_1=z)

            is_valid = (layer >= SM_MIN) & (layer <= SM_MAX)

            updated = (1.0 - alpha) * layer + alpha * ref_layer
            updated = updated.clip(min=SM_MIN, max=SM_MAX)

            coord_val = ds.zaxis_1.values[z]
            ds[v].loc[{"zaxis_1": coord_val}] = xr.where(is_valid, updated, layer)

        # reconstruct slc from updated smc
        if ice is not None and "smc" == p["target_var"]:
            smc_new = ds["smc"]
            slc_new = smc_new - ice
            slc_new = xr.where(slc_new < 0, 0, slc_new)
            slc_new = xr.where(slc_new > smc_new, smc_new, slc_new)

            ds["slc"] = slc_new

        for v in ds.data_vars:
            ds[v] = ds[v].drop_attrs(deep=True).drop_encoding()

        # Persist the nudged state so that a subsequent restart can use it as
        # its reference, then overwrite the live input.
        ds.to_netcdf(backup_path)
        ds.close()
        in_path.unlink()
        cp(backup_path, in_path)


def std_shift(
    v: str,
    z: int,
    layer: xr.DataArray,
    climo_path: Path,
    is_valid: xr.DataArray,
    grid: xr.Dataset,
    mean_scale: float,
    anom_scale: float,
    n_sigma: float,
    constant_value: float,
    pert_logs: list,
) -> xr.DataArray:

    if climo_path is not None:
        climo_ds = load_climo(climo_path, v)
        climo_layer = climo_ds[v].isel(zaxis_1=z, drop=False)
        std = climo_layer.std(dim="time", skipna=True).load()
        std = to_fv3_grid(std, grid)
    else:
        data = layer.where(is_valid)
        std = float(data.std(skipna=True))

    updated = layer + (std * n_sigma)
    updated = updated.clip(SM_MIN, SM_MAX)
    layer = xr.where(is_valid, updated, layer)

    pert_logs.append(f"Applied std_shift to {v} with n_sigma={n_sigma}")
    return layer


def climo_mean(
    v: str,
    z: int,
    layer: xr.DataArray,
    climo_path: Path,
    is_valid: xr.DataArray,
    grid: xr.Dataset,
    mean_scale: float,
    anom_scale: float,
    n_sigma: float,
    constant_value: float,
    pert_logs: list,
) -> xr.DataArray:

    if climo_path is None:
        raise ValueError("`climo_file` must be provided when using `climo_mean` method")

    climo_ds = load_climo(climo_path, v)
    climo_layer = climo_ds[v].isel(zaxis_1=z, drop=False)
    climo = climo_layer.mean(dim="time", skipna=True).load()
    climo = to_fv3_grid(climo, grid)

    layer = xr.where(is_valid, climo, layer)

    pert_logs.append(f"Applied climo_mean to {v} ")

    return layer


def anom_shift(
    v: str,
    z: int,
    layer: xr.DataArray,
    climo_path: Path,
    is_valid: xr.DataArray,
    grid: xr.Dataset,
    mean_scale: float,
    anom_scale: float,
    n_sigma: float,
    constant_value: float,
    pert_logs: list,
) -> xr.DataArray:

    data = layer.where(is_valid)
    mu = data.mean(skipna=True)
    anomaly = layer - mu

    updated = mu + (1.0 + anom_scale) * anomaly
    updated = updated.clip(SM_MIN, SM_MAX)

    layer = xr.where(is_valid, updated, layer)

    pert_logs.append(f"Applied anom_shift to {v} with anom_scale={anom_scale}")

    return layer


def mean_shift(
    v: str,
    z: int,
    layer: xr.DataArray,
    climo_path: Path,
    is_valid: xr.DataArray,
    grid: xr.Dataset,
    mean_scale: float,
    anom_scale: float,
    n_sigma: float,
    constant_value: float,
    pert_logs: list,
) -> xr.DataArray:

    data = layer.where(is_valid)
    updated = data * (1.0 + mean_scale)
    updated = updated.clip(SM_MIN, SM_MAX)

    layer = xr.where(is_valid, updated, layer)

    pert_logs.append(f"Applied mean_shift to {v} with mean_scale={mean_scale}")

    return layer


def constant_fill(
    v: str,
    z: int,
    layer: xr.DataArray,
    climo_path: Path,
    is_valid: xr.DataArray,
    grid: xr.Dataset,
    mean_scale: float,
    anom_scale: float,
    n_sigma: float,
    constant_value: float,
    pert_logs: list,
) -> xr.DataArray:

    if constant_value == "mean":
        data = layer.where(is_valid)
        mean_val = data.mean(skipna=True)
        updated = xr.full_like(layer, fill_value=mean_val)
    else:
        updated = xr.full_like(layer, fill_value=constant_value)

    layer = xr.where(is_valid, updated, layer)
    pert_logs.append(f"Applied constant_fill to {v} with fill_value={constant_value}")
    return layer


def adjust_soil_moisture(
    p: dict, backup_dir: Path, methods, restart_no: int, sm_clim_path: Path
):
    pert_logs = []
    mean_scale = p.get("mean_scale")
    anom_scale = p.get("anom_scale")
    n_sigma = p.get("n_sigma")
    use_climo = p.get("use_climo", False)
    constant_value = p.get("fill_value", None)
    climo_file = p.get("climo_file", None)

    perturbation_methods = {
        "std_shift": std_shift,
        "climo_mean": climo_mean,
        "anom_shift": anom_shift,
        "mean_shift": mean_shift,
        "constant_fill": constant_fill,
    }

    if isinstance(methods, str):
        methods = [methods]

    climo_path = None
    if use_climo:
        if climo_file is not None:
            climo_path = Path(climo_file)
        else:
            climo_path = sm_clim_path
        log.info(f"Using reference climatology: {climo_path}")

    for tile in p["tiles"]:
        nest_idx = f"nest{(tile - 5):02d}." if tile > 6 else ""
        filename = Path(f"sfc_data.{nest_idx}tile{tile}.nc")

        in_path = Path(state.input) / filename
        backup_path = backup_dir / f"{filename.stem}.r{restart_no:03d}.perturbed.nc"
        orig_path = backup_dir / f"{filename.stem}.r{restart_no:03d}.original.nc"

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        cp(in_path, orig_path)

        grid = load_grid(filename, tile)

        ds = xr.open_dataset(in_path, decode_cf=False, engine="netcdf4")
        ds = ds.load()

        ice = None
        if "smc" in ds and "slc" in ds:
            ice = ds["smc"] - ds["slc"]

        for z in p["soil_layers"]:
            v = p["target_var"]
            if v not in ds.data_vars:
                continue

            layer = ds[v].isel(zaxis_1=z)
            is_valid = (layer >= SM_MIN) & (layer <= SM_MAX)

            new_layer = layer.copy()

            for m in methods:
                new_layer = perturbation_methods[m](
                    v,
                    z,
                    new_layer,
                    climo_path,
                    is_valid,
                    grid,
                    mean_scale,
                    anom_scale,
                    n_sigma,
                    constant_value,
                    pert_logs,
                )

            # Write the layer once, after all methods have been chained.
            coord_val = ds.zaxis_1.values[z]
            ds[v].loc[{"zaxis_1": coord_val}] = new_layer

        # reconstruct slc from updated smc
        if ice is not None and "smc" == p["target_var"]:
            smc_new = ds["smc"]
            slc_new = smc_new - ice
            slc_new = xr.where(slc_new < 0, 0, slc_new)
            slc_new = xr.where(slc_new > smc_new, smc_new, slc_new)

            ds["slc"] = slc_new

        for v in ds.data_vars:
            ds[v] = ds[v].drop_attrs(deep=True).drop_encoding()

        ds.to_netcdf(backup_path)

        ds.close()

        in_path.unlink()
        cp(backup_path, in_path)

    for log_entry in dict.fromkeys(pert_logs):
        log.info(log_entry)


def apply_perturbations():
    """Apply soil moisture perturbations to the current input state according to the `sm_perturbations` config in the state."""

    perturbations = state.get("sm_perturbations", None)
    if not perturbations:
        return

    sm_clim_path = Path(state.fixed_dir) / "era5" / "sm_monthly_1950_2025.nc"

    restart_no = state.restart_no
    total_restarts = state.total_restarts
    max_restart_index = total_restarts - 1

    if not isinstance(perturbations, dict):
        raise TypeError("`sm_perturbations` config must be a mapping")

    p = perturbations

    allowed = ("std_shift", "mean_shift", "anom_shift", "constant_fill", "climo_mean")
    required = ("target_var", "soil_layers", "tiles", "method")
    missing = [k for k in required if k not in p]
    if missing:
        raise KeyError(f"Missing perturbation keys: {missing}")

    soft_keys = (
        "mean_scale",
        "anom_scale",
        "n_sigma",
        "tau_hours",
        "use_climo",
        "do_hold",
        "do_nudge",
        "climo_file",
        "apply_on_restarts",
        "fill_value",
    )

    for k in p.keys():
        if k not in required and k not in soft_keys:
            raise ValueError(f"Unknown key in perturbation config: {k}")

    methods = p.get("method")

    if isinstance(methods, str):
        check_methods = [methods]
    else:
        check_methods = methods

    for m in check_methods:
        if m not in allowed:
            raise ValueError(f"`method` must be one of {allowed}. Got `{m}`")

    # Conditional parameter checks

    if "std_shift" in check_methods and "n_sigma" not in p:
        raise KeyError("If method includes 'std_shift', you must provide key 'n_sigma'")

    if "mean_shift" in check_methods and "mean_scale" not in p:
        raise KeyError(
            "If method includes 'mean_shift', you must provide key 'mean_scale'"
        )

    if "anom_shift" in check_methods and "anom_scale" not in p:
        raise KeyError(
            "If method includes 'anom_shift', you must provide key 'anom_scale'"
        )
    if "constant_fill" in check_methods and "fill_value" not in p:
        raise KeyError(
            "If method includes 'constant_fill', you must provide key 'fill_value'"
        )

    hold = p.get("do_hold", False)
    do_nudge = p.get("do_nudge", False)

    if hold and do_nudge:
        raise ValueError("only one of `do_hold` and `do_nudge` can be true")

    if isinstance(p["soil_layers"], (int, float)):
        p["soil_layers"] = [int(p["soil_layers"])]

    # Determine whether this perturbation applies to the current restart.
    apply_on_restarts = p.get("apply_on_restarts", None)

    if apply_on_restarts is None:
        return
    elif apply_on_restarts == "all":
        apply_on_restarts = list(range(restart_no, max_restart_index + 1))
    elif isinstance(apply_on_restarts, int):
        apply_on_restarts = [apply_on_restarts]
    elif isinstance(apply_on_restarts, list):
        apply_on_restarts = [int(r) for r in apply_on_restarts]
    else:
        raise TypeError(
            "`apply_on_restarts` be one of None, 'all', int, or list of ints"
        )

    if restart_no not in apply_on_restarts:
        return

    log.info(
        f"sm_perturbations detected for restart {restart_no}; applying perturbations"
    )

    backup_dir = Path(state.ic_data) / "perts"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if restart_no >= 1:
        if do_nudge:
            do_nudge_soil_moisture(p, backup_dir, restart_no, sm_clim_path)
        elif hold:
            do_hold(p, backup_dir, restart_no)
        else:
            adjust_soil_moisture(p, backup_dir, methods, restart_no, sm_clim_path)
    else:
        adjust_soil_moisture(p, backup_dir, methods, restart_no, sm_clim_path)

    log.info("Finished applying soil moisture perturbations")
