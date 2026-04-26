from pathlib import Path

import xarray as xr
import xesmf as xe
from fv3gfs_runtime import log
from fv3gfs_state import state
from fv3gfs_utils import cp


def load_climo(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Climatology file not found: {path}")

    cdate = state.init_datetime
    ds = xr.open_dataset(path, engine="netcdf4")
    ds = ds.sel(time=ds.time.dt.month == cdate.month)

    if ds.sizes["time"] == 0:
        msg = f"No climatology entries found for month {cdate.month} in file {path}."
        msg += "\nUpdate your climatology file to include data for this month."
        raise ValueError(msg)

    return ds


def load_coords_ds(filename: Path, tile: int) -> xr.Dataset:
    grid_file = Path(state.init_data) / "PERTURBATIONS" / f"tile.{tile}.grid.nc"
    if not grid_file.exists():
        path = Path(state.init_data) / "INIT_INPUT" / filename
        ds = xr.open_dataset(path, decode_cf=False, engine="netcdf4")
        ds = ds[["geolat", "geolon"] + list(ds.coords) + list(ds.dims)]
        ds.to_netcdf(grid_file, engine="netcdf4")
    else:
        ds = xr.open_dataset(grid_file, decode_cf=False, engine="netcdf4")
    return ds


def to_fv3cube_grid(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray,
) -> xr.Dataset:
    """
    Remap source dataset to the grid of the destination dataset using bilinear interpolation.
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
        method="bilinear",
    )

    out = regridder(grid_in)
    if "Time" not in out.coords:
        out = out.expand_dims("Time", axis=0)

    out.attrs = grid_out.attrs
    out_coords = set(out.coords)
    for c in out_coords:
        if c in grid_out.coords:
            out[c].attrs = grid_out[c].attrs
        else:
            out = out.drop(c)

    out["Time"] = grid_out["Time"]

    new_dims = [d for d in grid_out.dims if d in out.dims]
    out = out.transpose(*new_dims)

    out.attrs = grid_out.attrs

    return out


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


def do_nudge_soil_moisture(p, backup_dir, restart_no):

    tau_hours = p.get("tau_hours", 24)
    dt_hours = state.run_nhours
    use_climo = p.get("use_climo", False)

    alpha = dt_hours / tau_hours
    alpha = min(max(alpha, 0.0), 1.0)

    for tile in p["tiles"]:
        nest_idx = f"nest{(tile - 5):02d}." if tile > 6 else ""
        filename = Path(f"sfc_data.{nest_idx}tile{tile}.nc")

        in_path = Path(state.input) / filename

        ds = xr.open_dataset(in_path, decode_cf=False, engine="netcdf4")

        if use_climo:
            log.info("Nudging soil moisture towards climatological mean")

            ref_path = Path(state.fix) / "era5" / "sm_monthly_1980_2020.nc"
            ds_ref = load_climo(ref_path)
            ds_ref = ds_ref.mean(dim="time", skipna=True)
            ds_ref = ds_ref.squeeze(drop=True)
            ds_ref = to_fv3cube_grid(ds_ref, load_coords_ds(filename, tile))

        else:
            log.info("Nudging soil moisture towards state from last restart")
            ref_path = backup_dir / f"{filename.stem}.r00{restart_no - 1}.perturbed.nc"
            ds_ref = xr.open_dataset(ref_path, decode_cf=False, engine="netcdf4")

        ds = ds.load()
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

            is_valid = (layer > 0.01) & (layer < 0.99)

            updated = (1.0 - alpha) * layer + alpha * ref_layer
            updated = updated.clip(min=0.01, max=0.99)

            coord_val = ds.zaxis_1.values[z]
            ds[v].loc[{"zaxis_1": coord_val}] = xr.where(is_valid, updated, layer)

        # reconstruct slc from updated smc
        if ice is not None and "smc" in p["target_var"]:
            smc_new = ds["smc"]
            slc_new = smc_new - ice
            slc_new = xr.where(slc_new < 0, 0, slc_new)
            slc_new = xr.where(slc_new > smc_new, smc_new, slc_new)

            ds["slc"] = slc_new

        for v in ds.data_vars:
            ds[v] = ds[v].drop_attrs(deep=True).drop_encoding()

        ds.to_netcdf(in_path)

        ds_ref.close()
        ds.close()


def adjust_soil_moisture(p, backup_dir, methods, restart_no):

    mean_scale = p.get("mean_scale")
    anom_scale = p.get("anom_scale")
    n_sigma = p.get("n_sigma")
    use_climo = p.get("use_climo", False)
    constant_value = p.get("fill_value", None)
    climo_file = p.get("climo_file", None)

    if isinstance(methods, str):
        methods = [methods]

    climo_path = None
    if use_climo:
        if climo_file is not None:
            climo_path = Path(climo_file)
        else:
            climo_path = Path(state.fix) / "era5" / "sm_monthly_1980_2020.nc"
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

        with xr.open_dataset(in_path, decode_cf=False, engine="netcdf4") as ds:
            ds = ds.load()

            ice = None
            if "smc" in ds and "slc" in ds:
                ice = ds["smc"] - ds["slc"]

            for z in p["soil_layers"]:
                v = p["target_var"]
                if v not in ds.data_vars:
                    continue

                layer = ds[v].isel(zaxis_1=z)
                is_valid = (layer > 0) & (layer < 1)

                layer_new = layer

                for method in methods:
                    if method == "std_shift":
                        data = layer_new.where(is_valid)
                        if climo_path is not None:
                            climo_ds = load_climo(climo_path)
                            climo_layer = climo_ds[v].isel(zaxis_1=z, drop=False).load()
                            std = climo_layer.std(dim="time", skipna=True)
                            std = to_fv3cube_grid(std, load_coords_ds(filename, tile))

                        else:
                            std = data.std(skipna=True)
                            std = float(std)

                        updated = layer_new + (std * n_sigma)
                        updated = updated.clip(0.01, 0.99)
                        layer_new = xr.where(is_valid, updated, layer_new)

                    elif method == "anom_shift":
                        data = layer_new.where(is_valid)
                        mu = data.mean(skipna=True)
                        anomaly = layer_new - mu

                        updated = mu + (1.0 + anom_scale) * anomaly
                        updated = updated.clip(0.01, 0.99)
                        layer_new = xr.where(is_valid, updated, layer_new)

                    elif method == "mean_shift":
                        data = layer_new.where(is_valid)
                        updated = data * (1.0 + mean_scale)
                        updated = updated.clip(0.01, 0.99)
                        layer_new = xr.where(is_valid, updated, layer_new)

                    elif method == "constant_fill":
                        if constant_value == "mean":
                            data = layer_new.where(is_valid)
                            mean_val = data.mean(skipna=True)
                            updated = xr.full_like(layer_new, fill_value=mean_val)
                        else:
                            updated = xr.full_like(layer_new, fill_value=constant_value)

                        layer_new = xr.where(is_valid, updated, layer_new)

                    else:
                        raise ValueError(f"Unknown method: {method}")

                    coord_val = ds.zaxis_1.values[z]
                    ds[v].loc[{"zaxis_1": coord_val}] = layer_new

            # reconstruct slc from updated smc
            if ice is not None and "smc" in p["target_var"]:
                smc_new = ds["smc"]
                slc_new = smc_new - ice
                slc_new = xr.where(slc_new < 0, 0, slc_new)
                slc_new = xr.where(slc_new > smc_new, smc_new, slc_new)

                ds["slc"] = slc_new

            for v in ds.data_vars:
                ds[v] = ds[v].drop_attrs(deep=True).drop_encoding()

            ds.to_netcdf(backup_path)

        in_path.unlink()
        cp(backup_path, in_path)


def apply_perturbations():

    perts = state.get("sm_perturbations", None)
    if not perts:
        return

    if state.n_nests == 0 or state.restart_no == 0:
        return

    log.info("sm_perturbations detected; applying perturbations")

    restart_no = state.restart_no
    total_restarts = state["total_restarts"]
    max_restart_index = total_restarts - 1

    if not isinstance(perts, list):
        perts = [perts]

    allowed = ("std_shift", "mean_shift", "anom_shift", "constant_fill")

    for i, p in enumerate(perts):
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
            raise KeyError(
                "If method includes 'std_shift', you must provide key 'n_sigma'"
            )

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

        apply_on_restarts = p.get("apply_on_restarts", None)

        if apply_on_restarts is None:
            apply_on_restarts = list(range(restart_no, max_restart_index + 1))
        elif isinstance(apply_on_restarts, int):
            apply_on_restarts = [apply_on_restarts]
        elif isinstance(apply_on_restarts, list):
            apply_on_restarts = [int(r) for r in apply_on_restarts]
        else:
            raise TypeError("`apply_on_restarts` must be None, int, or list[int]")

        if restart_no not in apply_on_restarts:
            continue

        if isinstance(p["soil_layers"], (int, float)):
            p["soil_layers"] = [int(p["soil_layers"])]

        backup_dir = Path(state.init_data) / "PERTURBATIONS" / f"{i}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        if restart_no > 1:
            if do_nudge:
                do_nudge_soil_moisture(p, backup_dir, restart_no)
            elif hold:
                do_hold(p, backup_dir, restart_no)
            else:
                adjust_soil_moisture(p, backup_dir, methods, restart_no)
        else:
            adjust_soil_moisture(p, backup_dir, methods, restart_no)
