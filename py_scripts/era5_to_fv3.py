from __future__ import annotations

from pathlib import Path

import xarray as xr
import xesmf as xe

from fv3gfs_state import state


def to_fv3_sfc_grid(
    era5_data: xr.Dataset,
    tile_data: xr.Dataset,
) -> xr.Dataset:
    """
    Remaps ERA5 data to FV3 Land Surface dimensions (yaxis_1, xaxis_1, zaxis_1).
    Uses bilinear for state variables (temp) and conservative for fluxes/fractions.
    """

    # 1. Source Grid (ERA5)
    src_grid = xr.Dataset(
        {
            "lat": era5_data["lat"],
            "lon": era5_data["lon"],
        }
    )

    # 2. Destination Grid (FV3 Tile)
    # Most land vars are on geolat/geolon (centers)
    dst_grid = xr.Dataset(
        {
            "lat": tile_data["geolat"],
            "lon": tile_data["geolon"],
        }
    )

    # 3. Create Regridders
    # Use bilinear for 'state' vars (temperatures, roughness)
    regrid_bilinear = xe.Regridder(src_grid, dst_grid, method="bilinear", periodic=True)

    # Use conservative for 'quantity/fraction' vars (soil moisture, precip, vfrac)
    # Note: 'conservative' requires bounds. If bounds aren't in your data,
    # bilinear is a safe fallback, though less physically accurate for land fractions.
    regrid_conserve = xe.Regridder(src_grid, dst_grid, method="bilinear", periodic=True)

    # Variables categorized by remapping type
    conserve_vars = ["slmsk", "vfrac", "canopy", "tprcp", "smc", "slc", "fice", "hice"]

    remapped_vars = {}

    for var_name, target_da in tile_data.data_vars.items():
        if var_name not in era5_data:
            continue

        # Select appropriate regridder
        method = regrid_conserve if var_name in conserve_vars else regrid_bilinear

        # Perform remapping
        # xESMF automatically handles the 'Time' and 'zaxis_1' (soil levels)
        # as "extra dimensions" as long as they are not lat/lon.
        out_da = method(era5_data[var_name])

        # Match dimension names exactly: lat -> yaxis_1, lon -> xaxis_1
        # This keeps the zaxis_1 and Time dimensions intact.
        rename_dict = {}
        if "lat" in out_da.dims:
            rename_dict["lat"] = "yaxis_1"
        if "lon" in out_da.dims:
            rename_dict["lon"] = "xaxis_1"
        out_da = out_da.rename(rename_dict)

        # Restore vertical coordinates if it's a soil variable (stc, smc, slc)
        if "zaxis_1" in target_da.dims:
            out_da = out_da.assign_coords(zaxis_1=target_da.zaxis_1)

        # Sync metadata
        out_da.attrs = target_da.attrs
        remapped_vars[var_name] = out_da

    # 4. Final Assembly
    new_ds = xr.Dataset(remapped_vars)

    # Re-attach coordinates (geolon, geolat, Time)
    new_ds = new_ds.assign_coords(tile_data.coords)
    new_ds.attrs = tile_data.attrs

    return new_ds


def to_fv3_atm_grid(
    era5_data: xr.Dataset,
    tile_data: xr.Dataset,
) -> xr.Dataset:
    """
    Remaps ERA5 data to an FV3 tile grid, accounting for staggered
    dimensions (latp, lonp) and matching metadata.
    """

    # 1. Define the source grid (ERA5 is usually a simple Lat/Lon)
    ll_grid = xr.Dataset(
        {
            "lat": era5_data["lat"],
            "lon": era5_data["lon"],
        }
    )

    remapped_vars = {}

    # 2. Iterate through variables in the target tile_data
    for var_name, target_da in tile_data.data_vars.items():
        # Determine the correct target coordinates based on dimensions
        # A-grid (Center): (lat, lon)
        # C-grid (West/East faces): (lat, lonp)
        # C-grid (South/North faces): (latp, lon)
        if "latp" in target_da.dims and "lonp" in target_da.dims:
            # Corner/D-grid if applicable
            curr_lat, curr_lon = tile_data["geolat_b"], tile_data["geolon_b"]
        elif "lonp" in target_da.dims:
            curr_lat, curr_lon = tile_data["geolat_w"], tile_data["geolon_w"]
        elif "latp" in target_da.dims:
            curr_lat, curr_lon = tile_data["geolat_s"], tile_data["geolon_s"]
        else:
            curr_lat, curr_lon = tile_data["geolat"], tile_data["geolon"]

        # Create destination grid for this specific stagger
        dest_grid = xr.Dataset({"lat": curr_lat, "lon": curr_lon})

        # 3. Initialize Regridder
        # Note: reuse_weights=True can speed this up if you have many time steps
        regridder = xe.Regridder(ll_grid, dest_grid, method="bilinear", periodic=True)

        # 4. Remap the corresponding variable from ERA5
        # We assume the variable name exists in era5_data or is mapped manually
        if var_name in era5_data:
            out_da = regridder(era5_data[var_name])

            # Match dimension names to the target (lat/lon -> latp/lonp if needed)
            out_da = out_da.rename(
                {
                    d: target_da.dims[i]
                    for i, d in enumerate(out_da.dims)
                    if d in ["lat", "lon"]
                }
            )

            # Transfer attributes
            out_da.attrs = target_da.attrs
            remapped_vars[var_name] = out_da

    # 5. Merge all remapped variables into one Dataset
    new_ds = xr.Dataset(remapped_vars)

    # Assign the original coordinates from the tile_data
    new_ds = new_ds.assign_coords(tile_data.coords)
    new_ds.attrs = tile_data.attrs

    return new_ds


def remap_era5_to_fv3cube() -> xr.Dataset:
    atm_files = list(Path(state.run).glob("gfs_data*.nc"))
    sfc_files = list(Path(state.input).glob("sfc_data*.nc"))
    era5_atm_file = Path(state.rundir) / "era5_atm.nc"  # same levels as tile_data
    era5_sfc_file = Path(state.rundir) / "era5_sfc.nc"  # same levels as tile_data

    era5_atm = xr.open_dataset(era5_atm_file).load()
    era5_sfc = xr.open_dataset(era5_sfc_file).load()

    for f in atm_files:
        tmp_f = f.with_suffix(".tmp")
        f.rename(tmp_f)
        with xr.open_dataset(tmp_f) as ds:
            new_atm_ds = to_fv3_atm_grid(era5_atm, ds.load())

        new_atm_ds.to_netcdf(f)
        f.unlink()

    for f in sfc_files:
        tmp_f = f.with_suffix(".tmp")
        f.rename(tmp_f)
        with xr.open_dataset(tmp_f) as ds:
            new_sfc_ds = to_fv3_sfc_grid(era5_sfc, ds.load())
        new_sfc_ds.to_netcdf(f)
        f.unlink()
