import logging

import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from metpy.units import units

log = logging.getLogger("USER_DERIVED_VARS")

g = 9.80665
R_earth_m = 6371000.0  # Earth radius [m]


def _check_dtype(da: xr.DataArray, dtype: str):
    """Check if the DataArray has the specified dtype, and convert if necessary."""
    if da.dtype != dtype:
        da = da.astype(dtype)
    return da


def calc_moisture_trans(ds: xr.Dataset) -> xr.Dataset:
    required_vars = ["q", "u", "v", "ps"]
    if not all(var in ds.data_vars for var in required_vars):
        log.warning(
            f"Missing variables {required_vars} for moisture transport calculation. Skipping!"
        )
        return ds

    ds = ds.sortby("level", ascending=False)

    # Get delta pressure and pressure at layer midpoints

    ds["level"] = ds["level"] * 100.0
    dp = abs(ds["level"].diff("level"))
    q = ds["q"].isel(level=slice(1, None))
    u = ds["u"].isel(level=slice(1, None))
    v = ds["v"].isel(level=slice(1, None))
    p = ds["level"].isel(level=slice(1, None))

    # Set layers below surface pressure to zero
    # dp = dp.broadcast_like(q)
    p3d = p.broadcast_like(q)
    dp = dp.where(p3d <= ds.ps, 0.0)

    qu = q * u
    qv = q * v

    ivtu = (qu * dp).sum(dim="level") / g
    ivtv = (qv * dp).sum(dim="level") / g

    ivtu = _check_dtype(ivtu, ds["q"].dtype)
    ivtv = _check_dtype(ivtv, ds["q"].dtype)

    # Magnitude of integrated vapor transport
    ivt = (ivtu**2 + ivtv**2) ** 0.5

    ivt = _check_dtype(ivt, ds["q"].dtype)

    ivtu.name = "ivtu"
    ivtv.name = "ivtv"
    ivt.name = "ivt"

    for var in [ivtu, ivtv, ivt]:
        var.attrs = {}
        var.attrs["units"] = "kg m-1 s-1"

    ivtu.attrs["long_name"] = "Integrated vapor transport zonal component"
    ivtv.attrs["long_name"] = "Integrated vapor transport meridional component"
    ivt.attrs["long_name"] = "Integrated vapor transport magnitude"

    ds["ivtu"] = ivtu
    ds["ivtv"] = ivtv
    ds["ivt"] = ivt

    lat_rad = np.deg2rad(ds["lat"])
    lon_rad = np.deg2rad(ds["lon"])
    coslat = np.cos(lat_rad)
    coslat = coslat.clip(min=1e-6)

    # 1. Create Radian-based versions of your IVT components
    # This ensures both variables share the exact same coordinate system
    ivtu_rad = ivtu.assign_coords(lat=lat_rad, lon=lon_rad)
    ivtv_rad = ivtv.assign_coords(lat=lat_rad, lon=lon_rad)

    # 2. Calculate derivatives using the radian-indexed arrays
    d_ivtu_dlon = ivtu_rad.differentiate("lon")

    # Ensure coslat also uses the radian coordinate for alignment
    coslat_rad = coslat.assign_coords(lat=lat_rad)
    v_coslat = ivtv_rad * coslat_rad
    d_vcoslat_dlat = v_coslat.differentiate("lat")

    # 3. Combine - now the coordinates match perfectly!
    vimfc = -(d_ivtu_dlon + d_vcoslat_dlat) / (R_earth_m * coslat_rad)

    # 4. Restore original degrees for the final output
    vimfc = vimfc.assign_coords(lat=ds["lat"], lon=ds["lon"])
    vimfc = _check_dtype(vimfc, ds["q"].dtype)

    vimfc.name = "vimfc"
    vimfc.attrs["units"] = "kg m-2 s-1"
    vimfc.attrs["short_name"] = "vimfc"
    vimfc.attrs["long_name"] = "Vertically Integrated Moisture Flux Convergence"

    ds["vimfc"] = vimfc

    ds["level"] = ds["level"] / 100.0

    return ds


def calc_bulk_richardson(ds: xr.Dataset) -> xr.Dataset:
    required_vars = ["cape", "shear06"]
    if not all(var in ds.data_vars for var in required_vars):
        log.warning(
            f"Missing variables {required_vars} for bulk Richardson number calculation. Skipping!"
        )
        return ds

    ds["brn"] = ds["cape"] / (0.5 * ds["shear06"] ** 2)
    ds["brn"] = ds["brn"].clip(0, 100)
    ds["brn"] = _check_dtype(ds["brn"], ds["t"].dtype)
    ds["brn"].attrs["units"] = "dimensionless"
    ds["brn"].attrs["long_name"] = "Bulk Richardson number"

    return ds


def calc_mse(ds: xr.Dataset) -> xr.Dataset:
    required_vars = ["t", "q", "z"]
    if not all(var in ds.data_vars for var in required_vars):
        log.warning(
            f"Missing variables {required_vars} for moist static energy calculation. Skipping!"
        )
        return ds

    t = ds["t"] * units.degK
    q = ds["q"] * 1000 * units("g/kg")
    z = ds["z"] * units.meter

    mse = mpcalc.moist_static_energy(z, t, q)
    mse = mse.metpy.dequantify()
    mse = _check_dtype(mse, ds["t"].dtype)

    mse.name = "mse"
    mse.attrs["units"] = "kJ/kg"
    mse.attrs["short_name"] = "mse"
    mse.attrs["long_name"] = "Moist Static Energy"

    ds["mse"] = mse
    return ds


def calc_evaporative_fraction(ds: xr.Dataset) -> xr.Dataset:
    required_vars = ("shtfl", "lhtfl")
    if not all(var in ds.data_vars for var in required_vars):
        log.warning(
            f"Missing variables {required_vars} for Bowen ratio calculation. Skipping!"
        )
        return ds
    epsilon = 1e-6
    ds["evf"] = ds["lhtfl"] / ((ds["lhtfl"] + ds["shtfl"]) + epsilon)
    ds["evf"] = ds["evf"].clip(0, 1)
    ds["evf"] = _check_dtype(ds["evf"], ds["shtfl"].dtype)
    ds["evf"].attrs["units"] = "dimensionless"
    ds["evf"].attrs["long_name"] = "Evaporative Fraction"
    return ds


def calc_vpd2m(ds: xr.Dataset) -> xr.Dataset:
    """Calculate 2 meter vapor pressure deficit using MetPy."""
    required_vars = ("t2m", "dpt2m")
    if not all(var in ds.data_vars for var in required_vars):
        return ds

    t2m = ds["t2m"] * units.kelvin
    td2m = ds["dpt2m"] * units.kelvin

    es = mpcalc.saturation_vapor_pressure(t2m).metpy.convert_units("hPa")
    e = mpcalc.saturation_vapor_pressure(td2m).metpy.convert_units("hPa")

    es = es.metpy.dequantify()
    e = e.metpy.dequantify()

    vpd2m = es - e
    vpd2m = _check_dtype(vpd2m, ds["t2m"].dtype)
    ds["vpd2m"] = vpd2m.clip(min=0.0)
    ds["vpd2m"].attrs["units"] = "hPa"
    ds["vpd2m"].attrs["long_name"] = "2 meter vapor pressure deficit"

    return ds


# implement your derived func and call it here in calc_derived_vars
def calc_derived_vars(ds: xr.Dataset) -> xr.Dataset:
    """Calculate derived variables and add them to the dataset."""
    ds = calc_moisture_trans(ds)
    ds = calc_bulk_richardson(ds)
    ds = calc_mse(ds)
    ds = calc_vpd2m(ds)
    ds = calc_evaporative_fraction(ds)
    return ds
