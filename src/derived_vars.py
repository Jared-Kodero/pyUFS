import logging

import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from metpy.units import units

from fv3_diag_files import variables

log = logging.getLogger("USER_DERIVED_VARS")

# variables maps the native diag_table field name to the output name the user
# gave it, for example variables["t_plev"] == "t". It is filled when the
# diag_table is loaded (fv3_diag_files.load_variables), so the calc_* functions
# below read their inputs as ds[variables["t_plev"]] and work with any output
# names. A field that is not in the table maps to its own native name, which
# is absent from the dataset, so the calculation is skipped with a warning.
# Derived variables are written under fixed names.

g = 9.80665
R_earth_m = 6371000.0  # Earth radius [m]


def calc_moisture_trans(ds: xr.Dataset) -> xr.Dataset:
    required_vars = (
        variables["q_plev"],
        variables["u_plev"],
        variables["v_plev"],
        variables["ps"],
    )
    if not all(var in ds.data_vars for var in required_vars) or "plev" not in ds.dims:
        log.warning(
            f"Missing variables {required_vars} or plev for moisture transport calculation. Skipping!"
        )
        return ds

    # Ensure pressure is sorted descending (top to bottom)
    ds = ds.sortby("plev", ascending=False)

    # Convert plev from hPa to Pa (assuming ps is also in Pa)
    plev_Pa = ds["plev"] * 100.0

    # Extract top and bottom boundaries for layers using shift
    p_top = plev_Pa.isel(plev=slice(1, None))
    p_bottom = plev_Pa.shift(plev=1).isel(plev=slice(1, None))

    q_top = ds[variables["q_plev"]].isel(plev=slice(1, None))
    u_top = ds[variables["u_plev"]].isel(plev=slice(1, None))
    v_top = ds[variables["v_plev"]].isel(plev=slice(1, None))

    q_bottom = ds[variables["q_plev"]].shift(plev=1).isel(plev=slice(1, None))
    u_bottom = ds[variables["u_plev"]].shift(plev=1).isel(plev=slice(1, None))
    v_bottom = ds[variables["v_plev"]].shift(plev=1).isel(plev=slice(1, None))

    # Accurate pressure thickness handling surface clipping (partial layers)
    dp = (
        xr.where(p_bottom <= ds[variables["ps"]], p_bottom, ds[variables["ps"]]) - p_top
    ).clip(min=0)

    # Layer-mean moisture flux (trapezoidal integration)
    qu = 0.5 * (q_bottom * u_bottom + q_top * u_top)
    qv = 0.5 * (q_bottom * v_bottom + q_top * v_top)

    ivtu = (qu * dp).sum(dim="plev") / g
    ivtv = (qv * dp).sum(dim="plev") / g

    ivtu = ivtu.astype(ds[variables["q_plev"]].dtype)
    ivtv = ivtv.astype(ds[variables["q_plev"]].dtype)

    # Magnitude of integrated vapor transport
    ivt = (ivtu**2 + ivtv**2) ** 0.5
    ivt = ivt.astype(ds[variables["q_plev"]].dtype)

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

    # Moisture Flux Convergence (VIMFC) calculation
    lat_rad = np.deg2rad(ds["lat"])
    lon_rad = np.deg2rad(ds["lon"])
    coslat = np.cos(lat_rad).clip(min=1e-6)

    ivtu_rad = ivtu.assign_coords(lat=lat_rad, lon=lon_rad)
    ivtv_rad = ivtv.assign_coords(lat=lat_rad, lon=lon_rad)

    d_ivtu_dlon = ivtu_rad.differentiate("lon")
    coslat_rad = coslat.assign_coords(lat=lat_rad)
    v_coslat = ivtv_rad * coslat_rad
    d_vcoslat_dlat = v_coslat.differentiate("lat")

    vimfc = -(d_ivtu_dlon + d_vcoslat_dlat) / (R_earth_m * coslat_rad)

    vimfc = vimfc.assign_coords(lat=ds["lat"], lon=ds["lon"])
    vimfc = vimfc.astype(ds[variables["q_plev"]].dtype)

    vimfc.name = "vimfc"
    vimfc.attrs["units"] = "kg m-2 s-1"
    vimfc.attrs["short_name"] = "vimfc"
    vimfc.attrs["long_name"] = "Vertically Integrated Moisture Flux Convergence"

    ds["vimfc"] = vimfc

    return ds


def calc_bulk_richardson(ds: xr.Dataset) -> xr.Dataset:
    required_vars = (variables["cape"], variables["shear06"])
    if not all(var in ds.data_vars for var in required_vars):
        log.warning(
            f"Missing variables {required_vars} for bulk Richardson number calculation. Skipping!"
        )
        return ds

    ds["brn"] = ds[variables["cape"]] / (0.5 * ds[variables["shear06"]] ** 2)
    ds["brn"] = ds["brn"].clip(0, 100)
    dtype = np.result_type(
        ds[variables["cape"]].dtype, ds[variables["shear06"]].dtype, np.float32
    )
    ds["brn"] = ds["brn"].astype(dtype)
    ds["brn"].attrs["units"] = "dimensionless"
    ds["brn"].attrs["long_name"] = "Bulk Richardson number"

    return ds


def calc_mse(ds: xr.Dataset) -> xr.Dataset:
    required_vars = (variables["t_plev"], variables["q_plev"], variables["h_plev"])
    if not all(var in ds.data_vars for var in required_vars):
        log.warning(
            f"Missing variables {required_vars} for moist static energy calculation. Skipping!"
        )
        return ds

    t = ds[variables["t_plev"]] * units.degK
    q = ds[variables["q_plev"]] * 1000 * units("g/kg")
    z = ds[variables["h_plev"]] * units.meter

    mse = mpcalc.moist_static_energy(z, t, q)
    mse = mse.metpy.dequantify()
    mse = mse.astype(ds[variables["t_plev"]].dtype)

    mse.name = "mse"
    mse.attrs["units"] = "kJ/kg"
    mse.attrs["short_name"] = "mse"
    mse.attrs["long_name"] = "Moist Static Energy"

    ds["mse"] = mse
    return ds


def calc_evaporative_fraction(ds: xr.Dataset) -> xr.Dataset:
    required_vars = (variables["shtfl_ave"], variables["lhtfl_ave"])
    if not all(var in ds.data_vars for var in required_vars):
        log.warning(
            f"Missing variables {required_vars} for evaporative fraction calculation. Skipping!"
        )
        return ds
    epsilon = 1e-6
    ds["evf"] = ds[variables["lhtfl_ave"]] / (
        (ds[variables["lhtfl_ave"]] + ds[variables["shtfl_ave"]]) + epsilon
    )
    ds["evf"] = ds["evf"].clip(0, 1)
    ds["evf"] = ds["evf"].astype(ds[variables["shtfl_ave"]].dtype)
    ds["evf"].attrs["units"] = "dimensionless"
    ds["evf"].attrs["long_name"] = "Evaporative Fraction"
    return ds


def calc_vpd2m(ds: xr.Dataset) -> xr.Dataset:
    """Calculate 2 meter vapor pressure deficit using MetPy."""
    required_vars = (variables["t2m"], variables["dpt2m"])
    if not all(var in ds.data_vars for var in required_vars):
        return ds

    t2m = ds[variables["t2m"]] * units.kelvin
    td2m = ds[variables["dpt2m"]] * units.kelvin

    es = mpcalc.saturation_vapor_pressure(t2m).metpy.convert_units("hPa")
    e = mpcalc.saturation_vapor_pressure(td2m).metpy.convert_units("hPa")

    es = es.metpy.dequantify()
    e = e.metpy.dequantify()

    vpd2m = es - e
    vpd2m = vpd2m.astype(ds[variables["t2m"]].dtype)
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
