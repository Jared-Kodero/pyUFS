"""Example derived_vars_module: 2 m temperature in degrees Celsius.

Set `derived_vars_module: ${UFS_UTILS_DIR}/config/examples/derived_vars_example.py`
in run_config.yaml. Functions are called once per output stream, so skip a stream
that lacks the inputs.
"""

import xarray as xr


def calc_t2m_celsius(ds: xr.Dataset, *, variables: dict[str, str], **_) -> xr.Dataset:
    if variables["t2m"] not in ds:
        return ds
    ds["t2m_c"] = ds[variables["t2m"]] - 273.15
    ds["t2m_c"].attrs.update(units="degC", long_name="2 meter temperature")
    return ds


derive = [calc_t2m_celsius]
