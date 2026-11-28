"""Example plugin: 2 m temperature in degrees Celsius, added to every output stream.

Enable it in the plugins file:

    plugin_dirs: ["${PYUFS_DIR}/config/examples/plugins"]
    t2m_celsius: true

regrid is called once per output stream, so it skips a stream that lacks the input.
"""

import xarray as xr


def regrid(
    ds: xr.Dataset, *, config: dict, state, variables: dict[str, str]
) -> xr.Dataset | None:
    if variables["t2m"] not in ds:
        return None
    ds["t2m_c"] = ds[variables["t2m"]] - 273.15
    ds["t2m_c"].attrs.update(units="degC", long_name="2 meter temperature")
    return ds
