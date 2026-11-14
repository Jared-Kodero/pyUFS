from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from plugins.sst_perturbation.config import DEFAULTS


def validate(*, config: dict) -> None:
    """Check the settings of the sst_perturbation section."""
    delta, months, lat, lon, taper = (config[key] for key in DEFAULTS)
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise TypeError(f"`delta` must be a number of kelvin. Got {delta!r}")
    if not months or any(m not in range(1, 13) for m in months):
        raise ValueError(f"`months` must be calendar months 1 to 12. Got {months!r}")
    if len(lat) != 2 or not -90 <= lat[0] < lat[1] <= 90:
        raise ValueError(f"`lat` must be [south, north] within -90 to 90. Got {lat!r}")
    if lon is not None and (len(lon) != 2 or any(not 0 <= x <= 360 for x in lon)):
        raise ValueError(f"`lon` must be [west, east] within 0 to 360. Got {lon!r}")
    if taper < 0:
        raise ValueError(f"`taper` must be >= 0. Got {taper!r}")


def apply(
    *,
    config: dict,
    state: Any,
    restart: bool,
    patch_nml: Callable,
    read_namelist: Callable,
    open_grib: Callable,
    save_grib: Callable,
    ll_regrid: Callable,
) -> None:
    """Write a perturbed copy of the SST climatology the namelist points at and point
    namsfc at the copy. The file and the namelist are kept by restart segments."""
    if restart:
        return
    from plugins.sst_perturbation.perturb import write_perturbed_sst

    namsfc = read_namelist(Path(state.case_dir) / "input.nml")["namsfc"]
    path = write_perturbed_sst(
        config,
        state=state,
        namsfc=namsfc,
        open_grib=open_grib,
        save_grib=save_grib,
        ll_regrid=ll_regrid,
    )
    patch_nml({"namsfc": {"fntsfc": path}})
