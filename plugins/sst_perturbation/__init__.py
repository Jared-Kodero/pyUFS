"""An SST anomaly added to the SST climatology that the namelist points at."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fv3_utils import open_nml

# Keys of the `sst_perturbation` section of the plugins file.
#   delta   K added to open-water SST (required); negative cools.
#   months  calendar months of the climatology that change.
#   lat     [south, north] degrees north.
#   lon     [west, east] degrees east (0 to 360, west > east wraps the date line); null: all.
#   taper   degrees over which the change rises from 0 at the box edge to delta.
DEFAULTS = {
    "delta": None,
    "months": [6, 7, 8],
    "lat": [50.0, 80.0],
    "lon": None,
    "taper": 5.0,
}
HASHED = True


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


def cold_start(*, config: dict, state: Any) -> dict[str, dict]:
    """Write a perturbed copy of the SST climatology the namelist points at and point
    namsfc at the copy. The file and the namelist are kept by restart segments."""
    from .perturb import write_perturbed_sst

    namsfc = open_nml(Path(state.case_dir) / "input.nml")["namsfc"]
    return {
        "namsfc": {"fntsfc": write_perturbed_sst(config, state=state, namsfc=namsfc)}
    }
