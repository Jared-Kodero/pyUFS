"""Surface analysis update of the cold-start ICs (run_global_cycle).

Follows ush/global_cycle_driver.sh: after chgres_cube, global_cycle updates
INPUT/sfc_data* from the climatologies and the optional SST, sea-ice and
snow analyses. The six global tiles run as one 6-rank job; each nest has
its own tile size and runs as a 1-rank job. The snow and ice analyses can
be built first with emcsfc_snow2mdl and emcsfc_ice_blend; they are kept in
IC/surface_analysis.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from emcsfc_ice_blend import run_ice_blend
from emcsfc_snow import run_emcsfc_snow
from fv3_runtime import log
from fv3_state import state
from global_cycle import namsfc_entries, run_global_cycle

GLOBAL_GTYPES = ("uniform", "stretch", "nest")


def surface_domains() -> list[tuple[str, list[tuple[Path, Path, Path]]]]:
    """(label, [(sfc, grid, orog), ...]) for the global tiles and each nest."""
    inp, grid, res = Path(state.input), Path(state.grid), state.c_res
    domains = [
        (
            "global",
            [
                (
                    inp / f"sfc_data.tile{t}.nc",
                    grid / f"C{res}_grid.tile{t}.nc",
                    inp / f"oro_data.tile{t}.nc",
                )
                for t in range(1, 7)
            ],
        )
    ]
    for k in range(1, int(state.n_nests or 0) + 1):
        tile, idx = 6 + k, k + 1
        domains.append(
            (
                f"nest{idx:02d}",
                [
                    (
                        inp / f"sfc_data.nest{idx:02d}.tile{tile}.nc",
                        grid / f"C{res}_grid.tile{tile}.nc",
                        inp / f"oro_data.nest{idx:02d}.tile{tile}.nc",
                    )
                ],
            )
        )
    return domains


def _path(value: str | None) -> str | None:
    return os.path.expandvars(str(value)) if value else None


def run_surface_cycle() -> None:
    if not state.run_global_cycle:
        return

    sst = _path(state.global_cycle_sst_file)
    ice = _path(state.global_cycle_ice_file)
    snow = _path(state.global_cycle_snow_file)
    keep = Path(state.ic_data) / "surface_analysis"
    if state.run_emcsfc_snow:
        snow = str(run_emcsfc_snow(keep / "emcsfc_snow"))
    if state.run_emcsfc_ice_blend:
        ice = str(run_ice_blend(keep / "emcsfc_ice_blend"))
    for name, value in (("SST", sst), ("ice", ice), ("snow", snow)):
        if value and not Path(value).exists():
            raise FileNotFoundError(f"global_cycle {name} analysis not found: {value}")
        log.info(f"global_cycle {name} analysis: {value or 'none (climatology)'}")

    namsfc = namsfc_entries(sst, ice, snow, state.global_cycle_vars)
    work = Path(state.tmp) / "global_cycle"
    for label, files in surface_domains():
        run_global_cycle(label, files, work / label, namsfc, state.init_datetime)
    shutil.rmtree(work, ignore_errors=True)
