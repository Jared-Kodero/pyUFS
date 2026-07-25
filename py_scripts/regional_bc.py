from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
from fv3_ic_data import get_ic_data
from fv3_state import state
from fv3_utils import rename

log = logging.getLogger("PREPROCESS")

# Lateral boundaries are refreshed on the driving-model analysis cadence. GFS and
# HRRR both publish analyses on the synoptic cycle, and run_config.yaml already
# constrains init_datetime to 00, 06, 12 or 18 UTC (fv3_utils.parse_datetime).
# A single 6 h interval therefore places every boundary time on an available
# cycle, so the required lead hours reduce to the [0, 6, 12, 18] cycle set that
# repeats across the integration.
BC_INTERVAL_HOURS = 6

# chgres_cube regional controls (UFS_UTILS program_setup.F90):
#   regional = 1  initial-condition pass; also writes the hour-0 boundary file.
#   regional = 2  boundary-only pass; one file per boundary time.
# halo_bndy is the pure lateral-boundary halo (the model reads halo + 1 rows),
# and halo_blend is the width over which model and boundary tendencies blend.
REGIONAL_IC = 1
REGIONAL_BC = 2
HALO_BLEND = 10

# chgres_cube writes one regional boundary file per pass under this fixed name;
# the model reads INPUT/gfs_bndy.tile7.HHH.nc, so each pass is renamed to its
# forecast hour (fv_regional_bc.F90).
CHGRES_BNDY_FILE = "gfs.bndy.nc"


def bc_forecast_hours() -> list[int]:
    """Forecast hours, from the cold-start cycle, that require a boundary file.

    Boundaries are supplied every ``BC_INTERVAL_HOURS`` from hour 0 through the
    full integration length ``state.total_run_hours``. When ``total_run_hours``
    is not a multiple of the interval, one further time is appended so the final
    boundary-update window is bracketed on both sides.
    """
    total = int(state.total_run_hours)
    hours = list(range(0, total + 1, BC_INTERVAL_HOURS))
    if hours[-1] < total:
        hours.append(hours[-1] + BC_INTERVAL_HOURS)
    return hours


def bc_external_model() -> str:
    """Driving model for the boundary, taken from the regional atmosphere source.

    The interior and boundary are drawn from the same model so the fields agree
    across the blending halo. chgres_cube records the atmosphere source per
    domain in ``state.regional_ic_source``; anything other than HRRR falls back
    to GFS, which has global coverage at every cycle.
    """
    source = (state.regional_ic_source or {}).get("atm", "GFS")
    return "HRRR" if source == "HRRR" else "GFS"


def generate_boundary_files(base_config, source_mosaic, run_spec) -> None:
    """Generate every lateral boundary file the regional run needs before start.

    Parameters
    ----------
    base_config
        The regional-domain ``ChgresCubeConfig`` used for the initial-condition
        pass. Its target-grid, vertical-coordinate, halo and mapping settings are
        inherited; only the input data, cycle time and conversion flags change.
    source_mosaic
        Per-domain mosaic staged onto the canonical target-grid path.
    run_spec
        Callable ``(domain_label, source_mosaic, config) -> Path`` that writes
        fort.41, launches chgres_cube and returns the working directory holding
        the boundary output. Passed in by chgres_cube so this module never
        imports it, avoiding a circular import.

    All files are written to ``state.bc_data`` as ``gfs_bndy.tile7.HHH.nc`` and
    are linked into ``state.input`` separately by ``link_bc_to_input``.
    """
    model = bc_external_model()
    bc_dir = Path(state.bc_data)
    bc_dir.mkdir(parents=True, exist_ok=True)

    hours = bc_forecast_hours()
    for fh in hours:
        valid = state.init_datetime + pd.Timedelta(hours=fh)
        data_dir, data_file = get_ic_data(model, datetime=valid, forecast_hour=0)

        config = replace(
            base_config,
            external_model=model,
            regional=REGIONAL_BC,
            convert_atm=True,
            convert_sfc=False,
            convert_nst=False,
            data_dir_input_grid=data_dir,
            grib2_file_input_grid=data_file,
            cycle_year=valid.year,
            cycle_mon=valid.month,
            cycle_day=valid.day,
            cycle_hour=valid.hour,
        )

        work_dir = Path(run_spec(f"regional_bc_{fh:03d}", source_mosaic, config))

        produced = work_dir / CHGRES_BNDY_FILE
        if not produced.exists():
            fallback = sorted(work_dir.glob("*bndy*.nc"))
            if not fallback:
                raise FileNotFoundError(
                    f"chgres_cube produced no boundary file for hour {fh:03d} in {work_dir}"
                )
            produced = fallback[0]

        rename(produced, bc_dir / f"gfs_bndy.tile7.{fh:03d}.nc")

    log.info(
        "Generated %d regional boundary files from %s at %d h spacing",
        len(hours),
        model,
        BC_INTERVAL_HOURS,
    )


def link_bc_to_input() -> None:
    """Symlink every staged boundary file into ``state.input`` for the model.

    Boundary files persist in ``state.bc_data`` for the whole run, but each
    warm-start segment rebuilds ``state.input`` by promoting the previous
    segment's RESTART directory, which drops the links. This is therefore called
    after initial staging and again on every restart so the full boundary
    sequence is always visible under INPUT/gfs_bndy.tile7.HHH.nc.
    """
    if state.gtype not in ("regional_gfdl", "regional_esg"):
        return

    input_dir = Path(state.input)
    input_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(Path(state.bc_data).glob("gfs_bndy.tile7.*.nc")):
        link = input_dir / src.name
        link.unlink(missing_ok=True)
        link.symlink_to(os.path.relpath(src, start=input_dir))
