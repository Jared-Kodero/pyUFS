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

# Generate lateral boundaries from one continuous driving-model forecast. The
# cycle remains fixed at the regional cold-start cycle while the source forecast
# lead advances in 3 h increments: f000, f003, f006, ... . The FV3 runtime
# namelist must use the same bc_update_interval.
BC_INTERVAL_HOURS = 3

# Operational cycle and forecast-lead availability.
#
# GFS:
#   - cycles: 00/06/12/18 UTC
#   - absolute forecast horizon: f384
#   - continuous 3-hourly output: through f240
#   - output after f240: 12-hourly through f384
#
# HRRR:
#   - cycles: every UTC hour
#   - standard cycles: through f018
#   - extended 00/06/12/18 UTC cycles: through f048
#
# This module requires one uninterrupted 3-hourly boundary sequence. Therefore,
# its applicable GFS limit is f240, not the absolute f384 forecast horizon.
SYNOPTIC_CYCLE_HOURS = frozenset({0, 6, 12, 18})

MODEL_CYCLE_HOURS: dict[str, frozenset[int]] = {
    "GFS": SYNOPTIC_CYCLE_HOURS,
    "HRRR": frozenset(range(24)),
}

HRRR_EXTENDED_CYCLE_HOURS = SYNOPTIC_CYCLE_HOURS

GFS_MAX_3H_FORECAST_HOUR = 240
HRRR_STANDARD_MAX_FORECAST_HOUR = 18
HRRR_EXTENDED_MAX_FORECAST_HOUR = 48

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
    """Return regional elapsed hours that require a boundary file.

    Boundaries are supplied every ``BC_INTERVAL_HOURS`` from regional hour zero
    through ``state.total_run_hours``. If the integration length is not an exact
    multiple of the interval, one additional boundary time is appended so that
    the final interpolation window is bracketed.
    """
    total = int(state.total_run_hours)
    hours = list(range(0, total + 1, BC_INTERVAL_HOURS))

    if hours[-1] < total:
        hours.append(hours[-1] + BC_INTERVAL_HOURS)

    return hours


def bc_external_model() -> str:
    """Return the atmosphere model used for initial and boundary forcing.

    The interior and lateral boundaries are drawn from the same model so that
    fields agree across the blending halo. Any atmosphere source other than
    HRRR falls back to GFS.
    """
    source = str((state.regional_ic_source or {}).get("atm", "GFS")).upper()

    return "HRRR" if source == "HRRR" else "GFS"


def max_3h_forecast_hour(model: str, cycle_hour: int) -> int:
    """Return the maximum continuous 3-hourly lead for a model cycle."""
    if model == "GFS":
        return GFS_MAX_3H_FORECAST_HOUR

    if cycle_hour in HRRR_EXTENDED_CYCLE_HOURS:
        return HRRR_EXTENDED_MAX_FORECAST_HOUR

    return HRRR_STANDARD_MAX_FORECAST_HOUR


def _cycle_list(hours: frozenset[int]) -> str:
    """Format UTC cycle hours for an actionable user message."""
    return "/".join(f"{hour:02d}" for hour in sorted(hours))


def _insufficient_horizon_guidance(
    model: str,
    cycle_hour: int,
    required_hour: int,
) -> str:
    """Return an appropriate recovery action for an insufficient horizon."""
    synoptic_cycles = _cycle_list(SYNOPTIC_CYCLE_HOURS)

    if model == "HRRR":
        if (
            cycle_hour not in HRRR_EXTENDED_CYCLE_HOURS
            and required_hour <= HRRR_EXTENDED_MAX_FORECAST_HOUR
        ):
            return (
                f"Change the HRRR cycle to {synoptic_cycles} UTC to use an "
                "extended forecast through f048, or switch the atmosphere "
                f"source to GFS using a {synoptic_cycles} UTC cycle."
            )

        if required_hour <= GFS_MAX_3H_FORECAST_HOUR:
            return (
                "No HRRR cycle provides enough forecast lead. Switch the "
                f"atmosphere source to GFS and use a {synoptic_cycles} UTC "
                "cycle. GFS provides a continuous 3-hourly sequence through "
                f"f{GFS_MAX_3H_FORECAST_HOUR:03d}."
            )

        return (
            "Neither HRRR nor the configured GFS product provides a "
            f"continuous 3-hourly single-cycle sequence through "
            f"f{required_hour:03d}. Reduce the integration length, use a "
            "larger boundary interval with appropriate temporal "
            "interpolation, or implement multi-cycle boundary forcing."
        )

    return (
        "GFS reaches f384, but its continuous 3-hourly output ends at "
        f"f{GFS_MAX_3H_FORECAST_HOUR:03d}. Changing the GFS cycle will not "
        "increase this limit. Reduce the integration length, use a larger "
        "boundary interval with appropriate temporal interpolation, or "
        "implement multi-cycle boundary forcing."
    )


def validate_forecast_horizon(
    model: str,
    cycle: pd.Timestamp,
    hours: list[int],
) -> None:
    """Validate cycle existence and complete single-cycle boundary coverage.

    A warning is logged and ``ValueError`` is raised before source data are
    downloaded when the requested boundary sequence cannot be obtained.
    """
    model = model.upper()

    if model not in MODEL_CYCLE_HOURS:
        supported = ", ".join(sorted(MODEL_CYCLE_HOURS))
        raise ValueError(
            f"Unsupported boundary-driving model {model!r}; supported models are {supported}."
        )

    cycle_hour = int(cycle.hour)
    valid_cycle_hours = MODEL_CYCLE_HOURS[model]

    if cycle_hour not in valid_cycle_hours:
        valid_cycles = _cycle_list(valid_cycle_hours)
        message = (
            f"{model} has no {cycle_hour:02d} UTC forecast cycle. "
            f"Change state.init_datetime to a {valid_cycles} UTC cycle."
        )
        log.warning(message)
        raise ValueError(message)

    required_hour = hours[-1]
    available_hour = max_3h_forecast_hour(model, cycle_hour)

    if required_hour <= available_hour:
        return

    guidance = _insufficient_horizon_guidance(
        model=model,
        cycle_hour=cycle_hour,
        required_hour=required_hour,
    )

    message = (
        f"Insufficient {model} boundary data for the "
        f"{cycle.strftime('%Y-%m-%d %H UTC')} cycle: a complete "
        f"{BC_INTERVAL_HOURS}-hourly sequence is available only through "
        f"f{available_hour:03d}, but the regional integration requires "
        f"f{required_hour:03d}. {guidance}"
    )

    log.warning(message)
    raise ValueError(message)


def generate_boundary_files(base_config, source_mosaic, run_spec) -> None:
    """Generate every lateral boundary file from one driving-model forecast.

    Parameters
    ----------
    base_config
        The regional-domain ``ChgresCubeConfig`` used for the initial-condition
        pass. Its target-grid, vertical-coordinate, halo and mapping settings
        are inherited; only the input data, valid time and conversion flags
        change.
    source_mosaic
        Per-domain mosaic staged onto the canonical target-grid path.
    run_spec
        Callable ``(domain_label, source_mosaic, config) -> Path`` that writes
        fort.41, launches chgres_cube and returns the working directory holding
        the boundary output. Passed in by chgres_cube so this module never
        imports it, avoiding a circular import.

    Source files remain on the cold-start driving cycle and advance by forecast
    lead: f000, f003, f006, ... . Output files are written to ``state.bc_data``
    as ``gfs_bndy.tile7.HHH.nc`` and linked into ``state.input`` separately by
    ``link_bc_to_input``.
    """
    model = bc_external_model()

    bc_dir = Path(state.bc_data)
    bc_dir.mkdir(parents=True, exist_ok=True)

    for stale in bc_dir.glob("gfs_bndy.tile7.*.nc"):
        stale.unlink()

    cycle = pd.Timestamp(state.init_datetime)
    hours = bc_forecast_hours()

    # Validate the complete sequence before downloading or processing any data.
    validate_forecast_horizon(model, cycle, hours)

    for fh in hours:
        valid = cycle + pd.Timedelta(hours=fh)

        data_dir, data_file = get_ic_data(
            model,
            datetime=cycle,
            forecast_hour=fh,
        )

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

        work_dir = Path(
            run_spec(
                f"regional_bc_{fh:03d}",
                source_mosaic,
                config,
            )
        )

        produced = work_dir / CHGRES_BNDY_FILE

        if not produced.exists():
            fallback = sorted(work_dir.glob("*bndy*.nc"))

            if not fallback:
                raise FileNotFoundError(
                    f"chgres_cube produced no boundary file for forecast hour {fh:03d} in {work_dir}"
                )

            produced = fallback[0]

        rename(
            produced,
            bc_dir / f"gfs_bndy.tile7.{fh:03d}.nc",
        )

    log.info(
        "Generated %d regional boundary files from the %s %s cycle at %d h spacing",
        len(hours),
        model,
        cycle.strftime("%Y-%m-%d %H UTC"),
        BC_INTERVAL_HOURS,
    )


def link_bc_to_input() -> None:
    """Symlink every staged boundary file into ``state.input``.

    Boundary files persist in ``state.bc_data`` for the whole run, but each
    warm-start segment rebuilds ``state.input`` by promoting the previous
    segment's RESTART directory, which drops the links. This is therefore
    called after initial staging and again on every restart so the full
    boundary sequence remains visible under
    ``INPUT/gfs_bndy.tile7.HHH.nc``.
    """
    if state.gtype not in ("regional_gfdl", "regional_esg"):
        return

    input_dir = Path(state.input)
    input_dir.mkdir(parents=True, exist_ok=True)

    for stale in input_dir.glob("gfs_bndy.tile7.*.nc"):
        stale.unlink(missing_ok=True)

    for src in sorted(Path(state.bc_data).glob("gfs_bndy.tile7.*.nc")):
        link = input_dir / src.name
        link.unlink(missing_ok=True)
        link.symlink_to(
            os.path.relpath(
                src,
                start=input_dir,
            )
        )
