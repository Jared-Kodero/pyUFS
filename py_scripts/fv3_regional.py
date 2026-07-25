from __future__ import annotations

import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

REGIONAL_GRID_TYPES = frozenset({"regional_gfdl", "regional_esg"})

# Use the native cycling cadence as the boundary-update cadence.  The external
# forecast is held to one cycle and successive forecast hours are used so the
# lateral forcing remains temporally consistent.
BOUNDARY_INTERVAL_HOURS = {"GFS": 6, "HRRR": 1}
GFS_CYCLE_HOURS = frozenset({0, 6, 12, 18})
HRRR_LONG_CYCLE_HOURS = frozenset({0, 6, 12, 18})
GFS_MAX_FORECAST_HOUR = 384
HRRR_STANDARD_MAX_FORECAST_HOUR = 18
HRRR_LONG_MAX_FORECAST_HOUR = 48


@dataclass(frozen=True)
class BoundaryTime:
    """One SHiELD lateral-boundary time and its external-model forecast hour."""

    model_hour: int
    source_forecast_hour: int
    valid_time: datetime


@dataclass(frozen=True)
class RegionalGridFiles:
    """Regional halo-4 target files consumed by chgres_cube."""

    resolution: int
    boundary_halo: int
    mosaic: Path
    grid: Path
    orography: Path


def is_regional_grid(gtype: str) -> bool:
    return gtype in REGIONAL_GRID_TYPES


def boundary_interval_hours(external_model: str) -> int:
    model = external_model.upper()
    try:
        return BOUNDARY_INTERVAL_HOURS[model]
    except KeyError as exc:
        supported = ", ".join(sorted(BOUNDARY_INTERVAL_HOURS))
        raise ValueError(
            f"Unsupported regional boundary source {external_model!r}; expected one of {supported}"
        ) from exc


def source_cycle_time(init_datetime: datetime, forecast_hour: int) -> datetime:
    """Return the external-model cycle whose lead initializes the model."""

    if not isinstance(forecast_hour, int) or isinstance(forecast_hour, bool):
        raise TypeError("forecast_hour must be an integer")
    if forecast_hour < 0:
        raise ValueError("forecast_hour must be non-negative")
    return init_datetime - timedelta(hours=forecast_hour)


def _validate_source_cycle(external_model: str, source_cycle: datetime) -> None:
    if external_model == "GFS" and source_cycle.hour not in GFS_CYCLE_HOURS:
        valid = ", ".join(f"{hour:02d}Z" for hour in sorted(GFS_CYCLE_HOURS))
        raise ValueError(
            f"GFS source cycle {source_cycle:%Y-%m-%d %HZ} is unavailable; "
            f"forecast_hour must select one of {valid}"
        )


def _latest_gfs_cycle(valid_time: datetime) -> datetime:
    cycle_hour = max(hour for hour in GFS_CYCLE_HOURS if hour <= valid_time.hour)
    return valid_time.replace(
        hour=cycle_hour,
        minute=0,
        second=0,
        microsecond=0,
    )


def plan_boundary_times(
    external_model: str,
    init_datetime: datetime,
    forecast_hour: int,
    total_run_hours: int,
) -> tuple[datetime, list[BoundaryTime]]:
    """Plan every required regional boundary from hour zero through run end."""

    model = external_model.upper()
    interval = boundary_interval_hours(model)

    if not isinstance(total_run_hours, int) or isinstance(total_run_hours, bool):
        raise TypeError("total_run_hours must be an integer")
    if total_run_hours < 0:
        raise ValueError("total_run_hours must be non-negative")
    if total_run_hours % interval:
        raise ValueError(
            f"total_run_hours={total_run_hours} is not divisible by the "
            f"{model} boundary interval of {interval} hour(s)"
        )

    source_cycle = source_cycle_time(init_datetime, forecast_hour)
    _validate_source_cycle(model, source_cycle)

    times = [
        BoundaryTime(
            model_hour=model_hour,
            source_forecast_hour=forecast_hour + model_hour,
            valid_time=init_datetime + timedelta(hours=model_hour),
        )
        for model_hour in range(0, total_run_hours + 1, interval)
    ]
    _validate_source_forecast_hours(model, source_cycle, times)
    return source_cycle, times


def _validate_source_forecast_hours(
    external_model: str,
    source_cycle: datetime,
    times: list[BoundaryTime],
) -> None:
    if not times:
        return

    forecast_hours = [item.source_forecast_hour for item in times]
    maximum = max(forecast_hours)

    if external_model == "GFS":
        if maximum > GFS_MAX_FORECAST_HOUR:
            raise ValueError(
                f"GFS boundary forcing requires f{maximum:03d}, beyond the "
                f"supported f{GFS_MAX_FORECAST_HOUR:03d} forecast"
            )

        unavailable = [hour for hour in forecast_hours if hour > 120 and hour % 3 != 0]
        if unavailable:
            hours = ", ".join(f"f{hour:03d}" for hour in unavailable)
            raise ValueError(
                "GFS 0.25-degree files are three-hourly after f120; "
                f"unavailable requested hours: {hours}"
            )
        return

    maximum_available = (
        HRRR_LONG_MAX_FORECAST_HOUR
        if source_cycle.hour in HRRR_LONG_CYCLE_HOURS
        else HRRR_STANDARD_MAX_FORECAST_HOUR
    )
    if maximum > maximum_available:
        raise ValueError(
            f"HRRR cycle {source_cycle:%Y-%m-%d %HZ} provides forecasts through "
            f"f{maximum_available:02d}, but regional forcing requires f{maximum:02d}"
        )


def boundary_filename(model_hour: int) -> str:
    if not isinstance(model_hour, int) or isinstance(model_hour, bool):
        raise TypeError("model_hour must be an integer")
    if model_hour < 0:
        raise ValueError("model_hour must be non-negative")
    return f"gfs_bndy.tile7.{model_hour:03d}.nc"


def _resolution_from_name(path: Path) -> int:
    match = re.match(r"C(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot determine regional C-resolution from {path}")
    return int(match.group(1))


def validate_regional_grid_files() -> RegionalGridFiles:
    """Resolve and validate the halo-4 target files before chgres_cube runs."""

    from fv3_state import state

    if not is_regional_grid(state.gtype):
        raise ValueError(f"Regional grid files requested for gtype={state.gtype!r}")

    boundary_halo = int(state.halo) + 1
    input_dir = Path(state.tmp) / "input"
    pattern = f"C*_grid.tile7.halo{boundary_halo}.nc"
    grids = sorted(input_dir.glob(pattern))
    if len(grids) != 1:
        raise FileNotFoundError(
            f"Expected exactly one regional halo-{boundary_halo} grid matching "
            f"{input_dir / pattern}; found {len(grids)}"
        )

    resolution = _resolution_from_name(grids[0])
    expected = RegionalGridFiles(
        resolution=resolution,
        boundary_halo=boundary_halo,
        mosaic=input_dir / f"C{resolution}_mosaic.halo{boundary_halo}.nc",
        grid=grids[0],
        orography=input_dir / f"C{resolution}_oro_data.tile7.halo{boundary_halo}.nc",
    )
    missing = [
        path
        for path in (expected.mosaic, expected.grid, expected.orography)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing regional chgres_cube target files:\n{names}")

    if int(state.c_res) != resolution:
        raise ValueError(
            f"Regional grid resolution mismatch: state.c_res={state.c_res}, "
            f"generated files use C{resolution}"
        )
    return expected


def prepare_regional_grid_files(
    *,
    c_res: int,
    idim: int,
    jdim: int,
    halo: int,
    exec_dir: Path,
    input_dir: Path,
) -> None:
    """Validate shaved regional grids and build a mosaic for each halo width."""

    import xarray as xr
    from fv3_state import state
    from fv3_utils import run_cmd

    if min(int(c_res), int(idim), int(jdim)) <= 0:
        raise ValueError("Regional resolution and dimensions must be positive")
    if int(halo) != 3:
        raise ValueError(f"Regional SHiELD grids require halo=3; received halo={halo}")

    input_dir = Path(input_dir)
    halo_widths = tuple(dict.fromkeys((0, int(halo), int(halo) + 1)))
    missing: list[Path] = []

    for width in halo_widths:
        grid = input_dir / f"C{c_res}_grid.tile7.halo{width}.nc"
        orography = input_dir / f"C{c_res}_oro_data.tile7.halo{width}.nc"
        width_missing = [
            path
            for path in (grid, orography)
            if not path.is_file() or path.stat().st_size == 0
        ]
        missing.extend(width_missing)
        if width_missing:
            continue

        expected_nx = 2 * (int(idim) + 2 * width)
        expected_ny = 2 * (int(jdim) + 2 * width)
        with xr.open_dataset(grid) as dataset:
            nx = int(dataset.sizes.get("nx", 0))
            ny = int(dataset.sizes.get("ny", 0))
            equivalent = dataset.attrs.get("RES_equiv")

        if (nx, ny) != (expected_nx, expected_ny):
            raise ValueError(
                f"{grid}: supergrid dimensions {(nx, ny)} do not match "
                f"expected {(expected_nx, expected_ny)} for halo {width}"
            )
        if equivalent is not None:
            match = re.search(r"\d+", str(equivalent))
            if not match or int(match.group()) != int(c_res):
                raise ValueError(
                    f"{grid}: RES_equiv={equivalent} does not match C{c_res}"
                )

        mosaic_stem = f"C{c_res}_mosaic.halo{width}"
        mosaic = input_dir / f"{mosaic_stem}.nc"
        mosaic.unlink(missing_ok=True)
        cmd = [
            str(Path(exec_dir) / "make_solo_mosaic"),
            "--num_tiles",
            "1",
            "--dir",
            str(input_dir),
            "--mosaic",
            mosaic_stem,
            "--tile_file",
            grid.name,
        ]
        log_file = Path(state.logs) / f"make_regional_mosaic_halo{width}.log"
        result, messages = run_cmd(
            cmd,
            cwd=input_dir,
            stdout=log_file,
            stderr=log_file,
        )
        if result != 0:
            raise RuntimeError(
                f"Failed to generate regional halo-{width} mosaic: {messages}"
            )
        if not mosaic.is_file() or mosaic.stat().st_size == 0:
            raise FileNotFoundError(f"make_solo_mosaic did not create {mosaic}")

    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Regional grid generation is incomplete:\n{names}")


@contextmanager
def regional_grid_aliases(*, c_res: int, halo: int, input_dir: Path) -> Iterator[None]:
    """Temporarily expose halo-4 files under names expected by sfc_climo_gen."""

    input_dir = Path(input_dir)
    boundary_halo = int(halo) + 1
    aliases = {
        input_dir / f"C{c_res}_grid.tile7.nc": input_dir
        / f"C{c_res}_grid.tile7.halo{boundary_halo}.nc",
        input_dir / f"oro.C{c_res}.tile7.nc": input_dir
        / f"C{c_res}_oro_data.tile7.halo{boundary_halo}.nc",
        input_dir / f"C{c_res}_mosaic.nc": input_dir
        / f"C{c_res}_mosaic.halo{boundary_halo}.nc",
    }
    for alias, target in aliases.items():
        if not target.is_file():
            raise FileNotFoundError(target)
        alias.unlink(missing_ok=True)
        alias.symlink_to(os.path.relpath(target, start=alias.parent))

    try:
        yield
    finally:
        for alias in aliases:
            alias.unlink(missing_ok=True)


def _model_settings(model: str) -> dict:
    from fv3_state import state

    varmap_dir = Path(state.fix_src) / "varmap_tables"
    if model == "HRRR":
        return {
            "varmap_file": varmap_dir / "GSDphys_var_map.txt",
            "geogrid_file_input_grid": Path(state.fix_src)
            / "am"
            / "geo_em.d01.nc_HRRRX",
        }
    return {
        "varmap_file": varmap_dir / "GFSphys_var_map.txt",
        "geogrid_file_input_grid": None,
    }


def _copy_boundary_output(domain: str, model_hour: int) -> None:
    from fv3_state import state
    from fv3_utils import cp

    source = Path(state.tmp) / "chgres_cube" / domain / "gfs.bndy.nc"
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(
            f"chgres_cube did not create the hour-{model_hour:03d} boundary: {source}"
        )
    destination_dir = Path(state.tmp) / "chgres_cube" / "boundaries"
    destination_dir.mkdir(parents=True, exist_ok=True)
    cp(source, destination_dir / boundary_filename(model_hour))


def _clear_chgres_outputs(domain: str) -> None:
    """Remove regional outputs that could otherwise mask a failed rerun."""

    from fv3_state import state

    directory = Path(state.tmp) / "chgres_cube" / domain
    if not directory.is_dir():
        return
    for pattern in ("out.*.nc", "gfs.bndy.nc", "gfs_ctrl.nc"):
        for path in directory.glob(pattern):
            path.unlink()


def run_regional_chgres_cube(n_cpus: int) -> None:
    """Generate regional ICs and every LBC needed through total_run_hours."""

    from chgres_cube import (
        ChgresCubeConfig,
        RunSpec,
        load_block,
        plan_runs,
        run_chgres,
    )
    from fv3_ic_data import get_ic_data
    from fv3_state import state

    grid = validate_regional_grid_files()
    block = load_block("regional")
    initial_plans = plan_runs("regional", 7, block)
    atmospheric_models = [
        model for model, overrides in initial_plans if overrides["convert_atm"]
    ]
    if len(atmospheric_models) != 1:
        raise ValueError(
            "Regional setup must resolve to exactly one atmospheric source"
        )
    atmospheric_model = atmospheric_models[0]

    source_cycle, boundary_times = plan_boundary_times(
        atmospheric_model,
        state.init_datetime,
        int(state.forecast_hour),
        int(state.total_run_hours),
    )
    interval = boundary_interval_hours(atmospheric_model)
    if state.halo_blend is None or int(state.halo_blend) < 0:
        raise ValueError("Regional halo_blend must be non-negative")

    initial_requests: dict[str, tuple[datetime, int]] = {}
    for model, _ in initial_plans:
        if model == atmospheric_model:
            model_cycle = source_cycle
            model_forecast_hour = int(state.forecast_hour)
        else:
            # HRRR has no soil levels.  Use the most recent GFS cycle and
            # forecast hour valid at the regional model initialization.
            model_cycle = _latest_gfs_cycle(state.init_datetime)
            model_forecast_hour = int(
                (state.init_datetime - model_cycle).total_seconds() // 3600
            )

        _validate_source_cycle(model, model_cycle)
        _validate_source_forecast_hours(
            model,
            model_cycle,
            [
                BoundaryTime(
                    model_hour=0,
                    source_forecast_hour=model_forecast_hour,
                    valid_time=state.init_datetime,
                )
            ],
        )
        initial_requests[model] = (model_cycle, model_forecast_hour)

    state.bc_update_interval = interval
    state.regional_bc_hours = [item.model_hour for item in boundary_times]
    state.regional_bc_source = atmospheric_model
    state.regional_source_cycle = source_cycle.strftime("%Y%m%d%HZ")

    boundary_dir = Path(state.tmp) / "chgres_cube" / "boundaries"
    shutil.rmtree(boundary_dir, ignore_errors=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    required_downloads = {
        (model, cycle, forecast_hour)
        for model, (cycle, forecast_hour) in initial_requests.items()
    }
    required_downloads.update(
        (atmospheric_model, source_cycle, item.source_forecast_hour)
        for item in boundary_times[1:]
    )
    downloaded = {
        key: get_ic_data(
            key[0],
            forecast_hour=key[2],
            cycle_datetime=key[1],
        )
        for key in sorted(
            required_downloads,
            key=lambda request: (
                request[0],
                request[1],
                request[2],
            ),
        )
    }

    canonical_mosaic = (
        Path(state.tmp)
        / "chgres_cube"
        / "mosaics"
        / f"C{grid.resolution}_mosaic.halo{grid.boundary_halo}.nc"
    )
    base = ChgresCubeConfig(
        mosaic_file_target_grid=canonical_mosaic,
        fix_dir_target_grid=Path(state.tmp) / "input" / "fix_sfc",
        orog_dir_target_grid=Path(state.tmp) / "input",
        orog_files_target_grid=[grid.orography.name],
        vcoord_file_target_grid=Path(state.fix_src)
        / "am"
        / f"global_hyblev.l{state.levels}.txt",
        cycle_year=state.init_datetime.year,
        cycle_mon=state.init_datetime.month,
        cycle_day=state.init_datetime.day,
        cycle_hour=state.init_datetime.hour,
        regional=1,
        halo_bndy=grid.boundary_halo,
        halo_blend=int(state.halo_blend),
    )

    managed = {
        "mosaic_file_target_grid",
        "orog_dir_target_grid",
        "orog_files_target_grid",
        "data_dir_input_grid",
        "grib2_file_input_grid",
        "external_model",
        "geogrid_file_input_grid",
        "varmap_file",
        "cycle_year",
        "cycle_mon",
        "cycle_day",
        "cycle_hour",
        "regional",
        "halo_bndy",
        "halo_blend",
    }
    atmospheric_overrides = next(
        overrides
        for model, overrides in initial_plans
        if model == atmospheric_model and overrides["convert_atm"]
    )
    boundary_base = replace(
        base,
        **{
            key: value
            for key, value in atmospheric_overrides.items()
            if key not in managed
        },
    )

    _clear_chgres_outputs("regional")
    for model, overrides in initial_plans:
        model_cycle, model_forecast_hour = initial_requests[model]
        data_dir, data_file = downloaded[(model, model_cycle, model_forecast_hour)]
        user_overrides = {
            key: value for key, value in overrides.items() if key not in managed
        }
        config = replace(base, **user_overrides)
        config = replace(
            config,
            external_model=model,
            data_dir_input_grid=Path(data_dir),
            grib2_file_input_grid=Path(data_file),
            **_model_settings(model),
        )
        run_chgres(
            RunSpec(
                domain="regional",
                source_mosaic=grid.mosaic,
                config=config,
            ),
            n_cpus,
        )
        if config.convert_atm:
            _copy_boundary_output("regional", 0)

    for item in boundary_times[1:]:
        data_dir, data_file = downloaded[
            (
                atmospheric_model,
                source_cycle,
                item.source_forecast_hour,
            )
        ]
        domain = f"regional_boundary{item.model_hour:03d}"
        _clear_chgres_outputs(domain)
        config = replace(
            boundary_base,
            external_model=atmospheric_model,
            data_dir_input_grid=Path(data_dir),
            grib2_file_input_grid=Path(data_file),
            convert_atm=True,
            convert_sfc=False,
            convert_nst=False,
            cycle_year=item.valid_time.year,
            cycle_mon=item.valid_time.month,
            cycle_day=item.valid_time.day,
            cycle_hour=item.valid_time.hour,
            regional=2,
            **_model_settings(atmospheric_model),
        )
        run_chgres(
            RunSpec(domain=domain, source_mosaic=grid.mosaic, config=config),
            n_cpus,
        )
        _copy_boundary_output(domain, item.model_hour)


def _expected_boundary_paths(directory: Path) -> list[Path]:
    from fv3_state import state

    hours = state.get("regional_bc_hours")
    if not hours:
        raise ValueError("Regional boundary-hour plan is missing from state")
    return [directory / boundary_filename(int(hour)) for hour in hours]


def link_regional_boundaries() -> None:
    """Link all pre-generated LBC files into the current INPUT directory."""

    from fv3_state import state

    if not is_regional_grid(state.gtype):
        return

    boundary_dir = Path(state.work_dir) / "BC"
    input_dir = Path(state.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    expected = _expected_boundary_paths(boundary_dir)
    missing = [
        path for path in expected if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing staged regional boundary files:\n{names}")

    for old_link in input_dir.glob("gfs_bndy.tile7.*.nc"):
        old_link.unlink()
    for source in expected:
        link = input_dir / source.name
        link.symlink_to(os.path.relpath(source, start=input_dir))


def stage_regional_boundaries() -> None:
    """Copy generated LBCs to work_dir/BC and link them into INPUT."""

    from fv3_state import state

    if not is_regional_grid(state.gtype):
        return

    source_dir = Path(state.tmp) / "chgres_cube" / "boundaries"
    sources = _expected_boundary_paths(source_dir)
    missing = [
        path for path in sources if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Regional boundary generation is incomplete:\n{names}")

    destination_dir = Path(state.work_dir) / "BC"
    shutil.rmtree(destination_dir, ignore_errors=True)
    destination_dir.mkdir(parents=True)
    for source in sources:
        shutil.copy2(source, destination_dir / source.name)
    link_regional_boundaries()


def link_regional_initial_conditions() -> None:
    """Expose regional cold-start/grid files under names used by FV3-LAM."""

    from fv3_state import state

    if not is_regional_grid(state.gtype):
        return

    input_dir = Path(state.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = Path(state.grid)
    halo = int(state.halo)

    # Generated bundles retain relative INPUT -> GRID links, but an externally
    # assembled bundle may contain only the canonical files under GRID.
    # Recreate those links before adding FV3-LAM's short runtime aliases.
    grid_targets = [
        *[f"C{state.c_res}_grid.tile7.halo{width}.nc" for width in (0, halo, halo + 1)],
        *[f"C{state.c_res}_mosaic.halo{width}.nc" for width in (0, halo, halo + 1)],
    ]
    for target_name in grid_targets:
        target = input_dir / target_name
        if target.is_file():
            continue
        source = grid_dir / target_name
        if not source.is_file():
            raise FileNotFoundError(source)
        target.unlink(missing_ok=True)
        target.symlink_to(os.path.relpath(source, start=input_dir))

    aliases = {
        "gfs_data.nc": "gfs_data.tile7.halo0.nc",
        "sfc_data.nc": "sfc_data.tile7.halo0.nc",
        "grid_spec.nc": f"C{state.c_res}_mosaic.halo{halo}.nc",
        "grid.tile7.halo0.nc": f"C{state.c_res}_grid.tile7.halo0.nc",
        f"grid.tile7.halo{halo + 1}.nc": (
            f"C{state.c_res}_grid.tile7.halo{halo + 1}.nc"
        ),
        "oro_data.nc": f"C{state.c_res}_oro_data.tile7.halo0.nc",
        f"oro_data.tile7.halo{halo + 1}.nc": (
            f"C{state.c_res}_oro_data.tile7.halo{halo + 1}.nc"
        ),
    }

    # The GSL orographic-drag fields are physics-suite dependent. Expose their
    # fixed runtime names when the optional preprocessing products are present.
    for field in ("ss", "ls"):
        target_name = f"C{state.c_res}_oro_data_{field}.tile7.halo0.nc"
        if (input_dir / target_name).is_file():
            aliases[f"oro_data_{field}.nc"] = target_name

    for alias_name, target_name in aliases.items():
        target = input_dir / target_name
        if not target.is_file():
            raise FileNotFoundError(target)
        alias = input_dir / alias_name
        alias.unlink(missing_ok=True)
        alias.symlink_to(target_name)
