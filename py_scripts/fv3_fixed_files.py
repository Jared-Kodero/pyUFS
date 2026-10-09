import os
from pathlib import Path

import pandas as pd
from fv3_runtime import fix_file, log
from fv3_state import state
from fv3_update_fix import PHYSICS_FILES, volcanic_files
from fv3_utils import cp

# Earliest year of the CO2 record (co2dat_4a/global_co2historicaldata_1956.txt).
CO2_FIRST_YEAR = 1956


def run_years() -> range:
    """Calendar years spanned by the whole run, all segments included."""
    start = state.init_datetime
    end = start + pd.Timedelta(hours=int(state.total_run_hours or 0))
    return range(start.year, end.year + 1)


def link_fixed_file(file: Path, link_dir: Path | None = None) -> None:
    """Copy one fix_src/am file into FIXED and link it into `link_dir`.

    `link_dir` defaults to INPUT. The volcanic tables are read from the model
    working directory (radiation_aerosols.f opens them without a directory).
    """
    dest = state.fix / file.name
    if not dest.exists():
        cp(file, dest)

    link_dir = Path(state.input if link_dir is None else link_dir)
    link = link_dir / file.name
    link.unlink(missing_ok=True)
    link.symlink_to(os.path.relpath(dest, start=link_dir))


def co2_file(year: int) -> Path | None:
    """CO2 table for `year`, or the latest earlier year that is available.

    The radiation code (radiation_gases.f) does the same search when the
    table of the model year is missing and extrapolates from the year found,
    so a missing table is a warning, not an error.
    """
    for y in range(year, CO2_FIRST_YEAR - 1, -1):
        path = fix_file(f"am/co2historicaldata_{y}.txt", required=False)
        if path is not None:
            if y != year:
                log.warning(
                    f"co2historicaldata_{year}.txt unavailable; staging {y}, from "
                    + "which the model extrapolates CO2"
                )
            return path
    return None


def update_fixed_files():
    fix_dir = state.fix_src / "am"

    for name in PHYSICS_FILES:
        fix_file(f"am/{name}")
        link_fixed_file(fix_dir / name)

    for year in run_years():
        path = co2_file(year)
        if path is None:
            raise FileNotFoundError(f"No CO2 table for {year} or earlier in {fix_dir}")
        link_fixed_file(fix_dir / path.name)

    # Volcanic tables exist for 1850-1999; later years use the lowest value.
    for name in volcanic_files(run_years()):
        fix_file(f"am/{name}")
        link_fixed_file(fix_dir / name, link_dir=state.work_dir)
