import os
from pathlib import Path

import pandas as pd
from fv3_runtime import log, report_missing_fixed_files
from fv3_state import state
from fv3_utils import cp


def run_years() -> range:
    """Calendar years spanned by the whole run, all segments included."""
    start = state.init_datetime
    end = start + pd.Timedelta(hours=int(state.total_run_hours or 0))
    return range(start.year, end.year + 1)


def link_fixed_file(file: Path) -> None:
    """Copy one fix_src/am file into FIXED and link it into INPUT."""
    dest = state.fix / file.name
    if not dest.exists():
        cp(file, dest)

    link = Path(state.input) / file.name
    link.unlink(missing_ok=True)
    link.symlink_to(os.path.relpath(dest, start=state.input))


def update_fixed_files():
    dt = state.init_datetime
    year = dt.year
    fix_dir = state.fix_src / "am"

    required_files = [
        "aerosol.dat",
        f"co2historicaldata_{year}.txt",
        "co2historicaldata_glob.txt",
        "co2monthlycyc.txt",
        "sfc_emissivity_idx.txt",
        "solarconstant_noaa_an.txt",
        "volcanic_aerosols_1990-1999.txt",
        "global_h2oprdlos.f77",
        "global_o3prdlos.f77",
    ]

    # Radiation reads the CO2 file of each model year and the volcanic file
    # of each decade (1850-1999); stage those spanned by multi-year runs.
    optional_files = [f"co2historicaldata_{y}.txt" for y in run_years()]
    optional_files += [
        f"volcanic_aerosols_{d}-{d + 9}.txt"
        for d in sorted({y - y % 10 for y in run_years()})
        if 1850 <= d <= 1990
    ]

    missing_files = []

    for name in required_files:
        file = fix_dir / name
        if file.exists():
            link_fixed_file(file)
        else:
            missing_files.append(file)

    for name in dict.fromkeys(optional_files):
        file = fix_dir / name
        if name in required_files:
            continue
        if file.exists():
            link_fixed_file(file)
        else:
            log.info(f"Optional fix file not found: {file}")

    if missing_files:
        report_missing_fixed_files(missing_files, sub_dir="am")
