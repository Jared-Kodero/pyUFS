import os
from pathlib import Path

from fv3_runtime import report_missing_fixed_files
from fv3_state import state
from fv3_utils import cp


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

    missing_files = []

    for name in required_files:
        file = fix_dir / name
        if file.exists():
            dest = state.fix / name
            if not dest.exists():
                cp(file, dest)

            link = Path(state.input) / name
            link.unlink(missing_ok=True)

            rel_target = os.path.relpath(dest, start=state.input)
            link.symlink_to(rel_target)
        else:
            missing_files.append(file)

    if missing_files:
        report_missing_fixed_files(missing_files, sub_dir="am")
