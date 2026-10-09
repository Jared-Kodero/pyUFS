"""emcsfc_snow2mdl (UFS_UTILS): snow analysis on the T1534 Gaussian grid.

Follows ush/emcsfc_snow.sh: the IMS snow cover (and optional AFWA snow
depth) is interpolated to the grid of the global_slmask, latitude and
longitude files, and written as GRIB1 for global_cycle (FNSNOA). The output
is dated by the IMS file, at 00 UTC as in the script.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fv3_runtime import fix_file, log
from fv3_state import state
from fv3_utils import find_tool, run_cmd

MODEL_GRID = "t1534.3072.1536"
OUTPUT = "snogrb_model"


def grib_date(path: Path) -> tuple[int, int, int]:
    """Year, month and day of the first record (GRIB2 or GRIB1)."""
    wgrib2 = find_tool("wgrib2")
    sec0 = subprocess.run(
        [wgrib2, "-Sec0", str(path)], capture_output=True, text=True, check=False
    )
    if "grib1 message" in sec0.stdout + sec0.stderr:
        out = subprocess.run(
            [find_tool("wgrib"), "-v", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        date = out.splitlines()[0].split("D=")[-1]
    else:
        out = subprocess.run(
            [wgrib2, "-t", str(path)], capture_output=True, text=True, check=True
        ).stdout
        date = out.splitlines()[0].split("d=")[-1]
    return int(date[0:4]), int(date[4:6]), int(date[6:8])


def _link(work_dir: Path, name: str, target: str | Path | None) -> str:
    """Link `target` into work_dir as `name`; fort.41 paths are character*200
    (program_setup.F90), so the namelist names the short link."""
    if not target:
        return ""
    target = Path(os.path.expandvars(str(target))).resolve()
    if not target.exists():
        raise FileNotFoundError(f"emcsfc_snow2mdl input not found: {target}")
    link = work_dir / name
    link.unlink(missing_ok=True)
    link.symlink_to(target)
    return name


def run_emcsfc_snow(work_dir: Path) -> Path:
    """Run emcsfc_snow2mdl; returns the snow analysis file."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ims = _link(work_dir, "ims_snow", state.ims_snow_file)
    year, month, day = grib_date(work_dir / ims)

    am = {
        name: _link(work_dir, name, fix_file(f"am/{name}"))
        for name in (
            f"global_slmask.{MODEL_GRID}.grb",
            f"global_latitudes.{MODEL_GRID}.grb",
            f"global_longitudes.{MODEL_GRID}.grb",
            f"global_lonsperlat.{MODEL_GRID}.txt",
            "emcsfc_snow_cover_climo.grib2",
        )
    }
    groups = {
        "source_data": {
            "autosnow_file": "",
            "nesdis_snow_file": ims,
            "nesdis_lsmask_file": "",
            "afwa_snow_global_file": _link(
                work_dir, "afwa_global", state.afwa_snow_global_file
            ),
            "afwa_snow_nh_file": _link(work_dir, "afwa_nh", state.afwa_snow_nh_file),
            "afwa_snow_sh_file": _link(work_dir, "afwa_sh", state.afwa_snow_sh_file),
            "afwa_lsmask_nh_file": "",
            "afwa_lsmask_sh_file": "",
        },
        "qc": {"climo_qc_file": am["emcsfc_snow_cover_climo.grib2"]},
        "model_specs": {
            "model_lat_file": am[f"global_latitudes.{MODEL_GRID}.grb"],
            "model_lon_file": am[f"global_longitudes.{MODEL_GRID}.grb"],
            "model_lsmask_file": am[f"global_slmask.{MODEL_GRID}.grb"],
            "gfs_lpl_file": am[f"global_lonsperlat.{MODEL_GRID}.txt"],
        },
        "output_data": {"model_snow_file": f"./{OUTPUT}", "output_grib2": False},
        "output_grib_time": {
            "grib_year": year,
            "grib_month": month,
            "grib_day": day,
            "grib_hour": 0,
        },
        "parameters": {
            "lat_threshold": 55.0,
            "min_snow_depth": 0.05,
            "snow_cvr_threshold": 50.0,
        },
    }
    text = ""
    for name, entries in groups.items():
        text += f" &{name}\n"
        for k, v in entries.items():
            if isinstance(v, bool):
                v = ".true." if v else ".false."
            elif isinstance(v, str):
                v = f'"{v}"'
            text += f"  {k}={v}\n"
        text += " /\n"
    (work_dir / "fort.41").write_text(text)

    log_file = state.logs / "emcsfc_snow2mdl.log"
    exe = str(state.ufs_exe / "emcsfc_snow2mdl")
    result, msgs = run_cmd([exe], cwd=work_dir, stdout=log_file, stderr=log_file)
    out = work_dir / OUTPUT
    if result != 0 or not out.exists():
        log.error(msgs)
        raise RuntimeError(f"emcsfc_snow2mdl failed (see {log_file})")
    log.info(f"emcsfc_snow2mdl wrote the snow analysis {out}")
    return out
