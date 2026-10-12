"""emcsfc_ice_blend (UFS_UTILS): blended sea-ice concentration analysis.

Follows ush/emcsfc_ice_blend.sh: the IMS ice cover is converted to GRIB2
when needed, its ICEC records are interpolated to the 5-minute grid
(copygb2), blended with the MMAB 5-minute concentration, and the result is
converted to GRIB1 with the land bitmap that global_cycle reads (FNACNA).
cnvgrib, copygb2 and copygb come from NCEP grib_util and must be on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from fv3_runtime import fix_file, log
from fv3_state import state
from fv3_utils import find_tool

GRID173 = (
    "0 0 0 0 0 0 0 0 4320 2160 0 0 89958000 42000 48 -89958000 359958000 83000 83000 0"
)
BLENDED = "seaice.5min.blend.grb"


def run_ice_blend(work_dir: Path) -> Path:
    """Run emcsfc_ice_blend; returns the blended GRIB1 ice analysis."""
    ims = Path(os.path.expandvars(str(state.ims_ice_file)))
    five_min = Path(os.path.expandvars(str(state.five_min_ice_file)))
    for path, what in ((ims, "IMS ice"), (five_min, "MMAB 5-minute ice")):
        if not path.exists():
            raise FileNotFoundError(f"{what} file not found: {path}")
    tools = {
        name: find_tool(name) for name in ("wgrib2", "cnvgrib", "copygb2", "copygb")
    }
    mask = fix_file("am/emcsfc_gland5min.grib2")

    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    log_file = state.logs / "emcsfc_ice_blend.log"

    def step(cmd: list[str], env: dict | None = None) -> None:
        with open(log_file, "a") as out:
            rc = subprocess.run(
                cmd, cwd=work_dir, env=env, stdout=out, stderr=out, check=False
            ).returncode
        if rc != 0:
            raise RuntimeError(
                f"emcsfc_ice_blend step failed ({' '.join(cmd)}); see {log_file}"
            )

    sec0 = subprocess.run(
        [tools["wgrib2"], "-Sec0", str(ims)],
        capture_output=True,
        text=True,
        check=False,
    )
    if "grib1 message" in sec0.stdout + sec0.stderr:
        step([tools["cnvgrib"], "-g12", "-p40", str(ims), "ims.grib2"])
    else:
        shutil.copyfile(ims, work_dir / "ims.grib2")
    step([tools["wgrib2"], "ims.grib2", "-match", "ICEC", "-grib", "ims.icec.grib2"])
    step(
        [
            tools["copygb2"],
            "-x",
            "-i3",
            "-g",
            GRID173,
            "ims.icec.grib2",
            "ims.icec.5min.grib2",
        ]
    )

    # The program reads the names from FORT11/15/17/51 into character(200)
    # (emcsfc_ice_blend.f90), so the inputs are linked under short names.
    (work_dir / "mask.grib2").symlink_to(Path(mask).resolve())
    (work_dir / "five_min.grib2").symlink_to(five_min.resolve())
    env = {
        **os.environ,
        "FORT17": "mask.grib2",
        "FORT11": "ims.icec.5min.grib2",
        "FORT15": "five_min.grib2",
        "FORT51": BLENDED,
    }
    step([str(state.ufs_exe / "emcsfc_ice_blend")], env=env)

    step(
        [
            tools["wgrib2"],
            "-set_int",
            "3",
            "51",
            "42000",
            BLENDED,
            "-grib",
            f"{BLENDED}.corner",
        ]
    )
    step([tools["cnvgrib"], "-g21", f"{BLENDED}.corner", f"{BLENDED}.bitmap"])
    (work_dir / BLENDED).unlink()
    step([tools["copygb"], "-M", "#1.57", "-x", f"{BLENDED}.bitmap", BLENDED])
    for suffix in (".corner", ".bitmap"):
        (work_dir / f"{BLENDED}{suffix}").unlink(missing_ok=True)

    out = work_dir / BLENDED
    if not out.exists():
        raise RuntimeError(f"emcsfc_ice_blend wrote no {out} (see {log_file})")
    log.info(f"emcsfc_ice_blend wrote the ice analysis {out}")
    return out
