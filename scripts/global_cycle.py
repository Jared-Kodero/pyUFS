"""global_cycle (UFS_UTILS) on one set of cubed-sphere surface files.

Follows ush/global_cycle.sh of the UFS_UTILS build in the preprocessing
image: namelists fort.35 (NAMSFC), fort.36 (NAMCYC) and fort.37 (NAMSFCD),
and the sstclm and salclm links. Each MPI rank processes the files
fnbgsi/fngrid/fnorog.NNN, NNN = rank + 1, and writes fnbgso.NNN
(global_cycle.fd/cycle.f90), so all files of one call must share the tile
dimensions (one namelist idim, jdim).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import netCDF4
import pandas as pd
from fv3_runtime import fix_file, get_launcher, log
from fv3_state import state
from fv3_update_fix import NAMSFC_FILES
from fv3_utils import run_cmd

# Relaxation settings of ush/global_cycle.sh (FSLPL, FSOTL, FVETL, FSMCL2).
RELAXATION = {"FSLPL": 99999.0, "FSOTL": 99999.0, "FVETL": 99999.0}
FSMCL2 = 60.0

# NAMCYC settings of ush/global_cycle.sh and global_cycle_driver.sh. ialb,
# isot and ivegsrc match gfs_physics_nml in configs/input_nml.yaml.
DELTSFC = 6.0
NAMCYC = {
    "ialb": 1,
    "use_ufo": True,
    "donst": "NO",
    "do_sfccycle": True,
    "do_lndinc": False,
    "isot": 1,
    "ivegsrc": 1,
    "zsea1_mm": 0,
    "zsea2_mm": 0,
    "max_tasks": 99999,
}

SALCLM = "am/global_salclm.t1534.3072.1536.nc"


def _value(v: object) -> str:
    """A Fortran namelist value."""
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _group(name: str, entries: dict) -> str:
    body = "".join(f"  {k}={_value(v)},\n" for k, v in entries.items())
    return f" &{name}\n{body} /\n"


def namsfc_entries(
    sst: str | None, ice: str | None, snow: str | None, extra: dict | None
) -> dict:
    """NAMSFC: the climatologies of the model's surface cycling, the analyses.

    The climatologies are those of the model namelist (fv3_namelists
    update_namsfc), so global_cycle and the in-model cycling (fhcyc) relax
    to the same fields. fnmldc is not a NAMSFC entry of the ccpp-physics
    sfcsub.F that global_cycle is built with. A blank analysis name keeps the
    first guess and climatology for that field (sfcsub.F).
    """
    entries = {
        key.upper(): str(fix_file(f"am/{name}"))
        for key, name in NAMSFC_FILES.items()
        if key != "fnmldc"
    }
    entries["FNZORC"] = "igbp"
    entries["FNTSFA"] = str(sst or "")
    entries["FNACNA"] = str(ice or "")
    entries["FNSNOA"] = str(snow or "")
    entries["LDEBUG"] = False
    entries.update(RELAXATION)
    for k in (2, 3, 4):
        entries[f"FSMCL({k})"] = FSMCL2
    for key, value in (extra or {}).items():
        entries[str(key).upper()] = value
    return entries


def _dims(sfc: Path) -> tuple[int, int, int]:
    with netCDF4.Dataset(sfc) as ds:
        return (
            len(ds.dimensions["xaxis_1"]),
            len(ds.dimensions["yaxis_1"]),
            len(ds.dimensions["zaxis_1"]),
        )


def run_global_cycle(
    label: str,
    files: list[tuple[Path, Path, Path]],
    work_dir: Path,
    namsfc: dict,
    valid: pd.Timestamp,
) -> None:
    """Update the surface files of `files` [(sfc, grid, orog), ...] in place.

    `valid` is the analysis time (the model start), passed as iy, im, id,
    ih with fh = 0. The files are replaced only after every rank succeeded.
    """
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    dims = {_dims(sfc) for sfc, _grid, _orog in files}
    if len(dims) != 1:
        raise ValueError(f"global_cycle {label}: tiles differ in size {sorted(dims)}")
    idim, jdim, lsoil = dims.pop()

    for n, (sfc, grid, orog) in enumerate(files, start=1):
        for path in (sfc, grid, orog):
            if not path.exists():
                raise FileNotFoundError(f"global_cycle {label}: missing {path}")
        (work_dir / f"fnbgsi.{n:03d}").symlink_to(sfc.resolve())
        (work_dir / f"fngrid.{n:03d}").symlink_to(grid.resolve())
        (work_dir / f"fnorog.{n:03d}").symlink_to(orog.resolve())
        shutil.copyfile(sfc, work_dir / f"fnbgso.{n:03d}")

    (work_dir / "sstclm").symlink_to(namsfc["FNTSFC"])
    salclm = fix_file(SALCLM, required=False)
    if salclm is not None:
        (work_dir / "salclm").symlink_to(salclm)

    namcyc = {
        "idim": idim,
        "jdim": jdim,
        "lsoil": lsoil,
        "iy": valid.year,
        "im": valid.month,
        "id": valid.day,
        "ih": valid.hour,
        "fh": 0.0,
        "deltsfc": DELTSFC,
        **NAMCYC,
    }
    namsfcd = {"NST_FILE": "NULL", "LND_SOI_FILE": "NULL", "DO_SNO_INC": False}
    (work_dir / "fort.35").write_text(_group("NAMSFC", namsfc))
    (work_dir / "fort.36").write_text(_group("NAMCYC", namcyc))
    (work_dir / "fort.37").write_text(_group("NAMSFCD", namsfcd))

    log_file = state.logs / f"global_cycle_{label}.log"
    cmd = [*get_launcher(len(files)), str(state.ufs_exe / "global_cycle")]
    result, msgs = run_cmd(cmd, cwd=work_dir, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"global_cycle failed for {label} (see {log_file})")

    for n, (sfc, _grid, _orog) in enumerate(files, start=1):
        tmp = sfc.with_name(f".{sfc.name}.cycled")
        shutil.copyfile(work_dir / f"fnbgso.{n:03d}", tmp)
        os.replace(tmp, sfc)
    log.info(f"global_cycle updated {len(files)} {label} surface file(s)")
