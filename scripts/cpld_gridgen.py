"""cpld_gridgen (UFS_UTILS): MOM6/CICE6 fix grids for this cubed sphere.

Follows ush/cpld_gridgen.sh. From the MOM6 supergrid, mask, bathymetry and
edits (fix/mom6/<res>), cpld_gridgen writes the tripole and CICE grids,
SCRIP files and the ocean mask mapped to the atmosphere mosaic, which
coupled UFS configurations (UFS-S2S) read. SHiELD does not read them.
For any resolution but 025 the program needs Ct.mx025_SCRIP.nc in the same
output directory (gen_fixgrid.F90), so the 025 grid is generated first.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import f90nml
import xarray as xr
from fv3_runtime import fix_file, get_launcher, log
from fv3_state import state
from fv3_utils import find_tool, run_cmd

# NI, NJ, bathymetry, edits and mask edit per resolution (cpld_gridgen.sh).
RESOLUTIONS = {
    "500": (72, 35, "ocean_topog.nc", "none", False),
    "100": (360, 320, "topog.nc", "topo_edits_011818.nc", True),
    "050": (720, 576, "ocean_topog.nc", "none", False),
    "025": (1440, 1080, "ocean_topog.nc", "All_edits.nc", False),
}
# Destination rectilinear grids of the post weights, per resolution.
RECTS = {
    "5p0": "36,72",
    "1p0": "181,360",
    "0p5": "361,720",
    "0p25": "721,1440",
}
POST_RECTS = {
    "500": ["5p0"],
    "100": ["5p0", "1p0"],
    "050": ["5p0", "1p0", "0p5"],
    "025": ["5p0", "1p0", "0p5", "0p25"],
}


def gridgen_res(value: object) -> str:
    """'100' from 100, '100' or 'mx100'; '025' from 25."""
    res = f"{int(str(value).removeprefix('mx')):03d}"
    if res not in RESOLUTIONS:
        raise ValueError(
            f"cpld_gridgen_res must be one of {sorted(RESOLUTIONS)}, got {value}"
        )
    return res


def _atm_mosaic(fv3_dir: Path) -> str:
    """fv3dir/C<res>/ as cpld_gridgen reads it: the six-tile mosaic and tiles.

    A nested run's C<res>_mosaic.nc includes the nests; its six global tiles
    are in C<res>_coarse_mosaic.nc (fv3_make_mosaic).
    """
    atmres = f"C{state.c_res}"
    dest = fv3_dir / atmres
    dest.mkdir(parents=True, exist_ok=True)
    grid = Path(state.grid)
    mosaic = "coarse_mosaic" if state.gtype == "nest" else "mosaic"
    links = {f"{atmres}_mosaic.nc": grid / f"{atmres}_{mosaic}.nc"}
    for t in range(1, 7):
        links[f"{atmres}_grid.tile{t}.nc"] = grid / f"{atmres}_grid.tile{t}.nc"
    for name, target in links.items():
        if not target.exists():
            raise FileNotFoundError(f"cpld_gridgen: missing {target}")
        (dest / name).unlink(missing_ok=True)
        (dest / name).symlink_to(target.resolve())
    return atmres


def _gridgen(
    res: str, out_dir: Path, fv3_dir: Path, atmres: str, postwgts: bool
) -> None:
    ni, nj, topog, edits, maskedit = RESOLUTIONS[res]
    src = fix_file(f"mom6/{res}/ocean_hgrid.nc").parent
    for name in ("ocean_mask.nc", topog) + (() if edits == "none" else (edits,)):
        fix_file(f"mom6/{res}/{name}")

    log_file = state.logs / f"cpld_gridgen_mx{res}.log"
    if postwgts:
        ncremap = find_tool("ncremap")
        for rect in POST_RECTS[res]:
            cmd = [
                ncremap,
                "-g",
                str(out_dir / f"rect.{rect}_SCRIP.nc"),
                "-G",
                f"latlon={RECTS[rect]}#lon_typ=grn_ctr#lat_typ=cap",
            ]
            result, msgs = run_cmd(cmd, cwd=out_dir, stdout=log_file, stderr=log_file)
            if result != 0:
                log.error(msgs)
                raise RuntimeError(f"ncremap failed for rect.{rect} (see {log_file})")

    # Directory names are character(256) and joined with file names in the
    # program (charstrings.F90, gen_fixgrid.F90), so short links in out_dir,
    # the working directory, stand in for them.
    links = {f"mom6_{res}": src, "fv3": fv3_dir}
    for name, target in links.items():
        (out_dir / name).unlink(missing_ok=True)
        (out_dir / name).symlink_to(Path(target).resolve())
    nml = {
        "grid_nml": {
            "ni": ni,
            "nj": nj,
            "dirsrc": f"mom6_{res}",
            "dirout": ".",
            "fv3dir": "fv3",
            "topofile": topog,
            "editsfile": edits,
            "res": res,
            "atmres": atmres,
            "npx": int(state.c_res),
            "editmask": maskedit,
            "debug": False,
            "do_postwgts": postwgts,
        }
    }
    f90nml.write(nml, out_dir / "grid.nml", force=True)

    cmd = [*get_launcher(1), str(state.ufs_exe / "cpld_gridgen")]
    result, msgs = run_cmd(cmd, cwd=out_dir, stdout=log_file, stderr=log_file)
    # The program ends with a plain stop (exit 0) on some errors, so its
    # outputs are checked as well.
    for name in links:
        (out_dir / name).unlink()
    scrip = out_dir / f"Ct.mx{res}_SCRIP_land.nc"
    cice = out_dir / f"grid_cice_NEMS_mx{res}.nc"
    if result != 0 or not scrip.exists() or not cice.exists():
        log.error(msgs)
        raise RuntimeError(f"cpld_gridgen failed for mx{res} (see {log_file})")

    mesh = out_dir / f"mesh.mx{res}.nc"
    cmd = [
        *get_launcher(1),
        find_tool("ESMF_Scrip2Unstruct"),
        str(scrip),
        str(mesh),
        "0",
    ]
    result, msgs = run_cmd(cmd, cwd=out_dir, stdout=log_file, stderr=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError(f"ESMF_Scrip2Unstruct failed for mx{res} (see {log_file})")

    # ncks -O -v kmt
    with xr.open_dataset(cice) as ds:
        ds[["kmt"]].load().to_netcdf(out_dir / f"kmtu_cice_NEMS_mx{res}.nc")
    log.info(f"cpld_gridgen wrote the mx{res} grids for {atmres} in {out_dir}")


def run_cpld_gridgen() -> None:
    if not state.run_cpld_gridgen:
        return
    res = gridgen_res(state.cpld_gridgen_res)
    postwgts = bool(state.cpld_gridgen_postwgts)
    out_dir = Path(state.ic_data) / "cpld_gridgen"
    out_dir.mkdir(parents=True, exist_ok=True)
    fv3_dir = Path(state.tmp) / "cpld_gridgen"
    atmres = _atm_mosaic(fv3_dir)
    if res != "025" and not (out_dir / "Ct.mx025_SCRIP.nc").exists():
        _gridgen("025", out_dir, fv3_dir, atmres, postwgts)
    _gridgen(res, out_dir, fv3_dir, atmres, postwgts)
    shutil.rmtree(fv3_dir, ignore_errors=True)
