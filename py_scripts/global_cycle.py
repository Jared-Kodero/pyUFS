from pathlib import Path

import f90nml
import pandas as pd
from fv3gfs_runtime import get_launcher, log
from fv3gfs_state import state
from fv3gfs_utils import cp, run_cmd


def run_global_cycle(
    datetime: str,
    c_res: str,
    fhour: str = "00",
    exec_dir: Path | None = None,
    fix_am: Path | None = None,
    tmp_dir: Path | None = None,
    tmp_ic_dir: Path | None = None,
    global_cycle: Path | None = None,
    jcap: int | None = None,
    lonb: int | None = None,
    latb: int | None = None,
    lsoil: int = 4,
    fsmcl2: int = 60,
    fslpl: float = 99999.0,
    fsotl: float = 99999.0,
    fvetl: float = 99999.0,
    ialb: int = 1,
    isot: int = 1,
    ivegsrc: int = 1,
    deltsfc: int = 0,
    use_ufo: bool = True,
    donst: str = "NO",
    do_sfccycle: bool = True,
    do_lndinc: bool = False,
    do_sno_inc: bool = False,
    zsea1: int = 0,
    zsea2: int = 0,
    max_tasks: int = 99999,
    nst_file: str = "NULL",
    lnd_soi_file: str = "NULL",
    cycle_vars: dict | None = None,
    n_nests: int = 0,  # NEW: number of nests for multinest support
    nest_idx: int | None = None,  # NEW: current nest index being processed
):
    """
    Python wrapper for global_cycle.sh.

    Generates global surface analysis (snow, sea ice, vegetation, etc.)
    by writing Fortran namelists and invoking the `global_cycle` executable.

    Parameters
    ----------
    cdate : str
        Analysis date in YYYYMMDDHH format (e.g., "2020010100").
    case : str
        Resolution case string (e.g., "C768").
    fhour : str
        Forecast hour string (default "00").
    basedir : Path
        Root of repository (default: /nwprod2).
    exec_dir : Path
        Executables directory (default: $basedir/gfs_ver/exec).
    fix_am : Path
        Directory containing climatology GRIBs (default: $basedir/gfs_ver/fix/am).
    fix_fv3gfs : Path
        Directory containing FV3 orog/grid files (default: $basedir/gfs_ver/fix/orog/$case).
    cycle_exec : Path
        Path to global_cycle executable (default: exec_dir/global_cycle).
    jcap, lonb, latb : int
        Gaussian climatology resolution parameters. Derived from case if None.
    lsoil, fsmcl2, etc. : misc options
        Numerical parameters from the shell script.
    cycle_vars : dict
        Extra entries to inject into the NAMSFC namelist.
    """

    # Resolution parsing
    cres = state.res
    jcap = jcap if jcap is not None else (2 * cres - 2)
    lonb = lonb if lonb is not None else (4 * cres)
    latb = latb if latb is not None else (2 * cres)

    tmp_dir = Path(tmp_dir)
    tmp_ic_dir = state.tmp / "ic"
    out_dir = tmp_ic_dir / "global_cycle"
    out_dir.mkdir(parents=True, exist_ok=True)
    global_cycle = exec_dir / "global_cycle"

    # Namelist fort.35 (NAMSFC)
    namsfc = {
        "NAMSFC": {
            "FNGLAC": str(fix_am / "global_glacier.2x2.grb"),
            "FNMXIC": str(fix_am / "global_maxice.2x2.grb"),
            "FNTSFC": str(fix_am / "RTGSST.1982.2012.monthly.clim.grb"),
            "FNSNOC": str(fix_am / "global_snoclim.1.875.grb"),
            "FNZORC": "igbp",
            "FNALBC": str(
                fix_am / f"global_snowfree_albedo.bosu.t{jcap}.{lonb}.{latb}.rg.grb"
            ),
            "FNALBC2": str(fix_am / "global_albedo4.1x1.grb"),
            "FNAISC": str(fix_am / "IMS-NIC.blended.ice.monthly.clim.grb"),
            "FNTG3C": str(fix_am / "global_tg3clim.2.6x1.5.grb"),
            "FNVEGC": str(fix_am / "global_vegfrac.0.144.decpercent.grb"),
            "FNVETC": str(fix_am / f"global_vegtype.igbp.t{jcap}.{lonb}.{latb}.rg.grb"),
            "FNSOTC": str(
                fix_am / f"global_soiltype.statsgo.t{jcap}.{lonb}.{latb}.rg.grb"
            ),
            "FNSMCC": str(
                fix_am / f"global_soilmgldas.statsgo.t{jcap}.{lonb}.{latb}.grb"
            ),
            "FNVMNC": str(fix_am / "global_shdmin.0.144x0.144.grb"),
            "FNVMXC": str(fix_am / "global_shdmax.0.144x0.144.grb"),
            "FNSLPC": str(fix_am / "global_slope.1x1.grb"),
            "FNABSC": str(
                fix_am / f"global_mxsnoalb.uariz.t{jcap}.{lonb}.{latb}.rg.grb"
            ),
            "FNMSKH": str(fix_am / "global_slmask.t1534.3072.1536.grb"),
            "FNTSFA": str(tmp_ic_dir / "sstgrb"),
            "FNACNA": str(tmp_ic_dir / "engicegrb"),
            "FNSNOA": str(tmp_ic_dir / "snogrb"),
            "LDEBUG": ".false.",
            "FSLPL": fslpl,
            "FSOTL": fsotl,
            "FVETL": fvetl,
            "FSMCL": [0, fsmcl2, fsmcl2, fsmcl2],
        }
    }
    if cycle_vars:
        namsfc["NAMSFC"].update(cycle_vars)

    try:
        datetime = pd.to_datetime(datetime, format="%Y%m%d%HZ")
    except ValueError:
        raise ValueError("Invalid cdate format. Expected YYYYMMDDHHZ.")

    # Namelist fort.36 (NAMCYC)
    namcyc = {
        "NAMCYC": {
            "idim": cres,
            "jdim": cres,
            "lsoil": lsoil,
            "iy": int(datetime.year),
            "im": int(datetime.month),
            "id": int(datetime.day),
            "ih": int(datetime.hour),
            "fh": fhour,
            "deltsfc": deltsfc,
            "ialb": ialb,
            "use_ufo": use_ufo,
            "donst": donst,
            "do_sfccycle": do_sfccycle,
            "do_lndinc": do_lndinc,
            "isot": isot,
            "ivegsrc": ivegsrc,
            "zsea1_mm": zsea1,
            "zsea2_mm": zsea2,
            "MAX_TASKS": max_tasks,
        }
    }

    # Namelist fort.37 (NAMSFCD)
    namsfcd = {
        "NAMSFCD": {
            "NST_FILE": nst_file,
            "LND_SOI_FILE": lnd_soi_file,
            "DO_SNO_INC": do_sno_inc,
        }
    }

    # Write namelists
    f90nml.write(namsfc, str(tmp_dir / "fort.35"), force=True)
    f90nml.write(namcyc, str(tmp_dir / "fort.36"), force=True)
    f90nml.write(namsfcd, str(tmp_dir / "fort.37"), force=True)

    # Run executable - adjust task count for nests
    ntasks = 6 if n_nests == 0 or nest_idx == 0 else 1
    cmd = [*get_launcher(ntasks), str(global_cycle)]

    # Create unique log file for each nest to avoid overwriting
    if nest_idx is not None and nest_idx > 0:
        log_file = state.logs / f"global_cycle_nest{nest_idx:02d}.log"
    else:
        log_file = state.logs / "global_cycle.log"

    result, msgs = run_cmd(cmd, cwd=tmp_dir, log_file=log_file)
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to run global_cycle")

    # cp outputs to out_dir

    cp(tmp_dir, out_dir)

    # Outputs should be in COMOUT (not explicitly moved by the shell script)
    if nest_idx is not None and nest_idx > 0:
        log.info(f"Global_cycle completed for nest {nest_idx:02d}")
        log.info(f"Global_cycle outputs in: {out_dir}")

    else:
        log.info("Global_cycle completed successfully for all nests.")
        log.info(f"Global_cycle outputs in: {out_dir}")
