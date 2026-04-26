from pathlib import Path

from fv3gfs_runtime import get_launcher, log
from fv3gfs_state import state
from fv3gfs_utils import run_cmd


def write_grid_nml(
    template: Path,
    tmp_ic_dir: Path,
    ni: int,
    NJ: int,
    fix: Path,
    out_dir: Path,
    mosaic_dir: Path,
    topo_file: str,
    edits_file: str,
    res_name: str,
    mosaic_cres: str,
    npx: int,
    mask_edit: str = ".false.",
    debug: str = ".false.",
    do_postwgts: str = ".true.",
):
    """
    Create grid.nml from a template with substitutions.
    """
    text = Path(template).read_text()
    text = text.replace("NI_GLB", str(ni))
    text = text.replace("NJ_GLB", str(NJ))
    text = text.replace("FIXDIR", str(fix))
    text = text.replace("OUTDIR", str(out_dir))
    text = text.replace("MOSAICDIR", str(mosaic_dir))
    text = text.replace("TOPOGFILE", topo_file)
    text = text.replace("EDITSFILE", edits_file)
    text = text.replace("RESNAME", res_name)
    text = text.replace("MOSAICRES", mosaic_cres)
    text = text.replace("NPX", str(npx))
    text = text.replace("DO_MASKEDIT", mask_edit)
    text = text.replace("DO_DEBUG", debug)
    text = text.replace("DO_POSTWGTS", do_postwgts)

    Path(tmp_ic_dir).write_text(text)


def run_gridgen(
    resname: str,
    mosaicres: str,
    mom6_fixdir: Path,
    outdir: Path,
    mosaicdir: Path,
    template: Path,
    gridgen_exec: Path,
    do_postwgts: bool = True,
):
    """
    Run cpld_gridgen and post-process for MOM6/CICE.

    Parameters
    ----------
    resname : str
        Resolution identifier ("500", "100", "050", "025").
    mosaicres : str
        Mosaic resolution (e.g., "C768").
    mom6_fixdir : Path
        Path to MOM6 fixdir root.
    outdir : Path
        Output directory.
    mosaicdir : Path
        Mosaic fix/orog dir.
    template : Path
        Path to grid.nml.IN template.
    gridgen_exec : Path
        Path to `cpld_gridgen` executable.
    do_postwgts : bool
        If True, pre-generate SCRIP files.
    """

    log_file = state.logs / "gridgen.log"

    outdir.mkdir(parents=True, exist_ok=True)

    # Map resname to NI/NJ and files
    if resname == "500":
        NI, NJ = 72, 35
        topog, edits = "ocean_topog.nc", "none"
    elif resname == "100":
        NI, NJ = 360, 320
        topog, edits = "topog.nc", "topo_edits_011818.nc"
    elif resname == "050":
        NI, NJ = 720, 576
        topog, edits = "ocean_topog.nc", "none"
    elif resname == "025":
        NI, NJ = 1440, 1080
        topog, edits = "ocean_topog.nc", "All_edits.nc"
    else:
        raise ValueError(f"Unsupported RESNAME: {resname}")

    # NPX by mosaicres
    npx_map = {
        "C3072": 3072,
        "C1152": 1152,
        "C768": 768,
        "C384": 384,
        "C192": 192,
        "C096": 96,
        "C048": 48,
    }
    NPX = npx_map.get(mosaicres, None)
    if NPX is None:
        raise ValueError(f"Unsupported MOSAICRES: {mosaicres}")

    fixdir = mom6_fixdir / resname

    # Generate grid.nml
    write_grid_nml(
        template,
        outdir / "grid.nml",
        NI,
        NJ,
        fixdir,
        outdir,
        mosaicdir,
        topog,
        edits,
        resname,
        mosaicres,
        NPX,
    )

    # Run gridgen
    cmd = [get_launcher(1), str(gridgen_exec)]
    result, msgs = run_cmd(
        cmd,
        cwd=outdir,
        log_file=log_file,
    )
    if result != 0:
        log.error(msgs)
        raise RuntimeError("Failed to run gridgen")
    # Generate SCRIP files if requested
    if do_postwgts:
        rects = {
            "500": ["rect.5p0_SCRIP.nc"],
            "100": ["rect.5p0_SCRIP.nc", "rect.1p0_SCRIP.nc"],
            "050": ["rect.5p0_SCRIP.nc", "rect.1p0_SCRIP.nc", "rect.0p5_SCRIP.nc"],
            "025": [
                "rect.5p0_SCRIP.nc",
                "rect.1p0_SCRIP.nc",
                "rect.0p5_SCRIP.nc",
                "rect.0p25_SCRIP.nc",
            ],
        }
        for grid in rects[resname]:
            if "5p0" in grid:
                dims = "36,72"
            elif "1p0" in grid:
                dims = "181,360"
            elif "0p5" in grid:
                dims = "361,720"
            elif "0p25" in grid:
                dims = "721,1440"
            else:
                continue

            cmd = [
                *get_launcher(1),
                "ncremap",
                "-g",
                str(outdir / grid),
                "-G",
                f"latlon={dims}#lon_typ=grn_ctr#lat_typ=cap",
            ]
            result, msgs = run_cmd(cmd, log_file=log_file)
            if result != 0:
                log.error(msgs)
                raise RuntimeError(f"Failed to generate scrip file for {grid}")

        # Ice mesh
        fs = outdir / f"Ct.mx{resname}_SCRIP_land.nc"
        fd = outdir / f"mesh.mx{resname}.nc"
        cmd = [*get_launcher(1), "ESMF_Scrip2Unstruct", str(fs), str(fd), "0"]
        result, msgs = run_cmd(cmd, log_file=log_file)
        if result != 0:
            log.error(msgs)
            raise RuntimeError("Failed to generate ice mesh file")

        # kmt file
        fs = outdir / f"grid_cice_NEMS_mx{resname}.nc"
        fd = outdir / f"kmtu_cice_NEMS_mx{resname}.nc"
        cmd = ["ncks", "-O", "-v", "kmt", str(fs), str(fd)]
        result, msgs = run_cmd(cmd, log_file=log_file)
        if result != 0:
            log.error(msgs)
            raise RuntimeError("Failed to generate kmt file")

    log.info(f"Grid generation completed for: {resname}")
    log.info(f"Outputs staged in: {outdir}")

    # coupled_grid = state.get("coupled_grid", None)
    # if coupled_grid:

    #     write_grid_nml_inputs = get_func_signature(write_grid_nml)
    #     write_grid_nml_inputs = {
    #         k: v
    #         if k in write_grid_nml_inputs and v is not None
    #     }

    #     grid_nml_path = write_grid_nml(
    #         **write_grid_nml_inputs,
    #     )

    #     grid_gen_inputs = get_func_signature(run_gridgen)
    #     grid_gen_inputs = {
    #         k: v for k, v in state.items() if k in grid_gen_inputs and v is not None
    #     }

    #     print(f"Using grid_nml file at: {grid_nml_path}")

    #     run_gridgen(
    #         **grid_gen_inputs,
    #     )
