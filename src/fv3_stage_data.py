from __future__ import annotations

import os
import shutil
from pathlib import Path

from fv3_runtime import log, sort_paths
from fv3_state import state
from fv3_utils import cp, rename
from regional_bc import link_bc_to_input


def stage_files() -> None:

    log.info("Staging requred files and data")

    n_nests = state.n_nests

    # get all subdirs in chgres_cube tmp dir
    chgres_cube = state.tmp / "chgres_cube"
    subdirs = [d.name for d in chgres_cube.iterdir() if d.is_dir()]
    nest_tile_dirs = sorted(
        [d for d in subdirs if d.startswith("nest")], key=sort_paths
    )
    nest_indices = [str(Path(d).name.replace("nest", "")) for d in nest_tile_dirs]
    nest_dict = dict(zip(nest_tile_dirs, nest_indices))

    if n_nests > 0 and len(nest_tile_dirs) != n_nests:
        raise ValueError(
            f"Number of nest directories [{len(nest_tile_dirs)}] does not match n_nests [{n_nests}]."
        )

    # Process global and regional domains first. Both use tile-named output
    # (regional owns tile7), and chgres writes each domain to its own subdir
    # (chgres_cube/<domain>). Without the regional case, a regional run stages
    # no atm/sfc ICs at all.
    for domain in ("global", "regional"):
        domain_dir = chgres_cube / domain
        if not domain_dir.is_dir():
            continue
        for f in domain_dir.glob("*.nc"):
            if "tile" in f.name and "mosaic" not in f.name:
                tile_str = f.stem.split(".")[-1]  # e.g., tile1, tile7
                kind = "atm" if "atm" in f.name else "sfc"
                name = "gfs" if kind == "atm" else "sfc"
                dest = state.input / f"{name}_data.{tile_str}.nc"

            else:
                dest = state.input / f.name
            cp(f, dest)

    # Now process nests
    for nest_dir, nest_idx in nest_dict.items():
        nest_dir = chgres_cube / nest_dir
        nest_files = nest_dir.glob("*.nc")
        tile = int(nest_idx) + 5

        for f in nest_files:
            if "tile" in f.name and "mosaic" not in f.name:
                kind = "atm" if "atm" in f.name else "sfc"
                name = "gfs" if kind == "atm" else "sfc"
                dest = state.input / f"{name}_data.nest{nest_idx}.tile{tile}.nc"
            else:
                continue

            cp(f, dest)

    fix_sfc_files = list((state.tmp / "input" / "fix_sfc").glob("*"))
    for f in fix_sfc_files:
        f = Path(f)
        if f.is_symlink() and f.name.startswith("."):
            f.unlink()

    tmp_ic_dir_files = list((state.tmp / "input").glob("*"))
    for f in tmp_ic_dir_files:
        dest_file = state.input / f.name

        if dest_file.exists() and dest_file.is_file():
            dest_file.unlink()

        elif dest_file.is_dir():
            shutil.rmtree(dest_file)

        cp(f, state.input)

    # rename INPUT/fix_sfc to state.fix/fix_sfc
    fix_sfc_dest = state.fix / "fix_sfc"
    fix_sfc_src = state.input / "fix_sfc"
    shutil.rmtree(fix_sfc_dest, ignore_errors=True)
    fix_sfc_src.rename(fix_sfc_dest)

    # Rename global orography files.
    # Match only the raw make_orog/filter_topo output (oro.C{res}.tile{N}.nc).
    # A broad *oro* pattern also matches this loop's own output (oro_data.*)
    # and the GSL products (C{res}_oro_data_ss/ls.tile{N}.nc), which would
    # collide on the same destination and either crash or clobber the field.
    for f in list(state.input.glob("oro.C*.tile*.nc")):
        tile_str = f.stem.split(".")[-1]

        # Global domain owns tiles 1 through 6.
        if tile_str in {f"tile{i}" for i in range(1, 7)}:
            new_file = state.input / f"oro_data.{tile_str}.nc"
            rename(f, new_file)

    # Rename nested orography files
    for nest_dir, nest_idx in nest_dict.items():
        tile = int(nest_idx) + 5

        for f in list(state.input.glob(f"oro.C*.tile{tile}.nc")):
            new_file = state.input / f"oro_data.nest{nest_idx}.tile{tile}.nc"
            rename(f, new_file)

    for f in list(state.input.glob("*")):
        if "grid" in f.name or "mosaic" in f.name:
            dest = state.grid / f.name
            shutil.move(str(f), str(dest))
            rel_target = os.path.relpath(dest, start=f.parent)
            f.symlink_to(rel_target)

    if state.gtype in ("regional_gfdl", "regional_esg"):
        stage_regional_inputs()

    # Empty TMP, including hidden entries such as the parent grid that
    # regional_gfdl bracketing writes to TMP/.tmp_make_grid.
    for entry in Path(state.tmp).iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _link(link: Path, target: Path) -> None:
    link.unlink(missing_ok=True)
    link.symlink_to(os.path.relpath(target, start=link.parent))


def stage_regional_inputs() -> None:
    """Give the regional inputs the names the model reads.

    Follows SHiELD_build RTS/GAEA_RTS/Regional3km.csh: the model grid is the
    halo-3 tile read through INPUT/grid_spec.nc (fv_grid_nml grid_file),
    the boundary code reads grid.tile7.halo4.nc and oro_data.tile7.halo4.nc
    (fv_regional_bc.F90), and the single-tile initial conditions and
    orography carry no tile suffix (FMS2 adds one only for multi-tile
    domains or tile numbers above 1).
    """
    res, halo = state.c_res, state.halo
    inp, grid = Path(state.input), Path(state.grid)

    rename(inp / f"C{res}_oro_data.tile7.halo0.nc", inp / "oro_data.nc")
    rename(
        inp / f"C{res}_oro_data.tile7.halo{halo + 1}.nc",
        inp / f"oro_data.tile7.halo{halo + 1}.nc",
    )
    rename(inp / "gfs_data.tile7.nc", inp / "gfs_data.nc")
    rename(inp / "sfc_data.tile7.nc", inp / "sfc_data.nc")

    _link(inp / "grid_spec.nc", grid / f"C{res}_mosaic.nc")
    _link(inp / f"C{res}_grid.tile7.nc", grid / f"C{res}_grid.tile7.halo{halo}.nc")
    _link(
        inp / f"grid.tile7.halo{halo + 1}.nc",
        grid / f"C{res}_grid.tile7.halo{halo + 1}.nc",
    )

    # Link the regional boundary files (kept in state.bc_data) into INPUT. The
    # links target state.bc_data, outside state.tmp, so they survive the cleanup.
    link_bc_to_input()
