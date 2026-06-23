from __future__ import annotations

import os
import shutil
from pathlib import Path

from fv3_runtime import log, sort_paths
from fv3_state import state
from fv3_utils import cp, rename


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

    # Process global 1st
    global_dir = chgres_cube / "global"
    global_files = global_dir.glob("*.nc")
    for f in global_files:
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

    fix_sfc_files = (state.tmp / "ic" / "fix_sfc").glob("*")
    for f in fix_sfc_files:
        f = Path(f)
        if f.is_symlink() and f.name.startswith("."):
            f.unlink()

    tmp_ic_dir_files = (state.tmp / "ic").glob("*")
    for f in tmp_ic_dir_files:
        dest_file = state.input / f.name

        if dest_file.exists():
            if dest_file.is_file():
                dest_file.unlink()
            elif dest_file.is_symlink():
                dest_file.unlink()
            elif dest_file.is_dir():
                shutil.rmtree(dest_file)

        cp(f, state.input)

    # rename INPUT/fix_sfc to state.fixed/fix_sfc
    fix_sfc_dest = state.fixed / "fix_sfc"
    fix_sfc_src = state.input / "fix_sfc"
    shutil.rmtree(fix_sfc_dest, ignore_errors=True)
    fix_sfc_src.rename(fix_sfc_dest)

    # Rename global orography files
    for f in state.input.glob("*oro*.tile*.nc"):
        tile_str = f.stem.split(".")[-1]

        # Global domain owns tiles 1 through 6.
        if tile_str in {f"tile{i}" for i in range(1, 7)}:
            new_file = state.input / f"oro_data.{tile_str}.nc"
            rename(f, new_file)

    # Rename nested orography files
    for nest_dir, nest_idx in nest_dict.items():
        tile = int(nest_idx) + 5

        for f in state.input.glob(f"*oro*.tile{tile}.nc"):
            new_file = state.input / f"oro_data.nest{nest_idx}.tile{tile}.nc"
            rename(f, new_file)

    # for file in INPUT, if "grid" in file name,or mosaic in file name, move to GRID dir
    for f in state.input.glob("*"):
        if "grid" in f.name or "mosaic" in f.name:
            dest = state.grid / f.name
            shutil.move(str(f), str(dest))
            rel_target = os.path.relpath(dest, start=f.parent)
            f.symlink_to(rel_target)

    os.system(f"rm -rf {state.tmp}/*")
