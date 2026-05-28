from __future__ import annotations

import os
import shutil
from pathlib import Path

from fv3gfs_runtime import log, sort_paths
from fv3gfs_state import compute_checksum, state
from fv3gfs_utils import cp, rename


def stage_files() -> None:
    log.info("Staging requred files and data")

    n_nests = state.n_nests

    # get all subdirs in chgres_cube tmp dir
    chgres_cube = state.tmp / "chgres_cube"
    subdirs = [d.name for d in chgres_cube.iterdir() if d.is_dir()]
    nest_tile_dirs = sorted(
        [d for d in subdirs if d.startswith("tile")], key=sort_paths
    )
    nest_indices = [f"{i:02d}" for i in range(2, len(nest_tile_dirs) + 2)]
    nest_dict = dict(zip(nest_tile_dirs, nest_indices))

    if n_nests > 0:
        if len(nest_tile_dirs) != n_nests:
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
    for tile_dir, nest_idx in nest_dict.items():
        nest_dir = chgres_cube / tile_dir
        nest_files = nest_dir.glob("*.nc")
        for f in nest_files:
            if "tile" in f.name and "mosaic" not in f.name:
                kind = "atm" if "atm" in f.name else "sfc"
                name = "gfs" if kind == "atm" else "sfc"
                dest = state.input / f"{name}_data.nest{nest_idx}.{tile_dir}.nc"
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

    # Rename Oro files in ic_dir
    for f in state.input.glob("*oro*.tile*.nc"):
        parent = Path(f.parent)
        tile_str = f.stem.split(".")[-1]
        if n_nests > 0 and tile_str in nest_tile_dirs:
            nest_idx = nest_dict[tile_str]
            new_file = parent / f"oro_data.nest{nest_idx}.{tile_str}.nc"
        else:
            new_file = parent / f"oro_data.{tile_str}.nc"

        if new_file.exists():
            continue
        rename(f, new_file)

    # for file in INPUT, if "grid" in file name,or mosaic in file name, move to GRID dir
    for f in state.input.glob("*"):
        if "grid" in f.name or "mosaic" in f.name:
            dest = state.grid / f.name
            shutil.move(str(f), str(dest))
            rel_target = os.path.relpath(dest, start=f.parent)
            f.symlink_to(rel_target)

    shutil.rmtree(state.tmp, ignore_errors=True)
    Path(state.tmp).mkdir(parents=True, exist_ok=True)

    cache_ic_files()


def cache_ic_files():
    # save GRID and INPUT files
    cache_dir = state.scratch_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    checksum = compute_checksum(state)
    cache_subdir = cache_dir / checksum
    cache_subdir.mkdir(parents=True, exist_ok=True)
    for dir in [state.grid, state.input]:
        dest = cache_subdir / dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(dir, dest)


def cached_ic_files():
    cache_dir = state.scratch_dir / ".cache"
    checksum = compute_checksum(state)
    cache_subdir = cache_dir / checksum

    if not cache_subdir.exists():
        return False

    for dir_name in ["grid", "input"]:
        src = cache_subdir / dir_name
        dest = getattr(state, dir_name)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)

    log.info("Loaded cached IC files for current configuration.")
    return True
