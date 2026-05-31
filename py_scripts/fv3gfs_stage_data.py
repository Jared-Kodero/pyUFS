from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

from fv3gfs_pes_config import calc_cpu_alloc
from fv3gfs_runtime import log, sort_paths
from fv3gfs_state import compute_checksum, merge_saved_state, save_state, state
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

    if not state.cached_ic:
        return False

    save_state()
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

    _keys_to_remove = (
        "n_cpus",
        "checksum",
        "n_nodes",
        "node_list",
        "n_cpus_per_node",
        "multi_node",
        "total_pes",
        "global_pes",
        "grid_pes",
        "case_name",
        "case_description",
        "description",
        "restart_no",
        "resubmit",
        "total_restarts",
        "total_run_hours",
        "run_nhours",
        "continue_run",
        "warm_start",
        "ensemble_id",
        "n_ensembles",
        "ensemble_run",
        "paired_ensembles",
        "archive_data",
        "cached_ic",
        "ic_gen",
        "ic_only",
        "sm_perturbations",
        "update_nml_only",
        "fv3_debug",
        "shield_exe",
    )

    cached_cfg = copy.deepcopy(state)

    for k in _keys_to_remove:
        cached_cfg.pop(k, None)

    state_yaml_cache = cache_subdir / checksum

    save_state(cached_cfg, path=state_yaml_cache)

    log.info("Cached IC files. Set `ic_cache: false` to disable IC caching.")


def cached_ic_files():
    if not state.cached_ic:
        return False

    cache_dir = state.scratch_dir / ".cache"
    checksum = compute_checksum(state)

    grid_dir = cache_dir / checksum / state.grid.name
    input_dir = cache_dir / checksum / state.input.name
    state_yaml_src = cache_dir / checksum / checksum
    if not grid_dir.exists() or not input_dir.exists() or not state_yaml_src.exists():
        return False

    for src, dest in [(grid_dir, state.grid), (input_dir, state.input)]:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)

    state_yaml_dest = state.home / "state.yaml"
    shutil.copy(state_yaml_src, state_yaml_dest)

    merge_saved_state()
    calc_cpu_alloc(state.input)
    save_state()

    log.info(
        "Loaded cached IC files. Set `ic_cache: false` in run_config.yaml to disable cached IC loading."
    )
    return True
