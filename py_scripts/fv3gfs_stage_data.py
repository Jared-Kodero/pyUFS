from __future__ import annotations

import os
import shutil
from pathlib import Path

from fv3gfs_runtime import log, sort_paths
from fv3gfs_state import state
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
        if Path(f).is_symlink():
            Path(f).unlink()

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

    update_table_files()
    update_fixed_files()


def update_fixed_files():
    dt = state.init_datetime
    year = dt.year
    fix_dirs = [state.fix_am, state.fix / "lut"]

    required_files = [
        "aerosol.dat",
        f"co2historicaldata_{year}.txt",
        "co2historicaldata_glob.txt",
        "co2monthlycyc.txt",
        "sfc_emissivity_idx.txt",
        "solarconstant_noaa_an.txt",
        "volcanic_aerosols_1990-1999.txt",
        "global_h2oprdlos.f77",
        "global_o3prdlos.f77",
    ]

    missing_files = []

    for name in required_files:
        found = None
        for fix_dir in fix_dirs:
            candidate = fix_dir / name
            if candidate.exists():
                found = candidate
                break
            else:
                # Attempt fuzzy match if file not found
                matches = list(fix_dir.glob(f"*{name}"))
                if matches:
                    found = matches[0]
                    break

        if found:
            dest = state.fixed / name
            if not dest.exists():
                cp(found, dest)
            link = Path(state.input) / name
            link.unlink(missing_ok=True)
            rel_target = os.path.relpath(dest, start=state.input)
            link.symlink_to(rel_target)

        else:
            missing_files.append(name)

    if missing_files:
        log.warning("Missing required files:")
        for f in missing_files:
            log.warning(f"   - {f}")
        raise FileNotFoundError(
            "One or more required fixed files are missing. See log for details."
        )


def update_table_files():

    dt = state.init_datetime
    update_fixed_files()

    restart_no = state.get("restart_no", 0)

    diag_table_path = state.home / "diag_table"
    field_table_path = state.home / "field_table.yaml"

    user_diag = state.rundir / "diag_table"
    user_field = state.rundir / "field_table"

    template_diag = state.configs / "diag_table"
    template_field = state.configs / "field_table.yaml"

    if user_diag.exists():
        diag_file = user_diag
    else:
        diag_file = template_diag

    if user_field.exists():
        field_file = user_field
    else:
        field_file = template_field

    cp(diag_file, diag_table_path)
    cp(field_file, field_table_path)

    with open(diag_table_path) as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.strip().startswith("#")]

    dt_str = f"{dt.year} {dt.month:02d} {dt.day:02d} {dt.hour:02d} 0 0\n"
    desc_str = f"{state.description}\n"
    lines = [desc_str, dt_str] + [
        line.replace("XX", f"{restart_no:02d}") for line in lines
    ]
    with open(diag_table_path, "w") as f:
        f.writelines(lines)
