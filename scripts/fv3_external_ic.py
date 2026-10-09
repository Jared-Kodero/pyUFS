from __future__ import annotations

import shutil
from pathlib import Path

from fv3_runtime import log
from fv3_state import load_fv3_state, state
from regional_bc import bc_forecast_hours

# Settings that describe the staged grid and initial conditions. The bundle's
# state.yaml is authoritative for these; every other setting (description,
# segment length and count, output, forcing, resources) comes from the case.
BUNDLE_KEYS = frozenset(
    {
        "c_res",
        "gtype",
        "levels",
        "init_datetime",
        "forecast_hour",
        "ic_cycle",
        "target_lon",
        "target_lat",
        "stretch_factor",
        "refine_ratio",
        "n_nests",
        "nest_type",
        "parent_tile",
        "halo",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "idim",
        "jdim",
        "delx",
        "dely",
        "res_km",
        "add_lake",
        "lake_cutoff",
        "make_gsl_orog",
        "istart_nest",
        "iend_nest",
        "jstart_nest",
        "jend_nest",
        "nest_ioffsets",
        "nest_joffsets",
        "ic_source",
    }
)


def _resolved_ok(path: Path) -> bool:
    """
    True if path exists after following symlinks and is a non-empty file
    or a non-empty directory. Broken symlinks and zero-byte files fail.
    """
    try:
        target = path.resolve()
    except OSError:
        return False
    if not target.exists():
        return False
    if target.is_dir():
        return any(target.iterdir())
    return target.stat().st_size > 0


def _expected_tiles(gtype: str, n_nests: int) -> tuple[list[int], list[int]]:
    """
    Return (global_tiles, nest_tiles) for the model configuration.

    Conventions verified against fv3_driver_grid.py and fv3_stage_data.py:
      uniform, stretch, nest        -> global cubed-sphere tiles 1..6
      regional_gfdl, regional_esg   -> single tile 7
      nest                     -> additional tiles 7..6+n_nests
    """
    if gtype in ("uniform", "stretch", "nest"):
        global_tiles = [1, 2, 3, 4, 5, 6]
    elif gtype in ("regional_gfdl", "regional_esg"):
        global_tiles = [7]
    else:
        raise ValueError(f"Unsupported gtype: {gtype!r}")

    nest_tiles = list(range(7, 7 + int(n_nests))) if gtype == "nest" else []
    return global_tiles, nest_tiles


def _ic_manifest(
    c_res: int, gtype: str, n_nests: int, grid_dir: Path, input_dir: Path
) -> tuple[list[Path], list[Path], int]:
    """
    Build the minimal file set the model reads at cold start.

    Returns:
      grid_required   : exact grid-tile paths in GRID/
      input_required  : exact IC paths in INPUT/ (gfs_ctrl, gfs_data,
                        sfc_data, oro_data per tile)
      min_mosaics     : minimum number of C{c_res}_*mosaic*.nc files expected
                        in GRID/ (mosaic filenames vary by gtype, so these
                        are matched by pattern rather than by exact name)
    """
    if gtype in ("regional_gfdl", "regional_esg"):
        # Names given by fv3_stage_data.stage_regional_inputs (halo 3).
        grid_required = [
            grid_dir / f"C{c_res}_grid.tile7.halo3.nc",
            grid_dir / f"C{c_res}_grid.tile7.halo4.nc",
        ]
        input_required = [
            input_dir / name
            for name in (
                "gfs_ctrl.nc",
                "gfs_data.nc",
                "sfc_data.nc",
                "oro_data.nc",
                "oro_data.tile7.halo4.nc",
                "grid_spec.nc",
            )
        ]
        return grid_required, input_required, 1

    global_tiles, nest_tiles = _expected_tiles(gtype, n_nests)

    grid_required = [
        grid_dir / f"C{c_res}_grid.tile{t}.nc" for t in (global_tiles + nest_tiles)
    ]

    input_required = [input_dir / "gfs_ctrl.nc"]
    for t in global_tiles:
        input_required += [
            input_dir / f"gfs_data.tile{t}.nc",
            input_dir / f"sfc_data.tile{t}.nc",
            input_dir / f"oro_data.tile{t}.nc",
        ]
    for t in nest_tiles:
        idx = t - 5  # nest index convention: tile 7 -> nest02
        input_required += [
            input_dir / f"gfs_data.nest{idx:02d}.tile{t}.nc",
            input_dir / f"sfc_data.nest{idx:02d}.tile{t}.nc",
            input_dir / f"oro_data.nest{idx:02d}.tile{t}.nc",
        ]

    min_mosaics = 1 + (int(n_nests) if gtype == "nest" else 0)
    return grid_required, input_required, min_mosaics


def _validate_ic_files(c_res: int, gtype: str, n_nests: int) -> None:
    """
    Verify the grid and initial-condition files required for the model to
    start. Raises FileNotFoundError naming every missing or empty file,
    grouped by directory. Fixed climatology files are not checked here:
    update_fixed_files() stages them from fix_src downstream and raises
    if any are absent.
    """
    grid_dir = Path(state.grid)
    input_dir = Path(state.input)

    grid_required, input_required, min_mosaics = _ic_manifest(
        c_res, gtype, n_nests, grid_dir, input_dir
    )

    failures: dict[str, list[str]] = {}

    grid_bad = [p.name for p in grid_required if not _resolved_ok(p)]
    valid_mosaics = [
        m for m in grid_dir.glob(f"C{c_res}_*mosaic*.nc") if _resolved_ok(m)
    ]
    if len(valid_mosaics) < min_mosaics:
        grid_bad.append(f"C{c_res}_*mosaic*.nc[{len(valid_mosaics)}/{min_mosaics}]")
    if grid_bad:
        failures["GRID"] = grid_bad

    input_bad = [p.name for p in input_required if not _resolved_ok(p)]
    if input_bad:
        failures["INPUT"] = input_bad

    if failures:
        detail = "; ".join(f"{d}: {', '.join(n)}" for d, n in failures.items())
        raise FileNotFoundError(f"IC validation failed in {state.work_dir}: {detail}")


def _validate_bc_files() -> None:
    """A regional bundle must hold a boundary file for every boundary hour of
    this case, whose run length may exceed the bundle's."""
    if state.gtype not in ("regional_gfdl", "regional_esg"):
        return
    bc_dir = Path(state.bc_data)
    missing = [
        h
        for h in bc_forecast_hours()
        if not _resolved_ok(bc_dir / f"gfs_bndy.tile7.{h:03d}.nc")
    ]
    if missing:
        raise FileNotFoundError(
            f"No boundary files {bc_dir}/gfs_bndy.tile7.HHH.nc for hours {missing} "
            + f"of this {state.total_run_hours} h run; regenerate the IC bundle "
            + "for the full run length"
        )


def _stage_external_bundle(src: Path, dst: Path) -> None:
    """
    Copy an external IC bundle into the case directory, preserving symlinks
    so that relative INPUT -> GRID links remain valid after the copy.
    """
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, symlinks=True, dirs_exist_ok=True)
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copy2(item, target, follow_symlinks=False)


def init_external_ic() -> bool:
    """
    Stage and validate a pre-generated grid and initial-condition bundle for
    a cold start, then load state and compute the CPU allocation.

    Validation is file-level and configuration-aware: it confirms the exact
    grid-tile and initial-condition files the model reads at start, derived
    from c_res, gtype, and n_nests, rather than only confirming that the
    staging directories are non-empty.
    """
    if state.work_dir is None:
        raise RuntimeError("Cannot stage external IC: work_dir is not set")

    case_home = Path(state.work_dir)
    external = bool(state.external_ic_dir)
    ic_dir = Path(state.external_ic_dir) if external else case_home

    # 1. Top-level staging directories must be present and non-empty.
    required_dirs = ("FIXED", "GRID", "INPUT")
    missing_dirs = [
        d
        for d in required_dirs
        if not (ic_dir / d).is_dir() or not any((ic_dir / d).iterdir())
    ]
    if missing_dirs:
        raise FileNotFoundError(
            f"Incomplete IC staging in {ic_dir}: {', '.join(missing_dirs)}"
        )

    # 2. Copy the bundle into the case directory when it comes from elsewhere.
    if external:
        _stage_external_bundle(ic_dir, case_home)
        log.info(f"Copied external IC data from {ic_dir} to {case_home}")
        # Archived INPUT directories of the bundle's own segments would collide
        # with the archive this case writes at its first restart.
        ic_archive = case_home / "IC"
        for archived in [ic_archive / "INPUT", *ic_archive.glob("R*_INPUT")]:
            if archived.is_dir() and not archived.is_symlink():
                shutil.rmtree(archived)
    else:
        log.info(f"IC data source: {case_home}")

    # 3. Load the state descriptor. This drives the required-file manifest.
    # The bundle supplies the grid and IC keys; the case keeps its own run
    # settings, so a bundle can be shared by experiment arms.
    if not (case_home / "state.yaml").exists():
        raise FileNotFoundError(f"Missing state.yaml in {case_home}")
    case = {k: v for k, v in state.items() if k not in BUNDLE_KEYS}
    expected = {k: state.get(k) for k in ("init_datetime", "gtype", "levels")}
    load_fv3_state()
    state.update(case)

    for key, value in expected.items():
        if value is not None and value != state.get(key):
            raise ValueError(
                f"Case {key}={value!r} differs from the IC bundle ({state.get(key)!r})"
            )

    # 4. Validate the exact files the model needs for this configuration.
    _validate_ic_files(c_res=state.c_res, gtype=state.gtype, n_nests=state.n_nests)
    _validate_bc_files()

    return True
