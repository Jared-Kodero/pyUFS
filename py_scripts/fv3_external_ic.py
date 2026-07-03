from __future__ import annotations

import shutil
from pathlib import Path

from fv3_pes_config import calc_cpu_alloc
from fv3_runtime import log
from fv3_state import load_fv3_state, state


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
      uniform, stretch, nest   -> global cubed-sphere tiles 1..6
      regional_gfdl, esg       -> single tile 7
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
    res: int, gtype: str, n_nests: int, grid_dir: Path, input_dir: Path
) -> tuple[list[Path], list[Path], int]:
    """
    Build the minimal file set the model reads at cold start.

    Returns:
      grid_required   : exact grid-tile paths in GRID/
      input_required  : exact IC paths in INPUT/ (gfs_ctrl, gfs_data,
                        sfc_data, oro_data per tile)
      min_mosaics     : minimum number of C{res}_*mosaic*.nc files expected
                        in GRID/ (mosaic filenames vary by gtype, so these
                        are matched by pattern rather than by exact name)
    """
    global_tiles, nest_tiles = _expected_tiles(gtype, n_nests)

    grid_required = [
        grid_dir / f"C{res}_grid.tile{t}.nc" for t in (global_tiles + nest_tiles)
    ]

    input_required = [input_dir / "gfs_ctrl.nc"]
    for t in global_tiles:
        input_required += [
            input_dir / f"gfs_data.tile{t}.nc",
            input_dir / f"sfc_data.tile{t}.nc",
            input_dir / f"oro_data.tile{t}.nc",
        ]
    for t in nest_tiles:
        idx = t - 5  # nest index convention: tile 7 -> nest2
        input_required += [
            input_dir / f"gfs_data.nest{idx}.tile{t}.nc",
            input_dir / f"sfc_data.nest{idx}.tile{t}.nc",
            input_dir / f"oro_data.nest{idx}.tile{t}.nc",
        ]

    min_mosaics = 1 + (int(n_nests) if gtype == "nest" else 0)
    return grid_required, input_required, min_mosaics


def _validate_ic_files(res: int, gtype: str, n_nests: int) -> None:
    """
    Verify the grid and initial-condition files required for the model to
    start. Raises FileNotFoundError naming every missing or empty file,
    grouped by directory. Fixed climatology files are not checked here:
    update_fixed_files() stages them from fixed_dir downstream and raises
    if any are absent.
    """
    grid_dir = Path(state.grid)
    input_dir = Path(state.input)

    grid_required, input_required, min_mosaics = _ic_manifest(
        res, gtype, n_nests, grid_dir, input_dir
    )

    failures: dict[str, list[str]] = {}

    grid_bad = [p.name for p in grid_required if not _resolved_ok(p)]
    valid_mosaics = [m for m in grid_dir.glob(f"C{res}_*mosaic*.nc") if _resolved_ok(m)]
    if len(valid_mosaics) < min_mosaics:
        grid_bad.append(f"C{res}_*mosaic*.nc[{len(valid_mosaics)}/{min_mosaics}]")
    if grid_bad:
        failures["GRID"] = grid_bad

    input_bad = [p.name for p in input_required if not _resolved_ok(p)]
    if input_bad:
        failures["INPUT"] = input_bad

    if failures:
        detail = "; ".join(f"{d}: {', '.join(n)}" for d, n in failures.items())
        raise FileNotFoundError(f"IC validation failed in {state.case_home}: {detail}")


def _check_sfc_fix_provenance() -> None:
    """
    Advisory check for the target-grid surface climatology directory
    (FIXED/fix_sfc). It is a chgres_cube preprocessing input, not a model
    run-time input, so its absence is not fatal for a cold start from a
    pre-staged bundle. It is required only to regenerate surface ICs, and
    only while &namsfc supplies climatology as GRIB (FNxxx = FIXED/*.grb).
    """
    sfc_fix = Path(state.fixed) / "fix_sfc"
    if not sfc_fix.is_dir() or not any(sfc_fix.iterdir()):
        log.warning("FIXED/fix_sfc absent; needed only to regenerate surface ICs.")


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
    from res, gtype, and n_nests, rather than only confirming that the
    staging directories are non-empty.
    """
    external = bool(state.external_ic_dir)
    ic_dir = Path(state.external_ic_dir) if external else Path(state.case_home)
    case_home = Path(state.case_home)

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
    else:
        log.info(f"IC data found directly in {case_home}")

    # 3. Load the state descriptor. This drives the required-file manifest.
    if not (case_home / "state.yaml").exists():
        raise FileNotFoundError(f"Missing state.yaml in {case_home}")
    load_fv3_state(merge=True)

    # 4. Validate the exact files the model needs for this configuration.
    _validate_ic_files(res=state.res, gtype=state.gtype, n_nests=state.n_nests)

    # 5. Compute the CPU allocation from the validated INPUT directory.
    calc_cpu_alloc(state.input)

    _check_sfc_fix_provenance()

    return True
