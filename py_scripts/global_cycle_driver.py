import os
from pathlib import Path

from fv3gfs_runtime import log
from fv3gfs_state import state
from fv3gfs_utils import cp
from global_cycle import run_global_cycle


def drive_global_cycle(
    datetime: str,
    res: int,
    fix: Path,  # fixed files directory
    tmp: Path,  # root temporary directory
    tmp_ic_dir: Path,  # Temporary directory for initial conditions
    n_nests: int = 0,
    n_tiles: int = 6,
    **kwargs,
):
    """
    Python driver for global_cycle.
    Reproduces FV3GFS surface update behavior, including symbolic linking
    of restart, grid, and orography files into the working directory.
    """

    # Resolution formatting
    c_res = f"C{res}"
    fix_am = fix / "am"

    # Default: global-only
    nest_range = range(n_nests + 1) if n_nests > 0 else [0]

    for nest_idx in nest_range:
        # Create working directory
        tmp_dir = (
            tmp / f"global_cycle_nest{nest_idx:02d}"
            if nest_idx > 0
            else tmp / "global_cycle"
        )
        tmp_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"PREPARING RUN DIRECTORY: {tmp_dir}")

        # ------------------------------------------------------------------------------
        # 1. Link restart and static files (same as Bash loop)
        # ------------------------------------------------------------------------------

        for n in range(1, n_tiles + 1):
            tile = f"tile{n}.nc"

            # Input and output surface files
            sfc_in = state.input / f"sfc_data.{tile}"
            sfc_out = state.input / f"sfcanl_data.{tile}"

            # Copy input -> output (like Bash does before linking)
            cp(sfc_in, sfc_out)
            os.chmod(sfc_out, 0o644)

            # Link using legacy names (the executable expects these)
            (tmp_dir / f"fnbgsi.{n:03d}").symlink_to(sfc_in)
            (tmp_dir / f"fnbgso.{n:03d}").symlink_to(sfc_out)

            # Grid and orography from fix_fv3gfs (here assumed under ic_dir)
            grid_file = state.input / f"{c_res}_grid.{tile}"
            orog_file = state.input / f"oro_data.{tile}"

            (tmp_dir / f"fngrid.{n:03d}").symlink_to(grid_file)
            (tmp_dir / f"fnorog.{n:03d}").symlink_to(orog_file)

            # Optional snow increment
            if kwargs.get("do_sno_inc", False):
                xainc = state.input / f"xainc.{tile}"
                if xainc.exists():
                    (tmp_dir / f"xainc.{n:03d}").symlink_to(xainc)

        # ------------------------------------------------------------------------------
        # 2. Call Fortran executable wrapper
        # ------------------------------------------------------------------------------

        run_global_cycle(
            datetime=datetime,
            c_res=c_res,
            tmp_dir=tmp_dir,
            fix_am=fix_am,
            tmp_ic_dir=tmp_ic_dir,
            exec_dir=state.ufs_exe,
            n_nests=n_nests,
            nest_idx=nest_idx,
            **kwargs,
        )

        log.info(f"Completed global_cycle for nest: {nest_idx}")

    log.info("Global cycle finished successfully for all nests.")

    # global_cycle_inputs = get_func_signature(drive_global_cycle)
    # global_cycle_inputs = {
    #     k: v for k, v in state.items() if k in global_cycle_inputs and v is not None
    # }

    # drive_global_cycle(
    #     **global_cycle_inputs,
    # )
