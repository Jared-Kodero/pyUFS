"""Finish a case after the regrid of its last segment: archive and run summary.

Run by launchers/finalize once per segment; every segment but the last returns
at once. Everything it needs is in state.yaml, which the regrid step has just
saved, plus a few variables of the job environment:

    CASE_RUN_START_TIME   epoch seconds of the first job of the case
    CASE_DATA_SYMLINK     ``run`` or ``memNN`` link, pointed at the archive
    CASE_LOG_FILE         log of the case, copied into the archive last
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import time
from pathlib import Path

from fv3_state import load_fv3_state, setup_logging

log = logging.getLogger("FINALIZE")


def point_link(link: Path, target: Path) -> None:
    """Point ``link`` at ``target``, replacing an existing link (ln -sfn)."""
    tmp = link.with_name(f"{link.name}.tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    os.replace(tmp, link)


def archive_case(case_dir: Path, archive_dir: Path, submit_dir: Path) -> None:
    """Copy the results of a finished case to ``archive_dir``.

    Only the final restart, state.yaml and run_config.yaml are kept in
    restart.tar.gz; the grid, orography and ICs are rebuilt from them
    (case_continue.sh --restart-dir).
    """
    archive_dir.mkdir(parents=True, exist_ok=True)

    for nc_file in (case_dir / "OUTPUT").glob("*.nc"):
        shutil.copy2(nc_file, archive_dir)
    shutil.copy2(case_dir / "state.yaml", archive_dir)

    shield_log = archive_dir / "shield_log"
    shield_log.mkdir(exist_ok=True)
    if (case_dir / "LOGS" / "shield").is_dir():
        shutil.copytree(case_dir / "LOGS" / "shield", shield_log, dirs_exist_ok=True)

    shutil.copy2(submit_dir / "run_config.yaml", case_dir)
    # Level 1: restart files are netCDF, which compress little for the time spent.
    with tarfile.open(archive_dir / "restart.tar.gz", "w:gz", compresslevel=1) as tar:
        for name in ("RESTART", "state.yaml", "run_config.yaml"):
            tar.add(case_dir / name, arcname=name)


def finalize_run() -> None:
    state = load_fv3_state()

    if state.restart_no != state.total_restarts - 1:
        return

    case_dir = Path(state.case_dir)
    archive = bool(state.archive_data)
    archive_dir = Path(state.archive_dir) if state.archive_dir else None

    if archive:
        archive_case(case_dir, archive_dir, Path(state.submit_dir))
        point_link(Path(os.environ["CASE_DATA_SYMLINK"]), archive_dir)

    # Segment times are recorded by the regrid step, so this excludes the archive.
    walltime = sum(state.segment_walltimes.values())
    taken = (time.time() - int(os.environ["CASE_RUN_START_TIME"])) / 3600

    log.info("Total Walltime: %.2f hours.", walltime)
    log.info("Total Time Taken: %.2f hours.", taken)
    if archive:
        log.info("Archived files to: %s", archive_dir)
    log.info("Case %s completed", os.getenv("SLURM_JOB_NAME") or state.case_name)


if __name__ == "__main__":
    setup_logging()
    finalize_run()
