#!/usr/bin/env python3

"""Resume a failed SHiELD case from its last good restart.

Usage
-----
Fix the cause of the failure, then run ``case_continue.sh`` from the case
directory, the directory that holds the case-local ``run_config.yaml``. Nothing
is cleared: ``RESTART``, ``HIST``, ``OUTPUT`` and ``IC`` are left as they are.

The segment to run is read from the model time of the newest restart on disk:

* ``RESTART`` holds files: the previous segment finished and its promotion to
  ``INPUT`` is still pending; the model time in ``RESTART/coupler.res`` is the
  start of the segment to run.
* ``RESTART`` is empty: the failed segment already promoted its restart; the
  model time in ``INPUT/coupler.res`` is the start of the segment to run.

That time must be the start of one of the ``resubmit + 1`` segments recorded in
``state.yaml``. The job is submitted with ``CASE_RESUBMIT_INDEX`` set to that
segment and ``CASE_CONTINUE_RUN=1``; ``case_run.sh`` resubmits the later
segments as usual.

Restart directory
-----------------
``--restart-dir DIR`` starts a new case from the final restart of an archived
one: ``DIR`` is a directory holding the unpacked ``restart.tar.gz``:
``RESTART/`` (``coupler.res`` and the other restart files) with ``state.yaml``
and ``run_config.yaml`` beside it. The grid, orography and initial conditions are rebuilt
from ``run_config.yaml``, which must keep the grid and start-time settings of
the archived case (checked against ``DIR/state.yaml``). The
segment is the one that starts at the model time of ``DIR/RESTART/coupler.res``, and
raising ``resubmit`` in ``run_config.yaml`` extends the run. With
``--ensembles`` or ``--mem`` the restart of member N is read from
``DIR/memNN``.

Ensembles
---------
Each member is resumed on its own, from its own ``<case_name>/memNN`` directory
and restart, so members that stopped at different segments each continue from
their own last restart::

    case_continue.sh                  single case (an error if n_ensembles > 0)
    case_continue.sh --ensembles      every member except those in skip_ensembles
    case_continue.sh --mem 2          member 2 only
    case_continue.sh --mem 2 5        members 2 and 5

A member with no case directory (not started, or already archived) or whose
last restart is the final segment is reported and skipped; the others are still
submitted. The exit status is 1 if any member failed its checks or nothing was
submitted.
"""

import argparse
import calendar
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from case_submit import SCRIPT_DIR, get_config, log_file_path, logger, run, yaml


class ContinueError(Exception):
    """The case cannot be continued; the message says why."""


class ContinueSkip(Exception):
    """Nothing to continue for this case (absent or already complete)."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="case_continue.sh",
        description="Resume a failed case, or members of an ensemble, from the last good restart.",
    )
    parser.add_argument(
        "--restart-dir",
        type=Path,
        metavar="DIR",
        help="start from the final restart of an archived case unpacked in DIR",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ensembles",
        action="store_true",
        help="resume every ensemble member (except those in skip_ensembles)",
    )
    group.add_argument(
        "--mem",
        type=int,
        nargs="+",
        metavar="N",
        help="resume only these ensemble members (numbered from 1)",
    )
    return parser.parse_args(argv)


def segment_edges(state: dict, resubmit: int) -> list:
    """Start time of every segment 0..resubmit+1 (the last is the end of the run)."""
    start = datetime.strptime(str(state["init_datetime"]), "%Y%m%d%HZ")
    length = int(state["run_length"])
    units = state["run_length_units"]
    edges = []
    for k in range(resubmit + 2):
        if units in ("months", "years"):
            months = length * k * (12 if units == "years" else 1)
            year, month = divmod(start.month - 1 + months, 12)
            year, month = start.year + year, month + 1
            day = min(start.day, calendar.monthrange(year, month)[1])
            edges.append(start.replace(year=year, month=month, day=day))
        else:
            edges.append(start + timedelta(**{units: length * k}))
    return edges


def model_time(coupler: Path) -> datetime:
    """Current model time of a restart, from the third line of ``coupler.res``."""
    if not coupler.exists():
        raise ContinueError(
            f"{coupler} not found; no complete restart to continue from."
        )
    # "Current model time: year, month, day, hour, minute, second".
    return datetime(*map(int, coupler.read_text().splitlines()[2].split()[:6]))


def find_segment(state_file: Path, coupler: Path, resubmit: int = None):
    """Return ``(segment, model_time)`` of the segment starting at ``coupler``'s time.

    ``resubmit`` defaults to the value in ``state_file``.
    """
    with open(state_file) as f:
        state = yaml.safe_load(f)
    resubmit = int(state["resubmit"]) if resubmit is None else resubmit

    edges = segment_edges(state, resubmit)
    current = model_time(coupler)
    if current not in edges:
        raise ContinueError(
            f"{coupler} is at {current}, which is not the start of a segment."
        )
    segment = edges.index(current)
    if segment == 0:
        raise ContinueError("The case has no restart yet; use case_submit.sh.")
    if segment > resubmit:
        raise ContinueSkip(f"The case is complete: the last restart is at {current}.")
    return segment, current


def job_names(env: dict, case_pwd: Path, member: int = None) -> dict:
    """Case name, case directory, job name, symlink, log file and ensemble id."""
    case_name = env["CASE_NAME"]
    job_name = f"{case_pwd.parent.name}.{case_pwd.name}"
    symlink = case_pwd / "run"
    log_file = log_file_path(env["CASE_OUTPUT"])
    if member is not None:
        mem_id = f"{member:02d}"  # zero padded, as in case_submit.py
        case_name = f"{case_name}/mem{mem_id}"
        job_name = f"{job_name}.MEM{mem_id}"
        symlink = case_pwd / f"mem{mem_id}"
        log_file = log_file_path(env["CASE_OUTPUT"], mem_id)
    return {
        "case_name": case_name,
        "case_dir": Path(env["CASE_ROOT_DIR"]) / case_pwd.parent.name / case_name,
        "job_name": job_name,
        "symlink": symlink,
        "log_file": log_file,
        "ensemble_id": member or 0,
    }


def local_restart(case_dir: Path) -> tuple:
    """``(state_file, coupler)`` of a case that is continued in place.

    ``RESTART`` holds files while the promotion of the last restart to
    ``INPUT`` is pending, and is empty once it has been promoted.
    """
    state_file = case_dir / "state.yaml"
    if not state_file.exists():
        raise ContinueSkip(
            f"{state_file} not found; there is no case to continue (not started, or already archived)."
        )
    restart = case_dir / "RESTART"
    pending = restart.is_dir() and any(restart.iterdir())
    return state_file, (restart if pending else case_dir / "INPUT") / "coupler.res"


def archived_restart(case_dir: Path, restart_dir: Path, member: int = None) -> tuple:
    """``(state_file, coupler)`` of an unpacked archived restart."""
    if (case_dir / "state.yaml").exists():
        raise ContinueError(
            f"{case_dir} already holds a case; run case_continue.sh without --restart-dir."
        )
    if member is not None:
        restart_dir = restart_dir / f"mem{member:02d}"
    state_file = restart_dir / "state.yaml"
    if not state_file.exists():
        raise ContinueError(f"{state_file} not found.")
    return state_file, restart_dir / "RESTART" / "coupler.res"


def resume(
    env: dict, case_pwd: Path, member: int = None, restart_dir: Path = None
) -> None:
    """Submit the resumed segment of the single case or of member ``member``."""
    names = job_names(env, case_pwd, member)

    if restart_dir is None:  # continue the case in place
        state_file, coupler = local_restart(names["case_dir"])
        segment, current = find_segment(state_file, coupler)
        restart_env = {}
    else:  # start the case from an archived restart
        state_file, coupler = archived_restart(names["case_dir"], restart_dir, member)
        segment, current = find_segment(
            state_file, coupler, int(env["CASE_RESUBMIT_MAX"])
        )
        restart_env = {
            "CASE_RESTART_DIR": state_file.parent,
            "CASE_RESTART_STATE": state_file,
        }

    iter_env = {
        **env,
        "CASE_NAME": names["case_name"],
        "TOTAL_WALLTIME_TIME": 0,
        "CASE_ENSEMBLE_ID": names["ensemble_id"],
        "CASE_RESUBMIT_INDEX": segment,
        "CASE_CONTINUE_RUN": int(restart_dir is None),
        **restart_env,
        "SLURM_JOB_NAME": names["job_name"],
        "SLURM_OPEN_MODE": "append",
        "CASE_DATA_SYMLINK": str(names["symlink"]),
        "CASE_LOG_FILE": str(names["log_file"]),
    }
    proc_env = {**os.environ, **{k: str(v) for k, v in iter_env.items()}}
    run(Path(env["UFS_UTILS_DIR"]) / "launchers" / "sbatch.sh", proc_env, case_pwd)
    logger.info(
        f"Case {names['case_name']} resubmitted from segment {segment} ({current})"
    )


def select_members(args, env: dict, skipped: list) -> list:
    """Ensemble members to resume (``[None]`` for a single case)."""
    n_ensembles = int(env["CASE_ENSEMBLES"])
    ensemble_requested = args.ensembles or args.mem is not None

    if ensemble_requested and n_ensembles < 1:
        logger.error(
            "--ensembles and --mem need ensemble_run: true and n_ensembles >= 1."
        )
        sys.exit(1)
    if not ensemble_requested and n_ensembles > 0:
        logger.error(
            "This is an ensemble case: use --ensembles for all members or --mem N for one."
        )
        sys.exit(1)

    if not ensemble_requested:
        return [None]
    if args.mem is not None:
        bad = sorted({m for m in args.mem if not 1 <= m <= n_ensembles})
        if bad:
            logger.error(f"Members {bad} are outside 1..{n_ensembles}.")
            sys.exit(1)
        return sorted(set(args.mem))
    return [m for m in range(1, n_ensembles + 1) if m not in skipped]


def main():
    args = parse_args()
    env = get_config()

    case_pwd = Path.cwd()
    env["CASE_PWD"] = str(case_pwd)
    env["CASE_DIR"] = case_pwd.name
    env["CASE_PARENT_DIR"] = case_pwd.parent.name
    env["UFS_UTILS_DIR"] = str(SCRIPT_DIR.parent.parent)
    env["CASE_NAME"] = env["CASE_NAME"] or case_pwd.name
    skipped = env.pop("CASE_SKIP_ENSEMBLES")

    members = select_members(args, env, skipped)
    restart_dir = args.restart_dir.resolve() if args.restart_dir else None

    submitted = failed = 0
    for member in members:
        label = "case" if member is None else f"member {member}"
        try:
            resume(env, case_pwd, member, restart_dir)
            submitted += 1
        except ContinueSkip as exc:
            logger.warning(f"Skipped {label}: {exc}")
        except ContinueError as exc:
            logger.error(f"{label}: {exc}")
            failed += 1

    if failed or not submitted:
        sys.exit(1)


if __name__ == "__main__":
    main()
