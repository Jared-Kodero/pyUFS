#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path

# =============================
# Configuration
# =============================

FIX_DIRS = ["orog", "am", "sfc_climo"]  # add as needed

PRE_GENERATED = [
    "C1152",
    "C12",
    "C128",
    "C192",
    "C24",
    "C3072",
    "C3359",
    "C3445",
    "C384",
    "C48",
    "C768",
    "C96",
    "grid_spec",
]

SYMLINK_TARGETS = (
    "global_co2historicaldata",
    "global_co2monthlycyc",
    "global_volcanic_aerosols",
)

DRY_RUN = False


# =============================
# Utilities
# =============================


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def get_latest_version(base: Path) -> str:
    versions = sorted(
        int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()
    )
    if not versions:
        raise RuntimeError(f"No version directories found in {base}")
    return str(versions[-1])


def sync_fix_dirs(fix_raw: Path) -> None:
    for d in FIX_DIRS:
        print(f"Syncing {d}")
        run(
            [
                "aws",
                "s3",
                "sync",
                f"s3://noaa-nws-global-pds/fix/{d}",
                str(fix_raw / d),
                "--no-sign-request",
            ]
        )


def copy_latest_versions(fix_raw: Path, fix_dir: Path) -> None:
    for d in FIX_DIRS:
        base = fix_raw / d
        latest = get_latest_version(base)
        src = base / latest
        dest = fix_dir / d

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(src, dest)


def remove_pre_generated(fix_root: Path) -> None:
    for path in fix_root.rglob("*"):
        if path.is_dir() and path.name in PRE_GENERATED:
            if not DRY_RUN:
                shutil.rmtree(path)


def safe_recreate_symlink(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        if dest.is_file() or dest.is_symlink():
            if not DRY_RUN:
                dest.unlink()
        else:
            return

    if src.suffix == ".txt":
        rel_target = os.path.relpath(src, start=dest.parent)
        if not DRY_RUN:
            dest.symlink_to(rel_target)


def recreate_symlinks(fix_root: Path) -> None:
    fix_am = fix_root / "am"

    for f in fix_am.rglob("global_*"):
        if not f.is_file():
            continue
        if not f.name.startswith(SYMLINK_TARGETS):
            continue

        dest = fix_am / f.name.replace("global_", "", 1)
        safe_recreate_symlink(f, dest)


# =============================
# Main
# =============================


def main():
    print("=== Starting SHiELD FIX directory update ===")

    pwd = Path.cwd()
    fix_raw = pwd / ".fix_raw"
    fix_dir = pwd / "fix"

    fix_raw.mkdir(exist_ok=True)
    fix_dir.mkdir(exist_ok=True)

    sync_fix_dirs(fix_raw)
    copy_latest_versions(fix_raw, fix_dir)
    shutil.rmtree(fix_raw)

    remove_pre_generated(fix_dir)
    recreate_symlinks(fix_dir)

    # delete .fix_raw if it still exists
    if fix_raw.exists():
        shutil.rmtree(fix_raw)

    print("=== FIX update complete ===")


if __name__ == "__main__":
    main()
