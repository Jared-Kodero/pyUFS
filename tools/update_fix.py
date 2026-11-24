#!/usr/bin/env python3
"""Mirror the NOAA fix datasets (am, orog, sfc_climo) and the Cartopy data into fix/.

    python tools/update_fix.py [--dest /path/to/fix] [--skip-carto]

The newest version directory of each dataset is copied to fix/<dataset>; the raw download
is kept in .fix_raw next to fix/ until the copy is done. Pre-generated
grid directories are removed, and the CO2 and volcanic aerosol files in fix/am are linked
without their global_ prefix.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pyufs_links import link_parent_dir

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

        new = dest.with_name(f".{d}.new")
        shutil.rmtree(new, ignore_errors=True)
        shutil.copytree(src, new)
        shutil.rmtree(dest, ignore_errors=True)
        new.rename(dest)


def remove_pre_generated(fix_root: Path) -> None:
    for path in fix_root.rglob("*"):
        if path.is_dir() and path.name in PRE_GENERATED:
            shutil.rmtree(path)


def recreate_symlink(src: Path, dest: Path) -> None:
    """Make `dest` a relative link to `src` when `src` is a .txt file."""
    if dest.is_dir() and not dest.is_symlink():
        return
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if src.suffix == ".txt":
        dest.symlink_to(os.path.relpath(src, start=dest.parent))


def recreate_symlinks(fix_root: Path) -> None:
    fixed_am = fix_root / "am"

    for f in fixed_am.rglob("global_*"):
        if not f.is_file():
            continue
        if not f.name.startswith(SYMLINK_TARGETS):
            continue

        dest = fixed_am / f.name.replace("global_", "", 1)
        recreate_symlink(f, dest)


# =============================
# Main
# =============================


def download_cartopy_data(output_dir: Path) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "cartopy.feature.download",
        "physical",
        "cultural",
        "cultural-extra",
        "gshhs",
        "--output",
        str(output_dir),
    ]

    print(f"Downloading Cartopy data to: {output_dir}")
    subprocess.run(command, check=True, cwd=output_dir)
    print(f"\nDownload complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize NOAA fixed datasets.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: fix in the parent of pyUFS, linked as pyUFS/fix).",
    )
    parser.add_argument(
        "--skip-carto", action="store_true", help="Skip the Cartopy data download."
    )
    args = parser.parse_args()

    print("=== Starting SHiELD FIX directory update ===")

    fix_dir = (
        args.dest.expanduser().resolve()
        if args.dest
        else link_parent_dir("fix", create=True)
    )
    fix_raw = fix_dir.with_name(".fix_raw")  # raw download, next to fix/ and not in the repo

    fix_raw.mkdir(exist_ok=True)
    fix_dir.mkdir(parents=True, exist_ok=True)

    sync_fix_dirs(fix_raw)
    copy_latest_versions(fix_raw, fix_dir)
    shutil.rmtree(fix_raw)

    remove_pre_generated(fix_dir)
    recreate_symlinks(fix_dir)

    print("=== FIX update complete ===")

    if not args.skip_carto:
        download_cartopy_data(fix_dir / "external" / "CARTOPY")


if __name__ == "__main__":
    main()
