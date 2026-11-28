#!/usr/bin/env python3
"""Build all UFS_UTILS tools, natively on Oscar.

    python tools/build_ufs.py
    python tools/build_ufs.py -j 16

Build UFS_UTILS release UFS_UTILS_TAG. Load the host modules of
config/run_config.yaml plus the build-only modules below, build the
NCEPLIBS libraries and every UFS_UTILS tool in /tmp/$USER/ufs_utils-<tag>, and replace
bin/preprocessing with the result. The executables carry the library paths (RPATH) of the
build modules, so a run needs only the host modules of run_config.yaml. Set ufs_bin_dir in
run_config.yaml to the path that the script prints.

What the script does, in order:
 1. Load the host modules plus the build-only modules and check the toolchain.
 2. Build the NCEPLIBS libraries that UFS_UTILS needs (bacio, sp, ip, ...), each at a
    pinned tag, into a private prefix.
 3. Build UFS_UTILS against those libraries and the loaded modules.
 4. Check that every executable that pyUFS calls exists, and that each one finds all of
    its shared libraries with only the host modules loaded (the run-time situation).
 5. Write bin/preprocessing/manifest.yaml (UFS_UTILS release, module and library versions,
    source tags and commits) and replace bin/preprocessing with the new build.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# utils must be imported first: it puts core/ on sys.path for fv3_utils.
from pyufs_links import link_parent_dir
from utils import (
    BUILD_ROOT,
    BuildError,
    default_config,
    module_script,
    version_record,
)

from fv3_utils import dump_yaml

UFS_UTILS_URL = "https://github.com/ufs-community/UFS_UTILS.git"
# Newest release whose chgres_cube writes the sfc_data variables SHiELD reads (sheleg,
# snwdph, zorl). From 1.13.0 it writes the fractional-grid format, which SHiELD cannot read.
UFS_UTILS_TAG = "ufs_utils_1_12_0"

# Loaded after the `modules` of run_config.yaml (esmf-mpi needs hpcx-mpi first).
BUILD_MODULES = ("esmf-mpi", "jasper", "libpng", "openblas")

# NCEPLIBS libraries, built in this order: name -> git tag and extra CMake options.
# The tags are the versions in the ci/spack.yaml of UFS_UTILS_TAG, except g2 (see below).
LIBS = {
    "bacio": {"tag": "v2.4.1"},
    "sp": {"tag": "v2.5.0"},
    "ip": {"tag": "v4.4.0", "options": ["-DBUILD_4=OFF"]},
    "w3emc": {
        "tag": "v2.10.0",
        "options": ["-DBUILD_WITH_BUFR=OFF", "-DBUILD_WITH_EXTRA_DEPS=OFF"],
    },
    "nemsio": {"tag": "v2.5.4"},
    "sigio": {"tag": "v2.3.2"},
    "sfcio": {"tag": "v1.4.1"},
    # 3.4.5 calls jpc_decode, which Jasper 3 and later no longer export; 3.4.6 uses the public API.
    "g2": {"tag": "v3.4.6"},
}

# Executables that core/ calls; the build fails if any is missing.
EXECUTABLES = (
    "chgres_cube",
    "cpld_gridgen",
    "emcsfc_ice_blend",
    "emcsfc_snow2mdl",
    "global_cycle",
    "global_equiv_resol",
    "inland",
    "lakefrac",
    "make_hgrid",
    "make_solo_mosaic",
    "orog",
    "orog_gsl",
    "regional_esg_grid",
    "sfc_climo_gen",
    "shave",
)

PREFLIGHT = """for tool in gfortran gcc mpif90 mpicc cmake make git; do
  command -v "$tool" >/dev/null || { echo "MISSING $tool"; missing=1; }
done
[ -n "${NETCDF:-}" ] || { echo "MISSING NETCDF (set by the netcdf module)"; missing=1; }
[ -n "${ESMFMKFILE:-}" ] || { echo "MISSING ESMFMKFILE (set by the esmf module)"; missing=1; }
[ -z "$missing" ] || exit 1
echo "gfortran: $(gfortran -dumpfullversion)"
echo "mpif90: $(command -v mpif90)"
echo "cmake: $(cmake --version | head -1)"
echo "netcdf: $NETCDF"
echo "esmf: $ESMFMKFILE"
echo "loaded modules: ${LOADEDMODULES//:/ }"
"""


def bash(script: str, *, cwd: Path | None = None, live: bool = True) -> str:
    """Run a bash script; raise BuildError on failure. Returns its output if not live."""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        text=True,
        capture_output=not live,
        check=False,
    )
    if result.returncode:
        message = f"command failed ({result.returncode})"
        if not live:
            message += ":\n" + result.stdout + result.stderr
        raise BuildError(message + "; see the output above" * live)
    return "" if live else result.stdout


def clone(url: str, ref: str, dest: Path) -> str:
    """Shallow clone of a tag or branch; returns the commit."""
    cmd = ["git", "clone", "-q", "--depth", "1", "--branch", ref, url, str(dest)]
    bash(shlex.join(cmd))
    return bash(
        shlex.join(["git", "-C", str(dest), "rev-parse", "HEAD"]), live=False
    ).strip()


def link_dirs(env: str) -> str:
    """Library directories of the loaded build modules, as a CMake list.

    CMAKE_INSTALL_RPATH_USE_LINK_PATH adds only the directories of libraries that are
    linked by full path and that are not implicit link directories. Libraries that reach
    the link line as -l flags (libpioc and others through esmf.mk) or whose directories
    are in LIBRARY_PATH (Jasper, OpenBLAS) are missed, so the directories are passed
    explicitly. They come from LD_LIBRARY_PATH, LIBRARY_PATH and the -L paths of esmf.mk.
    """
    out = bash(
        env
        + 'printf "%s\\n" "${LD_LIBRARY_PATH:-}" "${LIBRARY_PATH:-}"\n'
        + 'grep -hE "^ESMF_F90(ESMF)?LINKPATHS" "$ESMFMKFILE" | grep -oE -- "-L[^ ]+" | cut -c3-\n',
        live=False,
    )
    dirs: list[str] = []
    for line in out.splitlines():
        for d in line.split(":"):
            d = d.strip()
            if (
                d
                and Path(d).is_dir()
                and not d.startswith(("/usr/", "/lib"))
                and d not in dirs
            ):
                dirs.append(d)
    return ";".join(dirs)


def cmake_install(
    env: str,
    src: Path,
    build: Path,
    prefix: Path,
    deps: Path,
    options: list[str],
    jobs: int,
) -> None:
    configure = [
        "cmake", "-S", str(src), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DCMAKE_PREFIX_PATH={deps}",
        *options,
    ]  # fmt: skip
    bash(
        env
        + f"{shlex.join(configure)} && cmake --build {shlex.quote(str(build))} -j {jobs}"
        + f" && cmake --install {shlex.quote(str(build))}"
    )


def install_dir(exec_dir: Path, dest: Path) -> None:
    """Replace `dest` with a copy of `exec_dir` (the two are on different filesystems)."""
    new = dest.with_name(f".{dest.name}.new")
    old = dest.with_name(f".{dest.name}.old")
    for path in (new, old):
        shutil.rmtree(path, ignore_errors=True)
    shutil.copytree(exec_dir, new)
    if dest.exists():
        dest.rename(old)
    new.rename(dest)
    shutil.rmtree(old, ignore_errors=True)


def check_libraries(run_modules: str, exec_dir: Path) -> None:
    """Fail if an executable has an unresolved library under the run-time modules only."""
    ldd = bash(
        run_modules
        + f"cd {shlex.quote(str(exec_dir))} && for f in *; do"
        + ' ldd "$f" 2>/dev/null | sed "s|^|$f: |"; done',
        live=False,
    )
    unresolved = [line for line in ldd.splitlines() if "not found" in line]
    if unresolved:
        raise BuildError("unresolved libraries:\n" + "\n".join(unresolved))


def write_manifest(
    exec_dir: Path, versions: dict, sources: dict[str, dict[str, str]]
) -> None:
    """Write exec_dir/manifest.yaml: release, versions of the modules and tools, sources.

    ``versions`` is utils.version_record. pyUFS reads ``ufs_utils`` to log the
    release of every run.
    """
    manifest = {
        "ufs_utils": UFS_UTILS_TAG,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **versions,
        "sources": sources,
    }
    dump_yaml(manifest, exec_dir / "manifest.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 4, 16),
        help="Parallel build jobs",
    )
    args = parser.parse_args()

    try:
        # Step 1: host modules plus the build-only modules.
        modules = default_config()["modules"]
        run_modules = module_script(modules)
        build_env = (
            module_script(modules + list(BUILD_MODULES))
            + 'export CC=mpicc CXX=mpicxx FC=mpif90 NetCDF_ROOT="$NETCDF"\n'
        )

        print(
            f"modules: {', '.join(modules)}; build only: {', '.join(BUILD_MODULES)}",
            flush=True,
        )
        toolchain = bash(build_env + PREFLIGHT, live=False).strip().splitlines()
        for line in toolchain:
            print(f"    {line}")

        tree = BUILD_ROOT / f"ufs_utils-{UFS_UTILS_TAG}"
        deps, prefix = tree / "deps", tree / "install"
        shutil.rmtree(tree, ignore_errors=True)
        deps.mkdir(parents=True)

        # Step 2: build each NCEPLIBS library at its pinned tag, installed in deps.
        sources = {}
        for name, lib in LIBS.items():
            tag, options = lib["tag"], lib.get("options", [])
            print(f"building {name} {tag}", flush=True)
            src = tree / "src" / name
            url = f"https://github.com/NOAA-EMC/NCEPLIBS-{name}.git"
            sources[name] = {"tag": tag, "commit": clone(url, tag, src)}
            cmake_install(
                build_env, src, tree / "build" / name, deps, deps, options, args.jobs
            )

        # Step 3: build UFS_UTILS (some tools need its ccpp-physics submodule).
        print(f"building UFS_UTILS {UFS_UTILS_TAG}", flush=True)
        src = tree / "src" / "UFS_UTILS"
        sources["UFS_UTILS"] = {
            "tag": UFS_UTILS_TAG,
            "commit": clone(UFS_UTILS_URL, UFS_UTILS_TAG, src),
        }
        bash("git submodule update -q --init --depth 1 ccpp-physics", cwd=src)
        # The RPATH lets the executables find their libraries without the build modules.
        rpath = link_dirs(build_env)
        print(f"    install RPATH: {rpath}")
        options = [
            "-DCMAKE_INSTALL_BINDIR=exec",
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
            f"-DCMAKE_INSTALL_RPATH={rpath}",
        ]
        cmake_install(
            build_env,
            src,
            tree / "build" / "UFS_UTILS",
            prefix,
            deps,
            options,
            args.jobs,
        )

        # Step 4: check the result.
        exec_dir = prefix / "exec"
        missing = [name for name in EXECUTABLES if not (exec_dir / name).is_file()]
        if missing:
            raise BuildError(f"not built: {', '.join(missing)}")

        # Only the run_config.yaml modules are loaded at run time, so nothing may be missing.
        print("checking shared libraries with the run-time modules", flush=True)
        check_libraries(run_modules, exec_dir)

        # Step 5: record the build, then replace bin/preprocessing with it.
        write_manifest(
            exec_dir, version_record(toolchain, modules, BUILD_MODULES), sources
        )

        dest = link_parent_dir("bin", create=True) / "preprocessing"
        install_dir(exec_dir, dest)
        print(f"installed {dest}", flush=True)
        print(f"    set ufs_bin_dir: {dest}")
        return 0
    except (BuildError, OSError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
