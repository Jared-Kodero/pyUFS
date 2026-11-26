#!/usr/bin/env python3
"""Build SHiELD for the current pyUFS release using the Oscar toolchain.

    python tools/build_shield.py                    # default release, repro, 32bit, pic, cleanall
    python tools/build_shield.py --release 202411 --comp prod
    python tools/build_shield.py --release 202604 --comp debug --fflags "-fcheck=all"
    python tools/build_shield.py --release 202604 --no-compile

What the script does, in order:
 1. Load the host modules listed in config/run_config.yaml (the same ones a run uses).
 2. Clone NOAA-GFDL/SHiELD_build in /tmp/$USER/FV3-<release>-public, which is deleted
    first, and check out the tag of the release.
 3. Run its CHECKOUT_code, which clones the model, physics, driver, FMS and FMSCoupler
    sources, and verify that each one is at the tag the release requires.
 4. Patch one line of the FMS source (the Oscar glibc already declares gettid), and append
    --fflags, when given, to site/gnu.mk.
 5. Write site/environment.gnu.sh, the compiler and library settings that COMPILE reads,
    and check that the toolchain is complete (gfortran 10 or later, MPI, netCDF, libyaml).
 6. Compile with ./COMPILE.
 7. Copy the executable to bin/ and write its manifest next to it.

The executable is named <release>.fv3.public_SHiELD_nh.<comp>.<bit>.gnu.x, for example
202604.fv3.public_SHiELD_nh.repro.32bit.gnu.x, so builds of different releases and
compiler modes and precisions never overwrite each other. The manifest has the same name plus
.manifest.yaml and records the source tags and commits, the module and compiler versions and the
patches applied. The script
prints the shield_exe value to put in run_config.yaml.

The default compiler mode is repro: the optimization of prod (-O2) plus -ggdb, so a
crash backtrace names functions and lines.
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# utils must be imported first: it puts src/ on sys.path for fv3_yaml and release.
from pyufs_links import link_parent_dir
from utils import (
    BUILD_ROOT,
    BuildError,
    default_config,
    module_script,
    version_record,
)

from fv3_yaml import dump_yaml
from release import DEFAULT_RELEASE, VALID_RELEASES

COMP_MODES = ("repro", "prod", "debug")

SHIELD_BUILD_URL = "https://github.com/NOAA-GFDL/SHiELD_build.git"


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    quiet: bool = False,
    live: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a command and raise BuildError on failure.

    With `live` the command writes straight to the terminal, unformatted.
    Otherwise its output is captured, for the git and toolchain queries.
    """
    if not quiet:
        print(f"$ {shlex.join(cmd)}", flush=True)
    if live:
        result = subprocess.run(cmd, cwd=cwd, check=False)
        if result.returncode:
            raise BuildError(
                f"{shlex.join(cmd)} failed ({result.returncode}); see the output above"
            )
        return result
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise BuildError(
            f"{shlex.join(cmd)} failed ({result.returncode}):\n"
            + (result.stdout + result.stderr)
        )
    return result


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args], quiet=True).stdout.strip()


def parse_args() -> argparse.Namespace:
    """Command line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--release",
        choices=VALID_RELEASES,
        default=DEFAULT_RELEASE,
        help=f"SHiELD release to build (default: {DEFAULT_RELEASE}); selects the source tag FV3-<release>-public",
    )
    parser.add_argument("--no-compile", action="store_true", help="Stop after checks")
    parser.add_argument(
        "--comp",
        choices=COMP_MODES,
        default="repro",
        help="Optimization mode (default: repro), options are prod, repro, debug",
    )
    parser.add_argument(
        "--bit",
        choices=("64bit", "32bit"),
        default="32bit",
        help="Precision of the FV3 dynamical core; physics stays 64-bit (default: 32bit)",
    )
    parser.add_argument(
        "--pic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable -fPIC",
    )
    parser.add_argument(
        "--clean",
        choices=("cleanall", "clean", "noclean"),
        default="cleanall",
        help="Clean scope (default: cleanall)",
    )
    parser.add_argument(
        "--avx-level",
        default="-march=x86-64-v3",
        help="AVX instruction level",
    )
    parser.add_argument(
        "--fflags",
        default="",
        help="Extra Fortran flags",
    )
    return parser.parse_args()


def environment_script(module_lines: str, avx_level: str) -> str:
    """Text of site/environment.gnu.sh: load the modules and export the build variables."""
    return f"""#!/bin/bash
{module_lines}
if [[ -z ${{NETCDF:-}} ]]; then
  echo 'NETCDF is not defined by the loaded modules' >&2; exit 1
fi
export CPATH="${{NETCDF}}/include:${{CPATH:-}}"
export NETCDF_DIR="${{NETCDF}}"
export HDF5_DIR="${{HDF5_DIR:-${{HDF5_ROOT:-${{HDF5:-$NETCDF_DIR}}}}}}"
export FMS_CPPDEFS=""
export FC=mpif90
export CC=mpicc
export CXX=mpicxx
export LD=mpif90
export TEMPLATE=site/gnu.mk
export LAUNCHER="srun --mpi=pmix"
export AVX_LEVEL={shlex.quote(avx_level)}

echo -e ' '
module list
"""


def preflight_script(env_file: Path) -> str:
    """Shell script that sources env_file, checks the toolchain and prints its versions."""
    return f"""source {shlex.quote(str(env_file))} >/dev/null || exit 1
for tool in gfortran gcc mpif90 mpicc nf-config nc-config cmake make git pkg-config; do
  command -v "$tool" >/dev/null || {{ echo "MISSING $tool"; missing=1; }}
done
pkg-config --exists yaml-0.1 || {{ echo 'MISSING libyaml'; missing=1; }}
[ -z "$missing" ] || exit 1
echo "gfortran: $(gfortran -dumpfullversion)"
echo "mpif90: $(command -v mpif90)"
echo "netcdf: $(nc-config --version) / fortran $(nf-config --version)"
echo "prefix: $NETCDF_DIR  hdf5: $HDF5_DIR"
echo "loaded modules: ${{LOADEDMODULES//:/ }}"
"""


def main() -> int:
    args = parse_args()

    try:
        modules = default_config()["modules"]

        # The source tag of the release; it also names the temporary build tree.
        tag = f"FV3-{args.release}-public"
        exe_name = f"{args.release}.fv3.public_SHiELD_nh.{args.comp}.{args.bit}.gnu.x"

        # Step 1: check that the modules load before anything is downloaded.
        module_lines = module_script(modules)
        print(f"{tag}; modules: {', '.join(modules)}", flush=True)
        run(["bash", "-c", module_lines], quiet=True)

        # Step 2: clone SHiELD_build into a fresh tree and check out the release tag.
        tree = BUILD_ROOT / tag
        build = tree / "SHiELD_build"
        src = tree / "SHiELD_SRC"
        shutil.rmtree(tree, ignore_errors=True)
        tree.mkdir(parents=True)

        print("checking out SHiELD", flush=True)
        run(["git", "clone", SHIELD_BUILD_URL, str(build)], live=True)
        git(build, "checkout", "-q", tag)

        # Step 3: CHECKOUT_code clones the source repositories into SHiELD_SRC.
        run(["./CHECKOUT_code"], cwd=build, live=True)
        run(["git", "submodule", "update", "--init", "mkmf"], cwd=build, live=True)

        # Read the tags CHECKOUT_code requires and check each repository is at its tag.
        checkout_text = (build / "CHECKOUT_code").read_text()
        variables = {}
        for name in ("release", "fms_release", "fms_c_release"):
            match = re.search(rf'^{name}="?([^"\s]+)"?', checkout_text, re.MULTILINE)
            if match is None:
                raise BuildError(f"CHECKOUT_code does not define {name}")
            variables[name] = match.group(1)
        if variables["release"] != tag:
            raise BuildError(
                f"CHECKOUT_code requests {variables['release']}, not {tag}"
            )
        expected = {
            "GFDL_atmos_cubed_sphere": tag,
            "SHiELD_physics": tag,
            "atmos_drivers": tag,
            "FMS": variables["fms_release"],
            "FMSCoupler": variables["fms_c_release"],
        }
        print("verifying source tags", flush=True)
        for name, wanted in expected.items():
            actual = git(src / name, "tag", "--points-at", "HEAD").splitlines()
            if wanted not in actual:
                raise BuildError(
                    f"{name}: expected tag {wanted} at HEAD; found {actual}"
                )
            print(f"{name}: {wanted}", flush=True)

        # The manifest starts with what is known before compiling.
        manifest = {
            "release": tag,
            "executable_name": exe_name,
            "SHiELD_build": git(build, "rev-parse", "HEAD"),
            "modules": modules,
            "compile": {
                "comp": args.comp,
                "bit": args.bit,
                "pic": args.pic,
                "clean": args.clean,
                "avx_level": args.avx_level,
                "extra_fflags": args.fflags,
            },
            "sources": {},
        }
        for repo in sorted(path for path in src.iterdir() if (path / ".git").exists()):
            manifest["sources"][repo.name] = {
                "commit": git(repo, "rev-parse", "HEAD"),
                "describe": git(repo, "describe", "--tags", "--always"),
            }

        # Step 4: remove "static" from gettid in the temporary FMS source (glibc declares it).
        affinity = src / "FMS" / "affinity" / "affinity.c"
        content = affinity.read_text()
        old = "static pid_t gettid(void)"
        patches = []
        if content.count(old) == 1:
            affinity.write_text(content.replace(old, "pid_t gettid(void)", 1))
            patches.append("FMS/affinity/affinity.c: remove static from gettid")
            print("patched FMS/affinity/affinity.c", flush=True)
        elif not re.search(r"(?m)^\s*pid_t gettid\(void\)", content):
            raise BuildError(f"unrecognized gettid declaration in {affinity}")
        manifest["patches"] = patches

        # Step 4, continued: extra Fortran flags from --fflags.
        if args.fflags.strip():
            # Appended last so it follows the prod/repro/debug selection.
            gnu_mk = build / "site" / "gnu.mk"
            gnu_mk.write_text(
                gnu_mk.read_text() + f"\nFFLAGS += {args.fflags.strip()}\n"
            )
            print(f"added to FFLAGS in {gnu_mk}: {args.fflags.strip()}", flush=True)

        # Step 5: write the build environment, then check the toolchain in it.
        env_file = build / "site" / "environment.gnu.sh"
        env_file.write_text(environment_script(module_lines, args.avx_level))
        print(f"wrote {env_file}", flush=True)

        toolchain = (
            run(["bash", "-c", preflight_script(env_file)], quiet=True)
            .stdout.strip()
            .splitlines()
        )
        for line in toolchain:
            print(line, flush=True)
        match = re.search(r"^gfortran: (\d+)", "\n".join(toolchain), re.MULTILINE)
        if match is None or int(match.group(1)) < 10:
            raise BuildError("gfortran 10 or later is required")
        manifest.update(version_record(toolchain, modules))

        compile_cmd = ["./COMPILE", "shield", "nh", args.comp, args.bit, "gnu"]
        if args.pic:
            compile_cmd.append("pic")
        compile_cmd.append(args.clean)

        if args.no_compile:
            dump_yaml(manifest, tree / "manifest.yaml")
            print(f"--no-compile: sources in {src}", flush=True)
            print(
                f"would run in {build / 'Build'}: {shlex.join(compile_cmd)}", flush=True
            )
            print(f"would install bin/{exe_name}", flush=True)
            print(
                f"manifest of the checked sources: {tree / 'manifest.yaml'}", flush=True
            )
            return 0

        # Step 6: compile; the executable must be newer than this step's start.
        print(f"compiling SHiELD ({args.comp})", flush=True)
        started = time.time()
        run(compile_cmd, cwd=build / "Build", live=True)
        exe = build / "Build" / "bin" / f"SHiELD_nh.{args.comp}.{args.bit}.gnu.x"
        if (
            not exe.is_file()
            or exe.stat().st_size == 0
            or exe.stat().st_mtime < started
        ):
            raise BuildError(f"no new executable at {exe}; see the output above")

        # Step 7: install the executable under its release name, with its manifest.
        dest = link_parent_dir("bin", create=True) / exe_name
        shutil.copy2(exe, dest)
        manifest["built"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest_path = dest.with_name(dest.name + ".manifest.yaml")
        dump_yaml(manifest, manifest_path)
        print(f"installed {dest}", flush=True)
        print(f"manifest {manifest_path}", flush=True)
        print(
            f"set shield_exe: {dest} (with shield_release: {args.release})", flush=True
        )
        return 0
    except (BuildError, OSError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
