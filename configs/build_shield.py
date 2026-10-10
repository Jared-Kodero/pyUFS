#!/usr/bin/env python3
"""Build SHiELD for the current pyUFS release using the Oscar toolchain.

    python configs/build_shield.py                    # repro, 64bit, pic, cleanall
    python configs/build_shield.py --comp prod
    python configs/build_shield.py --comp debug --fflags "-fcheck=all"
    python configs/build_shield.py --no-compile
    python configs/build_shield.py --force

Read modules and the final executable path from configs/run_config.yaml. Build in
/tmp/$USER/FV3-<release>-public and install only the executable and its manifest.

The default compiler mode is repro: the optimization of prod (-O2) plus -ggdb, so a
crash backtrace names functions and lines. The installed file name carries the mode
(..._SHiELD_nh.repro.64bit.gnu.x); set shield_exe in run_config.yaml to the path that
the script prints.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

COMP_MODES = ("repro", "prod", "debug")
CONFIG_DIR = Path(__file__).resolve().parent
REPO = CONFIG_DIR.parent
BUILD_ROOT = Path("/tmp") / (os.environ.get("USER") or Path.home().name)
SHIELD_BUILD_URL = "https://github.com/NOAA-GFDL/SHiELD_build.git"


class BuildError(RuntimeError):
    """An invalid build configuration or failed build step."""


def step(message: str) -> None:
    """Print a progress line."""
    print(f">>> {message}", flush=True)


def detail(message: str) -> None:
    """Print an indented line under the current step."""
    print(f"    {message}", flush=True)


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
        detail(f"$ {shlex.join(cmd)}")
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


def installed_path(dest: Path, comp: str, bit: str) -> Path:
    """Name the installed file for the build mode.

    shield_exe names such as ..._SHiELD_nh.prod.64bit.gnu.x carry the mode and the
    precision; they are rewritten so the file name matches what was built.
    """
    name, count = re.subn(
        r"\.(?:prod|repro|debug)\.(?:32|64)bit\.", f".{comp}.{bit}.", dest.name, count=1
    )
    return dest.with_name(name) if count else dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--force", action="store_true", help="Replace existing build tree"
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
        default="64bit",
        help="Precision (default: 64bit)",
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
        default="-march=native",
        help="AVX instruction level",
    )
    parser.add_argument(
        "--fflags",
        default="",
        help="Extra Fortran flags",
    )
    args = parser.parse_args()

    try:
        config_path = CONFIG_DIR / "run_config.yaml"
        config = yaml.safe_load(config_path.read_text()) or {}
        modules = config.get("modules")
        if (
            not isinstance(modules, list)
            or not modules
            or not all(isinstance(name, str) and name.strip() for name in modules)
        ):
            raise BuildError(f"{config_path}: modules must be a nonempty list")
        raw_dest = config.get("shield_exe")
        if not isinstance(raw_dest, str) or not raw_dest.strip():
            raise BuildError(f"{config_path}: shield_exe must be a nonempty path")
        dest = Path(os.path.expandvars(raw_dest)).expanduser()
        config_dest = dest

        branch = git(REPO, "branch", "--show-current")
        if not re.fullmatch(r"\d{6}", branch):
            raise BuildError(f"pyUFS branch '{branch}' is not a six-digit release")
        tag = f"FV3-{branch}-public"
        named_release = re.match(r"(FV3-\d{6}-public)_", dest.name)
        if named_release and named_release.group(1) != tag:
            raise BuildError(
                f"shield_exe is named for {named_release.group(1)}, not {tag}"
            )

        # Source the Oscar module system and load the exact modules specified in YAML.
        loads = "\n".join(
            f"module load {shlex.quote(name)} || exit 1" for name in modules
        )
        module_init = """if [[ -z ${MODULESHOME:-} || ! -f $MODULESHOME/init/sh ]]; then
  echo 'Missing $MODULESHOME/init/sh' >&2; exit 1
fi
source "$MODULESHOME/init/sh"
"""
        step(f"{tag}; modules: {', '.join(modules)}")
        run(["bash", "-c", module_init + loads], quiet=True)

        tree = BUILD_ROOT / tag
        build = tree / "SHiELD_build"
        src = tree / "SHiELD_SRC"
        if tree.exists():
            if not args.force:
                raise BuildError(f"{tree} already exists; use --force to rebuild")
            shutil.rmtree(tree)
        tree.mkdir(parents=True)

        step("checking out SHiELD")
        run(["git", "clone", SHIELD_BUILD_URL, str(build)], live=True)
        git(build, "checkout", "-q", tag)
        run(["./CHECKOUT_code"], cwd=build, live=True)
        run(["git", "submodule", "update", "--init", "mkmf"], cwd=build, live=True)

        # Verify the five tagged source repositories selected by CHECKOUT_code.
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
        step("verifying source tags")
        for name, wanted in expected.items():
            actual = git(src / name, "tag", "--points-at", "HEAD").splitlines()
            if wanted not in actual:
                raise BuildError(
                    f"{name}: expected tag {wanted} at HEAD; found {actual}"
                )
            detail(f"{name}: {wanted}")

        manifest = [
            f"release: {tag}",
            f"SHiELD_build: {git(build, 'rev-parse', 'HEAD')}",
            f"modules: {', '.join(modules)}",
            f"compile: shield nh {args.comp} {args.bit} gnu{' pic' if args.pic else ''} {args.clean}",
            f"AVX_LEVEL: {args.avx_level}",
            f"extra FFLAGS: {args.fflags}",
        ]
        for repo in sorted(path for path in src.iterdir() if (path / ".git").exists()):
            manifest.append(
                f"{repo.name}: {git(repo, 'rev-parse', 'HEAD')} "
                + f"{git(repo, 'describe', '--tags', '--always')}"
            )

        # Apply the Oscar glibc gettid workaround only to the temporary FMS source.
        affinity = src / "FMS" / "affinity" / "affinity.c"
        content = affinity.read_text()
        old = "static pid_t gettid(void)"
        if content.count(old) == 1:
            affinity.write_text(content.replace(old, "pid_t gettid(void)", 1))
            step("patched FMS/affinity/affinity.c")
        elif not re.search(r"(?m)^\s*pid_t gettid\(void\)", content):
            raise BuildError(f"unrecognized gettid declaration in {affinity}")
        affinity_sha = hashlib.sha256(affinity.read_bytes()).hexdigest()
        manifest.append(f"FMS affinity.c sha256: {affinity_sha}")

        if args.fflags.strip():
            # Appended last, so it follows the prod/repro/debug selection; recipes
            # expand FFLAGS when they run, which is after the whole file is read.
            gnu_mk = build / "site" / "gnu.mk"
            gnu_mk.write_text(
                gnu_mk.read_text() + f"\nFFLAGS += {args.fflags.strip()}\n"
            )
            step(f"added to FFLAGS in {gnu_mk}: {args.fflags.strip()}")

        # Generate the Oscar build environment; module names come from YAML.
        env_file = build / "site" / "environment.gnu.sh"
        env_file.write_text(
            f"""#!/bin/bash
# Generated by configs/build_shield.py for Oscar.
{module_init}{loads}

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
export AVX_LEVEL={shlex.quote(args.avx_level)}

echo -e ' '
module list
"""
        )
        step(f"wrote {env_file}")

        # Check the same environment that COMPILE will source.
        preflight = f"""source {shlex.quote(str(env_file))} >/dev/null || exit 1
for tool in gfortran gcc mpif90 mpicc nf-config nc-config cmake make git pkg-config; do
  command -v "$tool" >/dev/null || {{ echo "MISSING $tool"; missing=1; }}
done
pkg-config --exists yaml-0.1 || {{ echo 'MISSING libyaml'; missing=1; }}
[ -z "$missing" ] || exit 1
echo "gfortran: $(gfortran -dumpfullversion)"
echo "mpif90: $(command -v mpif90)"
echo "netcdf: $(nc-config --version) / fortran $(nf-config --version)"
echo "prefix: $NETCDF_DIR  hdf5: $HDF5_DIR"
"""
        toolchain = (
            run(["bash", "-c", preflight], quiet=True).stdout.strip().splitlines()
        )
        for line in toolchain:
            detail(line)
        match = re.search(r"^gfortran: (\d+)", "\n".join(toolchain), re.MULTILINE)
        if match is None or int(match.group(1)) < 10:
            raise BuildError("gfortran 10 or later is required")

        compile_cmd = ["./COMPILE", "shield", "nh", args.comp, args.bit, "gnu"]
        if args.pic:
            compile_cmd.append("pic")
        compile_cmd.append(args.clean)
        dest = installed_path(dest, args.comp, args.bit)

        if args.no_compile:
            (tree / "manifest.txt").write_text("\n".join(manifest) + "\n")
            step(f"--no-compile: sources in {src}")
            step(f"would run in {build / 'Build'}: {shlex.join(compile_cmd)}")
            step(f"would install {dest}")
            return 0

        step(f"compiling SHiELD ({args.comp})")
        started = time.time()
        run(compile_cmd, cwd=build / "Build", live=True)
        exe = build / "Build" / "bin" / f"SHiELD_nh.{args.comp}.{args.bit}.gnu.x"
        if (
            not exe.is_file()
            or exe.stat().st_size == 0
            or exe.stat().st_mtime < started
        ):
            raise BuildError(f"no new executable at {exe}; see the output above")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, dest)
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        manifest.extend(
            [
                f"executable: {dest}",
                f"sha256: {sha}",
                "built: "
                + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                + f" on {os.uname().nodename}",
                "FMS_CPPDEFS: ",
                *[f"toolchain {line}" for line in toolchain],
                f"build tree: {tree}",
            ]
        )
        dest.with_name(dest.name + ".manifest").write_text("\n".join(manifest) + "\n")
        step(f"installed {dest}")
        detail(f"sha256 {sha}")
        if config_dest != dest:
            step(f"set shield_exe: {dest}")
        return 0
    except (BuildError, OSError, yaml.YAMLError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
