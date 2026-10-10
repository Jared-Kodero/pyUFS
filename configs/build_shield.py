#!/usr/bin/env python3
"""Build the native SHiELD executable that matches the checked-out pyUFS branch.

    python scripts/build_shield.py                 # build for the current branch
    python scripts/build_shield.py --no-compile    # stop after checkout, verification, environment

The pyUFS branch name selects the release (202604 -> FV3-202604-public, 202411 ->
FV3-202411-public). The script

  1. reads ``shield_exe`` and ``modules`` from configs/run_config.yaml (the executable
     must be built with the modules it is launched with),
  2. checks that each module exists on this host (``module avail``),
  3. clones SHiELD_build into its own tree for the release, runs CHECKOUT_code, and verifies
     every tagged repository against the tags that CHECKOUT_code requests (the untagged
     ice_param and *_null repositories stay at the commits that the clone provides),
  4. writes SHiELD_build/site/environment.gnu.sh for this machine (module loads, MPI
     wrappers, netCDF and HDF5 locations, FMS_CPPDEFS),
  5. tests the loaded toolchain, then runs ``COMPILE shield nh prod 64bit gnu pic cleanall``
     (fresh FMS, NCEP libraries and objects; see README section 19), and
  6. copies the executable to ``shield_exe`` and writes ``<shield_exe>.manifest``.

Each release is built in its own directory, so 2024 and 2026 builds never share sources,
libraries or objects.
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
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SHIELD_BUILD_URL = "https://github.com/NOAA-GFDL/SHiELD_build.git"
MODULE_INIT = (
    "${MODULESHOME:+$MODULESHOME/init/bash}",
    "/etc/profile.d/modules.sh",
    "/etc/profile.d/lmod.sh",
    "/usr/share/lmod/lmod/init/bash",
    "/usr/share/Modules/init/bash",
)


class BuildError(RuntimeError):
    pass


def run(cmd, cwd=None, env=None, check=True, log=None, quiet=False, **kw) -> subprocess.CompletedProcess:
    """Run a command; stream output to ``log`` when given."""
    shown = cmd if isinstance(cmd, str) else shlex.join(map(str, cmd))
    if not quiet:
        print(f"    $ {shown}", flush=True)
    if log is not None:
        with open(log, "ab") as fh:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, **kw)
    else:
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, **kw)
    if check and proc.returncode != 0:
        tail = ""
        if log is not None:
            tail = "\n".join(Path(log).read_text(errors="replace").splitlines()[-25:])
        else:
            tail = (proc.stdout or "") + (proc.stderr or "")
        raise BuildError(f"command failed ({proc.returncode}): {shown}\n{tail}")
    return proc


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args], quiet=True).stdout.strip()


def bash(script: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, **kw)


def module_prelude() -> str:
    """Shell lines that define the ``module`` function, or fail if the host has none."""
    lines = [f'for f in {" ".join(MODULE_INIT)}; do [ -f "$f" ] && . "$f" && break; done']
    lines.append("type module >/dev/null 2>&1 || { echo 'no module command' >&2; exit 90; }")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- inputs


def read_run_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    for key in ("shield_exe", "modules"):
        if key not in cfg:
            raise BuildError(f"{path}: missing key {key}")
    return cfg


def current_profile() -> str:
    branch = git(REPO, "branch", "--show-current")
    m = re.fullmatch(r"(\d{6})", branch)
    if not m:
        raise BuildError(
            f"pyUFS branch '{branch}' is not a release branch such as 202604 or 202411; "
            "check one out or pass --profile"
        )
    return m.group(1)


def check_exe_name(exe: Path, tag: str) -> None:
    m = re.match(r"(FV3-\d{6}-public)_", exe.name)
    if m and m.group(1) != tag:
        raise BuildError(
            f"shield_exe is named for {m.group(1)} but the branch builds {tag}; "
            "fix shield_exe or check out the matching branch"
        )


def check_modules(wanted: list[str]) -> list[str]:
    """Return the module names; stop if the host has the module system but lacks one."""
    probe = bash(module_prelude())
    if probe.returncode == 90:
        print("    no module command on this host; the environment is assumed to be set", flush=True)
        return []
    missing = []
    for name in wanted:
        out = bash(module_prelude() + f"\nmodule --terse avail {shlex.quote(name)} 2>&1")
        listed = [ln.strip() for ln in (out.stdout + out.stderr).splitlines() if ln.strip()]
        if not any(ln.split("/")[0].rstrip(":") == name.split("/")[0] for ln in listed):
            missing.append(name)
    if missing:
        raise BuildError(
            "modules not found on this host: " + ", ".join(missing)
            + "\nrun 'module avail' and change the modules key of configs/run_config.yaml"
        )
    return wanted


# ----------------------------------------------------------------------------- checkout


def checkout(profile: str, base: Path, force: bool) -> tuple[Path, Path, str]:
    tag = f"FV3-{profile}-public"
    tree = base / tag
    if tree.exists():
        if not force:
            raise BuildError(f"{tree} exists; pass --force to rebuild from scratch")
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    build, src = tree / "SHiELD_build", tree / "SHiELD_SRC"

    print(">>> SHiELD_build", tag, flush=True)
    run(["git", "clone", "-q", SHIELD_BUILD_URL, str(build)])
    git(build, "checkout", "-q", tag)
    run(["git", "-C", str(build), "submodule", "update", "--init", "mkmf"])

    print(">>> CHECKOUT_code", flush=True)
    # CHECKOUT_code is a POSIX sh script that sources $MODULESHOME/init/sh and aborts when
    # that file is missing; supply a stub on hosts without environment modules.
    env = dict(os.environ)
    stub = None
    if not (env.get("MODULESHOME") and Path(env["MODULESHOME"], "init/sh").is_file()):
        stub = tempfile.mkdtemp()
        Path(stub, "init").mkdir()
        Path(stub, "init/sh").write_text("module() { :; }\n")
        env["MODULESHOME"] = stub
    try:
        run(["./CHECKOUT_code"], cwd=build, env=env, log=tree / "checkout.log")
    finally:
        if stub:
            shutil.rmtree(stub, ignore_errors=True)
    return build, src, tag


def requested_tags(build: Path, tag: str) -> dict[str, str]:
    text = (build / "CHECKOUT_code").read_text()

    def var(name: str) -> str:
        m = re.search(rf'^{name}="?([^"\s]+)"?', text, re.M)
        if not m:
            raise BuildError(f"CHECKOUT_code does not define {name}")
        return m.group(1)

    release = var("release")
    if release != tag:
        raise BuildError(f"CHECKOUT_code requests {release}, not {tag}")
    return {
        "GFDL_atmos_cubed_sphere": release,
        "SHiELD_physics": release,
        "atmos_drivers": release,
        "FMS": var("fms_release"),
        "FMSCoupler": var("fms_c_release"),
    }


def verify_and_record(build: Path, src: Path, tag: str) -> list[str]:
    print(">>> verifying tagged repositories", flush=True)
    bad = []
    for repo, want in requested_tags(build, tag).items():
        tags = git(src / repo, "tag", "--points-at", "HEAD").split()
        ok = want in tags
        print(f"    {repo:26s} {want if ok else 'tags at HEAD: ' + (' '.join(tags) or 'none')}")
        if not ok:
            bad.append(f"{repo}: expected {want}")
    if bad:
        raise BuildError("checkout does not match " + tag + ":\n  " + "\n  ".join(bad))

    date = git(build, "log", "-1", "--format=%cI", tag)
    lines = [f"profile: {tag} (SHiELD_build {git(build, 'rev-parse', 'HEAD')}, release date {date})"]
    for repo in sorted(p for p in src.iterdir() if (p / ".git").exists()):
        lines.append(f"{repo.name}: {git(repo, 'rev-parse', 'HEAD')} {git(repo, 'describe', '--tags', '--always')}")
    return lines


# ----------------------------------------------------------------------------- environment


def glibc_has_gettid() -> bool:
    """glibc 2.30 and later declares gettid(); FMS affinity.c then needs -DHAVE_GETTID."""
    out = subprocess.run(["ldd", "--version"], text=True, capture_output=True).stdout
    m = re.search(r"(\d+)\.(\d+)\s*$", out.splitlines()[0]) if out else None
    return bool(m) and (int(m[1]), int(m[2])) >= (2, 30)


def write_environment(build: Path, modules: list[str], march: str, cppdefs: str) -> Path:
    """Replace site/environment.gnu.sh by a machine independent file (no hostname case)."""
    loads = "\n".join(f"  module load {m}" for m in modules) or "  :"
    text = f"""#!/bin/sh
# Generated by scripts/build_shield.py for this machine. Do not edit; rerun the script.

for f in "${{MODULESHOME}}/init/sh" /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
  if [ -f "$f" ]; then . "$f"; break; fi
done
if type module >/dev/null 2>&1; then
{loads}
fi

# Locations used by site/gnu.mk
export NETCDF_DIR="$(nc-config --prefix 2>/dev/null)"
export HDF5_DIR="${{HDF5_DIR:-${{HDF5_ROOT:-${{HDF5:-$NETCDF_DIR}}}}}}"
export CPATH="${{NETCDF_DIR}}/include:${{CPATH}}"
export LIBRARY_PATH="${{LIBRARY_PATH}}:${{NETCDF_DIR}}/lib:${{HDF5_DIR}}/lib"

export FMS_CPPDEFS="{cppdefs}"

export FC=mpif90
export CC=mpicc
export CXX=mpicxx
export LD=mpif90
export TEMPLATE=site/gnu.mk
export LAUNCHER=srun

# empty: generic x86-64 code that runs on every node; set with --march to tune
export AVX_LEVEL="{march}"

module list 2>&1 || true
"""
    path = build / "site" / "environment.gnu.sh"
    path.write_text(text)
    return path


def preflight(env_file: Path) -> list[str]:
    """Source the generated environment and test the tools the build needs."""
    print(">>> testing the toolchain", flush=True)
    script = f""". {shlex.quote(str(env_file))} >/dev/null 2>&1
for t in gfortran gcc mpif90 mpicc nf-config nc-config cmake make git pkg-config; do
  command -v $t >/dev/null || {{ echo "MISSING $t"; miss=1; }}
done
pkg-config --exists yaml-0.1 || {{ echo "MISSING libyaml (pkg-config yaml-0.1)"; miss=1; }}
echo "gfortran: $(gfortran -dumpfullversion 2>/dev/null)"
echo "mpif90:   $(command -v mpif90)"
echo "netcdf:   $(nc-config --version 2>/dev/null) / fortran $(nf-config --version 2>/dev/null)"
echo "prefix:   $NETCDF_DIR  hdf5: $HDF5_DIR"
[ -z "$miss" ]
"""
    proc = bash(script)
    out = proc.stdout.strip().splitlines()
    for line in out:
        print("    " + line)
    if proc.returncode != 0:
        raise BuildError("the loaded environment is incomplete (lines marked MISSING above);\n"
                         "load the modules that provide them (module avail) and add them to the "
                         "modules key of configs/run_config.yaml")
    ver = next((ln.split()[1] for ln in out if ln.startswith("gfortran:")), "")
    if ver and int(ver.split(".")[0]) < 10:
        raise BuildError(f"gfortran {ver} is too old; the build needs 10 or later (-fallow-argument-mismatch)")
    return out


# ----------------------------------------------------------------------------- compile


def compile_model(build: Path, tree: Path) -> Path:
    print(">>> COMPILE shield nh prod 64bit gnu pic cleanall (this takes a while)", flush=True)
    log = tree / "compile.log"
    started = time.time()
    run(["./COMPILE", "shield", "nh", "prod", "64bit", "gnu", "pic", "cleanall"],
        cwd=build / "Build", log=log, check=False)
    exe = build / "Build" / "bin" / "SHiELD_nh.prod.64bit.gnu.x"
    # COMPILE tests the exit status of the final mv, not of make.
    if not exe.is_file() or exe.stat().st_size == 0 or exe.stat().st_mtime < started:
        detail = build / "Build" / "build_shield_nh.prod.64bit.gnu.out"
        tail = ""
        if detail.is_file():
            errs = [ln for ln in detail.read_text(errors="replace").splitlines() if "Error" in ln]
            tail = "\n".join(errs[:15])
        raise BuildError(f"no new executable; see {detail} and {log}\n{tail}")
    return exe


def install(exe: Path, dest: Path, manifest: list[str], tree: Path, pre: list[str], cppdefs: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, dest)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    manifest += [
        f"executable: {dest}",
        f"sha256: {sha}",
        f"built: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} on {os.uname().nodename}",
        f"FMS_CPPDEFS: {cppdefs}",
        *[f"toolchain {ln}" for ln in pre],
        f"build tree: {tree}",
    ]
    Path(str(dest) + ".manifest").write_text("\n".join(manifest) + "\n")
    print(f">>> installed {dest}\n    sha256 {sha}")


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", help="release such as 202604 (default: the checked-out pyUFS branch)")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "run_config.yaml")
    ap.add_argument("--root", type=Path, default=Path.home() / "shield_build",
                    help="directory that receives FV3-<profile>-public/ (default ~/shield_build)")
    ap.add_argument("--exe", type=Path, help="install path (default: shield_exe of run_config.yaml)")
    ap.add_argument("--modules", help="comma separated module list (default: modules of run_config.yaml)")
    ap.add_argument("--march", default="", help="value of AVX_LEVEL, for example -march=x86-64-v3 (default none)")
    ap.add_argument("--no-compile", action="store_true", help="stop after checkout, verification and environment")
    ap.add_argument("--skip-preflight", action="store_true", help="do not test the toolchain")
    ap.add_argument("--force", action="store_true", help="delete an existing build tree first")
    args = ap.parse_args()

    try:
        cfg = read_run_config(args.config)
        profile = args.profile or current_profile()
        tag = f"FV3-{profile}-public"
        exe_path = Path(os.path.expandvars(str(args.exe or cfg["shield_exe"])))
        if not str(exe_path):
            raise BuildError("shield_exe is empty (container mode); pass --exe")
        check_exe_name(exe_path, tag)
        wanted = [m.strip() for m in args.modules.split(",")] if args.modules else list(cfg["modules"])
        print(f">>> pyUFS branch {git(REPO, 'branch', '--show-current')} -> {tag}; modules: {' '.join(wanted)}")
        mods = check_modules(wanted)

        build, src, tag = checkout(profile, args.root.expanduser().resolve(), args.force)
        tree = build.parent
        manifest = verify_and_record(build, src, tag)

        cppdefs = "-DHAVE_GETTID" if glibc_has_gettid() else ""
        env_file = write_environment(build, mods, args.march, cppdefs)
        print(f">>> wrote {env_file}")
        pre: list[str] = []
        if not args.skip_preflight:
            pre = preflight(env_file)
        if args.no_compile:
            (tree / "manifest.txt").write_text("\n".join(manifest) + "\n")
            print(f">>> --no-compile: sources in {src}")
            return 0

        exe = compile_model(build, tree)
        install(exe, exe_path, manifest, tree, pre, cppdefs)
    except BuildError as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
