"""Helpers shared by tools/build_shield.py and tools/build_ufs.py."""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD_ROOT = Path("/tmp") / (os.environ.get("USER") or Path.home().name)

# The build scripts read the default configuration (config/run_config.yaml) through src/.
sys.path.insert(0, str(REPO / "src"))
from fv3_yaml import load_config


class BuildError(RuntimeError):
    """A failed build step."""


def default_config() -> dict:
    """Repository defaults: config/run_config.yaml and its includes."""
    config = load_config(REPO / "config" / "run_config.yaml")
    modules = config.get("modules")
    if not modules or not isinstance(modules, list):
        raise BuildError("config/run_config.yaml: 'modules' must be a non-empty list")
    return config


# Spack appends a four-character hash to every module version, e.g. 4.9.3-kuxq.
SPACK_HASH = re.compile(r"-[a-z0-9]{4}$")


def module_versions(loaded: str) -> dict[str, str]:
    """Map the ``name/version-hash`` entries of a module list to ``name: version``.

    ``loaded`` is space or colon separated (LOADEDMODULES). The Spack hash suffix
    is removed so the versions stay readable and comparable between builds. A
    version that itself ends in a dash and four lowercase letters or digits (for
    example ``1.0-rc1``) loses that ending too.
    """
    versions = {}
    for item in loaded.replace(":", " ").split():
        name, _, version = item.partition("/")
        versions[name] = SPACK_HASH.sub("", version)
    return versions


def version_record(
    lines: list[str], modules: list[str], build_modules: tuple[str, ...] = ()
) -> dict:
    """Version fields of a build manifest from a preflight script's output.

    ``lines`` are the ``key: value`` lines the script printed. It must print
    ``gfortran: <version>`` and ``loaded modules: <module list>``; ``cmake:
    cmake version <version>`` is used when present. Returns ``modules`` (and
    ``build_modules`` when given) as name: version maps, ``toolchain`` with the
    compiler versions, and ``dependencies``, every other loaded module.
    """
    printed = dict(line.split(": ", 1) for line in lines if ": " in line)
    loaded = module_versions(printed.get("loaded modules", ""))
    if not loaded or "gfortran" not in printed:
        raise BuildError("the toolchain check printed no module list or gfortran version")

    def pick(names) -> dict[str, str]:
        return {name: loaded.get(name.split("/")[0], "unknown") for name in names}

    record = {"modules": pick(modules)}
    if build_modules:
        record["build_modules"] = pick(build_modules)
    toolchain = {"gfortran": printed["gfortran"].strip()}
    if "cmake" in printed:
        toolchain["cmake"] = printed["cmake"].removeprefix("cmake version ").strip()
    record["toolchain"] = toolchain
    requested = {name.split("/")[0] for name in [*modules, *build_modules]}
    record["dependencies"] = {k: v for k, v in loaded.items() if k not in requested}
    return record


def module_script(modules: list[str]) -> str:
    """Shell lines that load `modules` in a fresh bash."""
    init = (
        'if [[ -z ${MODULESHOME:-} || ! -f $MODULESHOME/init/sh ]]; then\n'
        "  echo 'Missing $MODULESHOME/init/sh' >&2; exit 1\n"
        "fi\n"
        'source "$MODULESHOME/init/sh"\n'
    )
    loads = "".join(f"module load {shlex.quote(name)} || exit 1\n" for name in modules)
    return init + loads
