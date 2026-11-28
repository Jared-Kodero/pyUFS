"""Dispatcher between the pyUFS core and the plugins.

A plugin is a folder ``<name>/`` holding an ``__init__.py``. It is found by name in the
``plugin_dirs`` listed in the plugins file (``plugins_config`` in run_config.yaml), then
in ``pyUFS/plugins``. A plugin is active when its section is in the plugins file, and
the active plugins run in the order the file lists them. Its module defines the hooks
it needs (at least one); a hook that is not defined is skipped:

    validate(*, config)                                     before grid and IC generation
    cold_start(*, config, state) -> dict | None             end of the cold start
    restart(*, config, state) -> dict | None                end of each restart segment
    regrid(ds, *, config, state, variables) -> ds | None    once per output stream

and optionally:

    DEFAULTS    the keys of its section with their defaults
    HASHED      true when its settings are part of the case checksum

cold_start and restart return namelist changes, ``{group: {key: value}}``, which the
core merges into the namelists before the next plugin runs. A plugin imports from
``core/`` only the state-free library modules: fv3_utils, grib_io and latlon_to_fv3.
See plugins/README.md.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import os
import sys
from difflib import get_close_matches
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal

from fv3_state import state
from fv3_utils import load_config

if TYPE_CHECKING:
    import xarray as xr

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"
# How the core calls each hook: positional arguments and keyword names.
HOOKS = {
    "validate": ((), ("config",)),
    "cold_start": ((), ("config", "state")),
    "restart": ((), ("config", "state")),
    "regrid": ((None,), ("config", "state", "variables")),
}


def plugin_dirs() -> list[Path]:
    """The folders searched for plugins, in order."""
    return [Path(d) for d in state.plugin_dirs or []] + [PLUGINS_DIR]


def load(name: str) -> ModuleType:
    """Import plugin ``name`` from the first folder that holds it. The module is
    registered as ``pyufs_plugin_<name>``, so it cannot replace a core module."""
    key = f"pyufs_plugin_{name}"
    if key in sys.modules:
        return sys.modules[key]
    for folder in plugin_dirs():
        init = folder / name / "__init__.py"
        if init.is_file():
            spec = importlib.util.spec_from_file_location(
                key, init, submodule_search_locations=[str(init.parent)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            try:
                spec.loader.exec_module(module)
                check(name, module, init)
            except BaseException:
                del sys.modules[key]
                raise
            return module
    names = [p.parent.name for d in plugin_dirs() for p in d.glob("*/__init__.py")]
    message = f"Unknown plugin `{name}`: no {name}/__init__.py in {plugin_dirs()}."
    if match := get_close_matches(name, names, n=1):
        message += f" Did you mean {match[0]!r}?"
    raise ValueError(message)


def check(name: str, module: ModuleType, init: Path) -> None:
    """Raise unless the plugin defines at least one hook and each hook it defines
    accepts the arguments the core passes."""
    if not any(hasattr(module, hook) for hook in HOOKS):
        raise ValueError(f"Plugin `{name}` ({init}) defines none of {tuple(HOOKS)}.")
    for hook, (args, keys) in HOOKS.items():
        if func := getattr(module, hook, None):
            try:
                inspect.signature(func).bind(*args, **dict.fromkeys(keys))
            except TypeError as error:
                call = ", ".join(["ds"] * len(args) + ["*", *keys])
                raise TypeError(
                    f"Plugin `{name}` ({init}): {hook} must accept ({call}): {error}"
                ) from error


def resolve(name: str, user: dict, defaults: dict) -> dict:
    """The user's settings of plugin ``name`` over its defaults; an unknown key stops
    the run with the closest valid key suggested."""
    for key in user:
        if key not in defaults:
            message = f"Unknown key `{key}` in plugin `{name}`."
            if match := get_close_matches(key, defaults, n=1):
                message += f" Did you mean {match[0]!r}?"
            raise ValueError(message)
    return {**copy.deepcopy(defaults), **user}


def load_plugins(path: Path, includes: list[str] | None = None) -> dict[str, dict]:
    """Read the plugins file into ``{plugin: settings}`` and set state.plugin_dirs. A
    section ``true`` enables a plugin with its defaults; null, false and empty sections
    are dropped, so a plugin is active exactly when its name is a key. Relative
    plugin_dirs are read from the plugins file's folder. The resolved paths of the files
    it includes are appended to ``includes`` when it is given."""
    data = load_config(path, includes=includes)
    state.plugin_dirs = [
        str((path.parent / Path(os.path.expandvars(d)).expanduser()).resolve())
        for d in data.pop("plugin_dirs", None) or []
    ]

    plugins = {}
    for name, user in data.items():
        module = load(name)
        if user is True:
            user = {}
        elif not user:
            continue
        if not isinstance(user, dict):
            raise TypeError(f"Plugin `{name}` in {path} must be a mapping.")
        defaults = getattr(module, "DEFAULTS", None)
        plugins[name] = user if defaults is None else resolve(name, user, defaults)
    return plugins


def hashed(plugins: dict | None) -> dict:
    """The settings of the plugins that are part of the case checksum."""
    return {
        name: settings
        for name, settings in (plugins or {}).items()
        if getattr(load(name), "HASHED", False)
    }


def validate() -> None:
    """Check the settings of the active plugins. Called before grid and IC generation,
    so a bad configuration fails early."""
    for name, config in state.plugins.items():
        if hook := getattr(load(name), "validate", None):
            hook(config=config)


def apply(
    stage: Literal["cold_start", "restart", "regrid"],
    ds: xr.Dataset | None = None,
    variables: dict[str, str] | None = None,
) -> xr.Dataset | None:
    """Call hook ``stage`` of each active plugin in the order of the plugins file.

    cold_start and restart: the last step of the cold start and of each restart
    segment; namelist changes a plugin returns are merged before the next plugin runs.
    regrid: once per output stream, after the built-in derived variables; a dataset a
    plugin returns replaces ds, and ds is returned.
    """
    patched = False
    for name, config in state.plugins.items():
        hook = getattr(load(name), stage, None)
        if hook is None:
            continue
        if stage == "regrid":
            new = hook(ds, config=config, state=state, variables=variables)
            ds = ds if new is None else new
        elif changes := hook(config=config, state=state):
            from fv3_namelists import patch_nml

            patch_nml(changes)
            patched = True
    if patched:
        from fv3_fixed_files import update_fixed_files

        update_fixed_files()  # the CO2 table follows ictm in the patched namelist
    return ds
