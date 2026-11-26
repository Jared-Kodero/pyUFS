"""Dispatcher between the pyUFS core and the optional modules in plugins/.

This is the only module in src/ that imports plugins. A plugin is a package in plugins/
listed in plugins.__all__; its hooks are called only when its section is in the
plugins file (``plugins_config`` in run_config.yaml), which the cold start loads
into ``state.plugins`` as a mapping of plugin name to its settings.

A plugin never imports from src/. Every hook receives its settings (``config``), the
live run state (``state=state``) and, where it needs them, the core functions as
keyword arguments. Each call below lists exactly what that plugin gets. The contract is
documented in plugins/README.md.
"""

from __future__ import annotations

import copy
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import TYPE_CHECKING


import fv3_utils
from fv3_state import state
from fv3_yaml import load_config

if TYPE_CHECKING:
    import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plugins as manifest
from plugins import derived, pamip, sm_perturbations, sst_perturbation


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
    """Read the plugins file into ``{plugin: settings}``. Sections that are null or
    empty are dropped, so a plugin is active exactly when its name is a key. The
    resolved paths of the files it includes are appended to ``includes`` when it
    is given."""
    data = load_config(path, loaded=includes)

    plugins = {}
    for name, user in data.items():
        if name not in manifest.__all__:
            message = f"Unknown plugin `{name}` in {path}."
            if match := get_close_matches(name, manifest.__all__, n=1):
                message += f" Did you mean {match[0]!r}?"
            raise ValueError(message)
        if not user:
            continue
        if not isinstance(user, dict):
            raise TypeError(f"Plugin `{name}` in {path} must be a mapping.")
        defaults = getattr(getattr(manifest, name), "DEFAULTS", None)
        plugins[name] = user if defaults is None else resolve(name, user, defaults)
    return plugins


def active_plugins() -> list[str]:
    """The plugins set in state.plugins, in the order of plugins.__all__."""
    return [name for name in manifest.__all__ if name in state.plugins]


def hashed(plugins: dict | None) -> dict:
    """The settings of the plugins that are part of the case checksum."""
    return {
        name: settings
        for name, settings in (plugins or {}).items()
        if getattr(getattr(manifest, name), "HASHED", False)
    }


def add_derived_variables(ds: xr.Dataset) -> xr.Dataset:
    """Add the derived variables to ds. Called once per output stream."""
    from fv3_diag_files import variables

    return derived.add_derived_variables(
        ds,
        config=state.plugins.get("derived"),
        state=state,
        variables=variables,
    )


def validate() -> None:
    """Check and normalise the settings of every active plugin. Called before grid and
    IC generation, so a bad configuration fails early."""
    for name in active_plugins():
        plugin = getattr(manifest, name)
        if hasattr(plugin, "validate"):
            plugin.validate(config=state.plugins[name])


def stage_derived_module(*, restart: bool) -> None:
    """Copy the user's derived-variable file to the case directory (cold start). The
    variables themselves are computed per output stream by add_derived_variables."""
    derived.apply(config=state.plugins["derived"], state=state, restart=restart)


def apply_pamip(*, restart: bool) -> None:
    """Stage the PAMIP external files and boundary data, then patch the namelists."""
    import grib_io
    import latlon_to_fv3
    from fv3_fixed_files import update_fixed_files
    from fv3_namelists import patch_nml

    pamip.apply(
        config=state.plugins["pamip"],
        state=state,
        restart=restart,
        patch_nml=patch_nml,
        update_fixed_files=update_fixed_files,
        open_grib=grib_io.open_grib,
        save_grib=grib_io.save_grib,
        ll_regrid=latlon_to_fv3.ll_regrid,
        fill_nearest=latlon_to_fv3.fill_nearest,
        load_grid=latlon_to_fv3.load_grid,
        sfc_filename=latlon_to_fv3.sfc_filename,
        to_fv3_grid=latlon_to_fv3.to_fv3_grid,
        copy_path=fv3_utils.copy_path,
    )


def apply_sst_perturbation(*, restart: bool) -> None:
    """Perturb the SST climatology the namelist points at, after pamip has set it."""
    import grib_io
    import latlon_to_fv3
    from fv3_namelists import patch_nml
    from fv3_runtime import read_namelist

    sst_perturbation.apply(
        config=state.plugins["sst_perturbation"],
        state=state,
        restart=restart,
        patch_nml=patch_nml,
        read_namelist=read_namelist,
        open_grib=grib_io.open_grib,
        save_grib=grib_io.save_grib,
        ll_regrid=latlon_to_fv3.ll_regrid,
    )


def apply_sm_perturbations(*, restart: bool) -> None:
    """Perturb the surface restart files (cold start and each later segment)."""
    import latlon_to_fv3

    sm_perturbations.apply(
        config=state.plugins["sm_perturbations"],
        state=state,
        restart=restart,
        copy_path=fv3_utils.copy_path,
        segment_hours=fv3_utils.segment_hours,
        load_grid=latlon_to_fv3.load_grid,
        sfc_filename=latlon_to_fv3.sfc_filename,
        to_fv3_grid=latlon_to_fv3.to_fv3_grid,
    )


APPLY = {
    "derived": stage_derived_module,
    "pamip": apply_pamip,
    "sst_perturbation": apply_sst_perturbation,
    "sm_perturbations": apply_sm_perturbations,
}


def do_plugins(*, restart: bool) -> None:
    """Last step of a cold start and of every restart segment: run the active plugins
    in the order of plugins.__all__."""
    for name in active_plugins():
        APPLY[name](restart=restart)
