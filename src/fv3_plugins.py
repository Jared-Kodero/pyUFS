"""Dispatcher between the pyUFS core and the optional modules in plugins/.

This is the only module in src/ that imports plugins. A plugin is a package in plugins/
listed in plugins.__all__; its hooks are called only when one of its ACTIVE_KEYS is
set in run_config.yaml.

A plugin never imports from src/. Every hook receives the live run state
(``state=state``) and, where it needs them, the core functions as keyword arguments.
Each call below lists exactly what that plugin gets. A hook changes the run by
assigning to ``state``; pamip changes the namelists with ``patch_nml``. The contract is
documented in plugins/README.md.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import fv3_utils
from fv3_state import state

if TYPE_CHECKING:
    import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plugins as manifest
from plugins import derived, pamip, sm_perturbations

log = logging.getLogger("PLUGINS")


def is_active(plugin: ModuleType) -> bool:
    """True when one of the plugin's ACTIVE_KEYS is set in run_config.yaml."""
    return any(state[key] is not None for key in plugin.ACTIVE_KEYS)


def active_plugins() -> list[ModuleType]:
    """The plugins with an active key set, in the order of plugins.__all__."""
    plugins = [getattr(manifest, name) for name in manifest.__all__]
    return [plugin for plugin in plugins if is_active(plugin)]


def hash_keys() -> list[str]:
    """run_config keys that plugins add to the case checksum when set."""
    return [
        key for name in manifest.__all__ for key in getattr(manifest, name).HASH_KEYS
    ]


def add_derived_variables(ds: xr.Dataset) -> xr.Dataset:
    """Add the derived variables to ds. Called once per output stream."""
    from fv3_diag_files import variables

    return derived.add_derived_variables(ds, state=state, variables=variables)


def validate() -> None:
    """Check and normalise the run_config keys of every active plugin. Called before
    grid and IC generation, so a bad configuration fails early."""
    for plugin in active_plugins():
        if hasattr(plugin, "validate"):
            plugin.validate(state=state)


def stage_derived_module(*, restart: bool) -> None:
    """Copy the user's derived_vars_module to the case directory (cold start). The
    variables themselves are computed per output stream by add_derived_variables."""
    derived.apply(state=state, restart=restart)


def apply_pamip(*, restart: bool) -> None:
    """Stage the PAMIP external files and boundary data, then patch the namelists."""
    import grib_io
    import latlon_to_fv3
    from fv3_fixed_files import update_fixed_files
    from fv3_namelists import patch_nml

    pamip.apply(
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
        cp=fv3_utils.cp,
    )


def apply_sm_perturbations(*, restart: bool) -> None:
    """Perturb the surface restart files (cold start and each later segment)."""
    import latlon_to_fv3

    sm_perturbations.apply(
        state=state,
        restart=restart,
        cp=fv3_utils.cp,
        segment_hours=fv3_utils.segment_hours,
        load_grid=latlon_to_fv3.load_grid,
        sfc_filename=latlon_to_fv3.sfc_filename,
        to_fv3_grid=latlon_to_fv3.to_fv3_grid,
    )


APPLY = {
    "derived": stage_derived_module,
    "pamip": apply_pamip,
    "sm_perturbations": apply_sm_perturbations,
}


def do_plugins(*, restart: bool) -> None:
    """Last step of a cold start and of every restart segment: run the active plugins
    in the order of plugins.__all__."""
    for plugin in active_plugins():
        APPLY[plugin.__name__.rsplit(".", 1)[-1]](restart=restart)
