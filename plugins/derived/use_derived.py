"""Built-in derived variables plus the functions of the user's derived_vars_module."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import xarray as xr


def user_module_path(state) -> Path | None:
    if not state.derived_vars_module:
        return None
    return Path(os.path.expandvars(state.derived_vars_module)).expanduser().resolve()


@lru_cache
def load_user_module(path: Path) -> ModuleType:
    """Load the file under a private name. Its directory is appended to sys.path,
    so helper files next to it import normally and cannot shadow the standard
    library."""
    sys.path.append(str(path.parent))
    spec = importlib.util.spec_from_file_location("pyufs_derived_user", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(*, state) -> None:
    if not state.derived_vars_module:
        state.derived_vars_module = None
        return
    path = user_module_path(state)
    if not path.is_file():
        raise FileNotFoundError(
            f"`derived_vars_module` is {state.derived_vars_module!r}, which "
            f"resolves to {path}."
        )


def apply(*, state, restart: bool, **_) -> None:
    """Keep a copy of the user file in the case directory (cold start)."""
    if restart:
        return
    shutil.copyfile(
        user_module_path(state), Path(state.case_dir) / "derived_vars_user.py"
    )


def add_derived_variables(
    ds: xr.Dataset, *, state, variables: dict[str, str]
) -> xr.Dataset:
    """Add the built-in derived variables, then those of the user file. Called once
    per output stream."""
    path = user_module_path(state)
    from plugins.derived.derived_vars import derive as builtins

    funcs = list(builtins)
    if path is not None:
        funcs += load_user_module(path).derive
    for func in funcs:
        ds = func(ds, state=state, variables=variables)

    return ds
