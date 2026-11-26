"""YAML reading, writing and include resolution for the launchers and the workflow.

Only the standard library and PyYAML are imported here: the launchers
(``case_submit.py``, ``case_continue.py``) load this module on the login node,
before any workflow environment exists, so it must not import xarray, pandas or
the path and state modules.

Errors are raised, never reported with ``sys.exit``: the caller decides whether
a bad file ends the program (the launchers).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("fv3_yaml")

__all__ = [
    "YamlError",
    "dump_yaml",
    "load_config",
    "open_yaml",
    "resolve_includes",
]


class YamlError(ValueError):
    """A YAML file that cannot be parsed or has the wrong structure.

    ``str(error)`` is a message that names the file, the line and column and
    shows the offending line with a caret, ready to print.
    """

    def __init__(
        self,
        path: Path | str,
        problem: str,
        line: int | None = None,
        column: int | None = None,
        source: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.problem = problem
        self.line = line
        self.column = column
        self.source = source
        super().__init__(self._message())

    def _message(self) -> str:
        text = f"Bad YAML file: {self.path}"
        if self.line is not None and self.column is not None:
            text += f"\nLine: {self.line}, Column: {self.column}, {self.problem}"
            if self.source is not None:
                text += f"\n\t-> {self.source}\n\t   {' ' * (self.column - 1)}^"
        else:
            text += f"\n{self.problem}"
        return text


def locate_err(error: yaml.YAMLError, path: Path) -> YamlError:
    """Convert a PyYAML error to a YamlError that carries the position."""
    mark = getattr(error, "problem_mark", None)
    problem = getattr(error, "problem", None) or str(error)
    if mark is None:
        return YamlError(path, problem)
    try:
        lines = path.read_text(encoding="utf-8").splitlines() or [""]
        source = lines[min(mark.line, len(lines) - 1)]
    except (OSError, UnicodeDecodeError):
        source = None
    return YamlError(path, problem, mark.line + 1, mark.column + 1, source)


def open_yaml(path: Path | str) -> Any:
    """Parse a YAML file with ``yaml.safe_load``; an empty file gives ``{}``.

    Raises YamlError (a ValueError) for a syntax error and OSError when the file
    cannot be read.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise locate_err(error, path) from error
    return {} if data is None else data


def resolve_includes(
    conf: dict,
    conf_path: Path | str,
    *,
    stack: frozenset[Path] = frozenset(),
    loaded: list[str] | None = None,
) -> dict:
    """Merge the files named by the ``include`` key of ``conf`` into it.

    ``include`` is a path or a list of paths, expanded for ``$VARS`` and ``~``
    and taken relative to the directory of ``conf_path``; it may be nested.
    Keys of ``conf`` win over included keys. Two includes that set the same
    key to different values raise ValueError, as does a circular include. An
    include that does not exist is skipped with a warning. ``conf`` itself is
    not modified. The resolved path of every included file is appended to
    ``loaded`` when it is given.
    """
    conf = dict(conf)
    conf_path = Path(os.path.expandvars(str(conf_path))).expanduser().resolve()
    stack |= {conf_path}
    include_paths = conf.pop("include", None)
    if isinstance(include_paths, str):
        include_paths = [include_paths]

    base: dict = {}
    for path in include_paths or []:
        include_path = Path(os.path.expandvars(str(path))).expanduser()
        include_path = (conf_path.parent / include_path).resolve()

        if include_path in stack:
            raise ValueError(f"Circular include of {include_path} in {conf_path}")
        if not include_path.exists():
            log.warning(
                "Include %s of %s does not exist; skipped", include_path, conf_path
            )
            continue
        if loaded is not None:
            loaded.append(str(include_path))
        included = resolve_includes(
            open_yaml(include_path) or {}, include_path, stack=stack, loaded=loaded
        )
        conflicts = [k for k in base.keys() & included.keys() if base[k] != included[k]]
        if conflicts:
            raise ValueError(
                f"Key(s) {conflicts} in {include_path} are already set by an earlier include of {conf_path}"
            )
        base.update(included)
    return base | conf


def load_config(
    path: Path | str,
    loaded: list[str] | None = None,
) -> dict:
    """Read a configuration file and resolve its includes.

    The resolved paths of the included files are appended to ``loaded`` when it
    is given. Raises YamlError when the file is not a mapping at the top level.
    """
    path = Path(path)
    conf = open_yaml(path) or {}
    if not isinstance(conf, dict):
        raise YamlError(path, "the top level must be a mapping of keys to values")
    return resolve_includes(conf, path, loaded=loaded)


def dump_yaml(
    data: Any, path: Path | str, *, atomic: bool = True, **dump_kwargs: Any
) -> None:
    """Write ``data`` as YAML; keys keep their order unless ``sort_keys`` is given.

    With ``atomic`` the text goes to a temporary file in the same directory
    that then replaces ``path``, so a reader never sees a half-written file and
    an interrupted write leaves the old file. ``dump_kwargs`` go to
    ``yaml.safe_dump`` (``default_flow_style``, ``allow_unicode``, ...).
    """
    path = Path(path)
    dump_kwargs.setdefault("sort_keys", False)
    target = path.with_name(f".{path.name}.tmp") if atomic else path
    try:
        with target.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, **dump_kwargs)
        if atomic:
            target.replace(path)
    except BaseException:
        if atomic:
            target.unlink(missing_ok=True)
        raise
