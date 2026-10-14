"""Read, validate and stage the FMS diagnostic tables.

The legacy ASCII diag_table and the native FMS diag_table.yaml are read through
AsciiDiagTable and YamlDiagTable. Every field of a table is a native
(module, field) pair of the model, for example ("gfs_sfc", "t2m"), and the
output_name is the user's own name for it in the history files. The native field
is the stable key: tables/diag_field.csv lists every pair the model registers,
check_diag_fields() validates a table against it, and variables maps each native
field name to its output name so that code downstream (derived_vars.py, the
regridder) finds a variable whatever the user called it.
"""

import csv
import difflib
from functools import lru_cache
from itertools import groupby
from pathlib import Path

import pandas as pd
import yaml

from fv3_paths import paths
from fv3_state import state
from fv3_utils import cp

DIAG_FIELD_CSV = Path(__file__).resolve().parent.parent / "tables" / "diag_field.csv"


class AsciiDiagTable:
    """Legacy ASCII diag_table as a dict with header, streams and variables.

    `variables` maps each native field name to its list of entries (a field can
    be written to several streams); every entry holds the module, field,
    output_name, stream, time_sampling, reduction_method, regional_section and
    packing columns of the table.
    """

    STREAM_KEYS = (
        "file_name",
        "output_freq",
        "output_freq_units",
        "file_format",
        "time_axis_units",
        "time_axis_name",
        "new_file_freq",
        "new_file_freq_units",
        "start_time",
        "file_duration",
        "file_duration_units",
    )
    VARIABLE_KEYS = (
        "module",
        "field",
        "output_name",
        "stream",
        "time_sampling",
        "reduction_method",
        "regional_section",
        "packing",
    )
    NUMERIC_KEYS = frozenset(
        {"output_freq", "file_format", "new_file_freq", "file_duration"}
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self.read(path)

    @classmethod
    def read(cls, path: str | Path) -> dict:
        lines = [
            line.split("#", 1)[0].strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
        ]
        lines = [line for line in lines if line]
        if len(lines) < 2:
            raise ValueError(f"{path}: missing title and base date lines")
        data = {
            "header": {"title": lines[0], "base_date": lines[1]},
            "streams": {},
            "variables": {},
        }

        for line in lines[2:]:
            values = [
                value.strip()
                for value in next(csv.reader([line], skipinitialspace=True))
                if value.strip()
            ]
            if len(values) > 1 and values[1].lstrip("+-").isdigit():
                stream = dict(zip(cls.STREAM_KEYS, values))
                data["streams"][stream.pop("file_name")] = stream
            else:
                variable = dict(zip(cls.VARIABLE_KEYS, values))
                data["variables"].setdefault(variable["field"], []).append(variable)
        return data

    @property
    def streams(self) -> dict:
        return self.data["streams"]

    @property
    def variables(self) -> dict:
        return self.data["variables"]

    @staticmethod
    def _quote(value: object) -> str:
        return '"' + str(value).replace('"', '""') + '"'

    def write(self, path: str | Path | None = None) -> None:
        header = self.data["header"]
        lines = [header["title"], header["base_date"], ""]

        for name, stream in self.streams.items():
            values = [self._quote(name)]
            for key in self.STREAM_KEYS[1:]:
                if key in stream:
                    value = stream[key]
                    values.append(
                        str(value) if key in self.NUMERIC_KEYS else self._quote(value)
                    )
            lines.append(", ".join(values))

        lines.append("")
        rows = sorted(
            (item for definitions in self.variables.values() for item in definitions),
            key=lambda item: item["module"],
        )
        for _, group in groupby(rows, key=lambda item: item["module"]):
            for variable in group:
                values = []
                for key in self.VARIABLE_KEYS:
                    value = variable[key]
                    if isinstance(value, bool):
                        values.append(".true." if value else ".false.")
                    elif key == "packing" or (
                        key == "reduction_method"
                        and str(value).lower() in (".true.", ".false.")
                    ):
                        values.append(str(value))
                    else:
                        values.append(self._quote(value))
                lines.append(", ".join(values))
            lines.append("")

        Path(path or self.path).write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )

    def stream_names(self) -> list[str]:
        """File names of the streams, in table order."""
        return list(self.streams)

    def stream_outputs(self) -> list[tuple[str, str, str]]:
        """(stream, module/field, output name) of every variable line."""
        return [
            (
                v["stream"],
                f"{v['module']}/{v['field']}",
                v.get("output_name") or v["field"],
            )
            for definitions in self.variables.values()
            for v in definitions
        ]

    def entries(self) -> list[tuple[str, str, str]]:
        """(module, native field, output name) of every variable line."""
        return [
            (v["module"], v["field"], v.get("output_name") or v["field"])
            for definitions in self.variables.values()
            for v in definitions
        ]


class YamlDiagTable:
    """Native FMS diag_table.yaml, read and written without restructuring it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self.read(path)

    @staticmethod
    def read(path: str | Path) -> dict:
        with Path(path).open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    def write(self, path: str | Path | None = None) -> None:
        with Path(path or self.path).open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.data, file, sort_keys=False, allow_unicode=True)

    def stream_names(self) -> list[str]:
        """File names of the diag_files, in table order."""
        return [d["file_name"] for d in self.data.get("diag_files") or []]

    def stream_outputs(self) -> list[tuple[str, str, str]]:
        """(stream, module/field, output name) of every varlist entry."""
        return [
            (
                d["file_name"],
                f"{v['module']}/{v['var_name']}",
                v.get("output_name") or v["var_name"],
            )
            for d in self.data.get("diag_files") or []
            for v in d.get("varlist") or []
        ]

    def entries(self) -> list[tuple[str, str, str]]:
        """(module, native field, output name) of every varlist entry.

        FMS uses var_name as the output name when output_name is not given.
        """
        return [
            (
                v["module"],
                v["var_name"],
                v.get("output_name") or v["var_name"],
            )
            for diag_file in self.data.get("diag_files") or []
            for v in diag_file.get("varlist") or []
        ]


def read_diag_table(path: str | Path) -> AsciiDiagTable | YamlDiagTable:
    """Read a diagnostic table, choosing the format from the file suffix."""
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return YamlDiagTable(path)
    return AsciiDiagTable(path)


@lru_cache(maxsize=1)
def diag_fields() -> pd.DataFrame:
    """tables/diag_field.csv: module, field, long_name and units of every field."""
    # The file starts with a byte order mark, which utf-8-sig removes.
    return pd.read_csv(DIAG_FIELD_CSV, encoding="utf-8-sig", dtype=str).fillna("")


def check_diag_fields(table: AsciiDiagTable | YamlDiagTable) -> None:
    """Raise when the table has an unknown field or a repeated output name.

    A (module, field) pair must be a model field, and an output name may be
    used once per stream.

    FMS skips a field it does not know without an error, which leaves the
    output files without the variable. The message gives the closest valid
    names, and the module that holds a field named for the wrong one.
    """
    fields = diag_fields()
    by_module = fields.groupby("module")["field"].apply(set).to_dict()
    problems = []
    for module, field, _ in table.entries():
        if module not in by_module:
            hint = difflib.get_close_matches(module, list(by_module), n=1)
            problems.append(
                f"unknown module {module!r}"
                + (f", did you mean {hint[0]!r}?" if hint else "")
            )
        elif field not in by_module[module]:
            elsewhere = sorted(m for m, f in by_module.items() if field in f)
            hint = difflib.get_close_matches(field, sorted(by_module[module]), n=3)
            msg = f"{module}/{field}: not a field of module {module!r}"
            if elsewhere:
                msg += f"; it is in {', '.join(elsewhere)}"
            elif hint:
                msg += f"; did you mean {', '.join(hint)}?"
            problems.append(msg)
    # Two variables of one stream with the same output name would write the
    # same variable twice in that file.
    seen: dict[tuple[str, str], str] = {}
    for stream, native, output in table.stream_outputs():
        if (stream, output) in seen:
            first = seen[(stream, output)]
            problems.append(
                f"output name {output!r} is used twice in stream {stream!r}: "
                f"{first} and {native}"
            )
        else:
            seen[(stream, output)] = native

    if problems:
        raise ValueError(
            f"{table.path}: invalid diagnostic fields (see tables/diag_field.csv)\n  "
            + "\n  ".join(dict.fromkeys(problems))
        )


def output_names(
    table: AsciiDiagTable | YamlDiagTable,
) -> dict[tuple[str, str], list[str]]:
    """Output names written for each native (module, field), in table order."""
    names: dict[tuple[str, str], list[str]] = {}
    for module, field, out in table.entries():
        if out not in names.setdefault((module, field), []):
            names[(module, field)].append(out)
    return names


def staged_diag_table() -> AsciiDiagTable | YamlDiagTable | None:
    """The diag_table staged in the work directory for the model, if any."""
    work_dir = Path(paths["work_dir"])
    for name in ("diag_table.yaml", "diag_table"):
        if (work_dir / name).exists():
            return read_diag_table(work_dir / name)
    return None


def stream_handles() -> list[str]:
    """File names of the streams of the staged diag_table (HIST/<handle>.<nn>)."""
    table = staged_diag_table()
    if table is None:
        raise FileNotFoundError(f"No diag_table staged in {paths['work_dir']}")
    return table.stream_names()


class _Variables(dict):
    """dict[str, str] of native field name to output name.

    A field the table does not write maps to its own name, so code that indexes
    it finds nothing in the dataset and skips its calculation instead of
    raising.
    """

    def __missing__(self, key: str) -> str:
        return key


# Native diag_table field name -> output name, for example variables["t_plev"]
# == "t". Filled by load_variables() whenever a diag_table is loaded; the object
# is only ever updated in place, so `from fv3_diag_files import variables` stays
# valid. A field written under several names, or shared by two modules (ps in
# dynamics and gfs_dyn), takes the first entry of the table.
variables: dict[str, str] = _Variables()


def load_variables(table: AsciiDiagTable | YamlDiagTable | None = None) -> dict:
    """Fill variables from a diag_table, by default the staged one, and return it.

    Empty when no table is staged, so every field keeps its native name.
    """
    table = table or staged_diag_table()
    variables.clear()
    if table is not None:
        for _, field, output in table.entries():
            variables.setdefault(field, output)
    return variables


def field_table_source() -> tuple[Path, bool]:
    """Field table to stage and whether it is in YAML format.

    A case-local field_table.yaml takes precedence over a case-local legacy
    ASCII field_table; without either, templates/field_table.yaml is used. The
    format sets field_manager_nml use_field_table_yaml.
    """
    for name, is_yaml in (("field_table.yaml", True), ("field_table", False)):
        path = state.run_dir / name
        if path.exists():
            return path, is_yaml
    return state.ufs_utils / "templates" / "field_table.yaml", True


def update_data_table():
    """Stage a case-local data table in the format the namelist selects.

    data_table.yaml is used with use_modern_diag (data_override_nml
    use_data_table_yaml); otherwise a legacy data_table. Without a case-local
    table no data override is applied.
    """
    for name in ("data_table", "data_table.yaml"):
        (state.work_dir / name).unlink(missing_ok=True)
    name = "data_table.yaml" if state.use_modern_diag else "data_table"
    if (state.run_dir / name).exists():
        cp(state.run_dir / name, state.work_dir / name)


def update_yaml_diag(restart_no: int):
    """Stage and update the modern YAML-based diagnostic table configuration."""
    (state.work_dir / "diag_table.yaml").unlink(missing_ok=True)
    dt = state.init_datetime
    src = state.run_dir / "diag_table.yaml"
    if not src.exists():
        raise FileNotFoundError(f"use_modern_diag: true requires {src}")

    table = YamlDiagTable(src)
    check_diag_fields(table)
    load_variables(table)
    table.data["title"] = str(state.description)
    table.data["base_date"] = f"{dt.year} {dt.month} {dt.day} {dt.hour} 0 0"
    for diag_file in table.data["diag_files"]:
        diag_file["file_name"] = f"HIST/{diag_file['file_name']}.{restart_no:02d}"
    table.write(state.work_dir / "diag_table.yaml")


def update_legacy_diag(restart_no: int):
    """Stage and update the classic/legacy ASCII diagnostic table.

    The title and base date (literal or the DESCRIPTION and DATETIME
    placeholders) are replaced by the case description and initialization
    time, every stream is renamed HIST/<handle>.<restart_no>, and variable
    lines that name a stream the table does not define are dropped.
    """
    (state.work_dir / "diag_table").unlink(missing_ok=True)
    dt = state.init_datetime

    user_diag = state.run_dir / "diag_table"
    # Month and year segments default to monthly means instead of hourly output.
    monthly = state.run_length_units in ("months", "years")
    template_diag = (
        state.ufs_utils
        / "templates"
        / ("diag_table.monthly" if monthly else "diag_table")
    )
    table = AsciiDiagTable(user_diag if user_diag.exists() else template_diag)
    check_diag_fields(table)
    load_variables(table)

    def tag(handle: str) -> str:
        return f"HIST/{handle}.{restart_no:02d}"

    table.data["header"] = {
        "title": str(state.description),
        "base_date": f"{dt.year} {dt.month:02d} {dt.day:02d} {dt.hour:02d} 0 0",
    }
    handles = set(table.streams)
    table.data["variables"] = {
        field: kept
        for field, entries in table.variables.items()
        if (
            kept := [
                {**e, "stream": tag(e["stream"])}
                for e in entries
                if e["stream"] in handles
            ]
        )
    }
    table.data["streams"] = {tag(name): item for name, item in table.streams.items()}
    table.write(state.work_dir / "diag_table")


def update_diag_table():
    """Dispatch to YAML or legacy diagnostic table updater based on state."""
    restart_no = state.get("restart_no", 0)

    # Clean up both possible old outputs
    for name in ("diag_table", "diag_table.yaml"):
        (state.work_dir / name).unlink(missing_ok=True)

    if state.use_modern_diag:
        update_yaml_diag(restart_no)
    else:
        update_legacy_diag(restart_no)


def update_table_files():
    """Main orchestrator to update all FMS configuration tables."""

    field_file, is_yaml = field_table_source()
    for name in ("field_table", "field_table.yaml"):
        (state.work_dir / name).unlink(missing_ok=True)
    cp(field_file, state.work_dir / ("field_table.yaml" if is_yaml else "field_table"))

    # Delegate to diagnostic and data table updaters
    update_diag_table()
    update_data_table()
