import re
from pathlib import Path

import yaml
from fv3_runtime import get_stream_handles
from fv3_state import state
from fv3_utils import cp


def field_table_source() -> tuple[Path, bool]:
    """Field table to stage and whether it is in YAML format.

    A case-local field_table.yaml takes precedence over a case-local legacy
    ASCII field_table; without either, configs/field_table.yaml is used. The
    format sets field_manager_nml use_field_table_yaml.
    """
    for name, is_yaml in (("field_table.yaml", True), ("field_table", False)):
        path = state.run_dir / name
        if path.exists():
            return path, is_yaml
    return state.configs / "field_table.yaml", True


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

    with open(src) as f:
        table = yaml.safe_load(f)

    table["title"] = str(state.description)
    table["base_date"] = f"{dt.year} {dt.month} {dt.day} {dt.hour} 0 0"
    for diag_file in table["diag_files"]:
        diag_file["file_name"] = f"HIST/{diag_file['file_name']}.{restart_no:02d}"

    with open(state.work_dir / "diag_table.yaml", "w") as f:
        yaml.safe_dump(table, f, sort_keys=False)


def update_legacy_diag(restart_no: int):
    """Stage and update the classic/legacy ASCII diagnostic table."""
    (state.work_dir / "diag_table").unlink(missing_ok=True)
    dt = state.init_datetime

    user_diag = state.run_dir / "diag_table"
    # Month and year segments default to monthly means instead of hourly output.
    monthly = state.run_length_units in ("months", "years")
    template_diag = state.configs / ("diag_table.monthly" if monthly else "diag_table")
    diag_table_path = state.work_dir / "diag_table"

    diag_file = user_diag if user_diag.exists() else template_diag
    cp(diag_file, diag_table_path)

    streams = get_stream_handles()

    with open(diag_table_path) as f:
        lines = [
            line for line in f if line.strip() and not line.strip().startswith("#")
        ]

    # The first two entries of a legacy diag_table are the title and the base
    # date (FMS diag_manager), given either literally or as the DESCRIPTION
    # and DATETIME placeholders. Both are rewritten for this case.
    if len(lines) < 2:
        raise ValueError(f"{diag_file}: missing title and base date lines")
    lines = lines[2:]
    dt_str = f"{dt.year} {dt.month:02d} {dt.day:02d} {dt.hour:02d} 0 0\n"
    out = [f"{state.description}\n", dt_str]

    for line in lines:
        names = [c.strip().strip("\"'") for c in line.split(",")]
        is_file = names[0] in streams
        if not is_file and not (len(names) > 3 and names[3] in streams):
            continue
        stream = names[0] if is_file else names[3]
        # Exact quoted match, so a handle that prefixes another is not touched.
        pattern = rf"([\"']){re.escape(stream)}\1"
        out.append(re.sub(pattern, rf"\g<1>HIST/{stream}.{restart_no:02d}\g<1>", line))

    with open(diag_table_path, "w") as f:
        f.writelines(out)


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
