import re

import yaml
from fv3_runtime import get_stream_handles, log
from fv3_state import state
from fv3_utils import cp


def update_data_table():
    """Stage the data_table.yaml if it exists in the run directory."""
    (state.work_dir / "data_table.yaml").unlink(missing_ok=True)
    if (state.run_dir / "data_table.yaml").exists():
        cp(state.run_dir / "data_table.yaml", state.work_dir / "data_table.yaml")


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
    log.info(f"diag_table: {diag_file}")
    cp(diag_file, diag_table_path)

    streams = get_stream_handles()

    with open(diag_table_path) as f:
        lines = f.readlines()
        lines = [line for line in lines if line and not line.strip().startswith("#")]

        skipped_header = 0
        if lines[0].strip().lower() == "DESCRIPTION".strip().lower():
            lines = lines[1:]  # Skip the header line
            skipped_header += 1
        if lines[0].strip().lower() == "DATETIME".strip().lower():
            lines = lines[1:]  # Skip the second header line
            skipped_header += 1

    out = []
    if skipped_header == 2:
        dt_str = f"{dt.year} {dt.month:02d} {dt.day:02d} {dt.hour:02d} 0 0\n"
        desc_str = f"{state.description}\n"
        out = [desc_str, dt_str]

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
        if (state.run_dir / "data_table.yaml").exists():
            cp(state.run_dir / "data_table.yaml", state.work_dir / "data_table.yaml")
    else:
        update_legacy_diag(restart_no)


def update_table_files():
    """Main orchestrator to update all FMS configuration tables."""

    # Handle field_table staging
    user_field = state.run_dir / "field_table"
    template_field = state.configs / "field_table.yaml"
    field_file = user_field if user_field.exists() else template_field
    field_table_path = state.work_dir / "field_table.yaml"

    (state.work_dir / "field_table.yaml").unlink(missing_ok=True)
    cp(field_file, field_table_path)

    # Delegate to diagnostic and data table updaters
    update_diag_table()
    update_data_table()
