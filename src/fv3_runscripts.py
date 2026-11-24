import logging
import os
from pathlib import Path

from fv3_runtime import SRUN
from fv3_state import state

log = logging.getLogger("PREPROCESS")


def gen_shield_run_sh() -> None:
    mpi_launcher = ["mpirun", "-np"]
    native_modules = state.modules
    native_launcher = " ".join([*SRUN, "-n"])
    container_launcher = " ".join(mpi_launcher)

    gen_shield_container_scripts(native_modules, native_launcher, container_launcher)


def gen_shield_container_scripts(
    native_modules: list, native_launcher: str, container_launcher: str
) -> None:
    if state.multi_node and not state.shield_exe:
        raise RuntimeError(
            "Set `shield_exe` in run_config.yaml when running in multi-node mode."
        )
    preprocess_logs = state.logs / "preprocess"
    preprocess_logs.mkdir(parents=True, exist_ok=True)
    for log_path in state.logs.glob("*.log"):
        log_path.replace(preprocess_logs / log_path.name)

    restart_no = state.get("restart_no", 0)
    log_file = state.logs / "shield" / f"shield_{restart_no:03d}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    modules = ""

    if state.shield_exe:
        modules = "\n".join(f"module load {m}" for m in native_modules)

        cfg = {
            "log_file": log_file,
            "exe": state.shield_exe,
            "modules": modules,
            "launcher": native_launcher,
        }

        (state.case_dir / "shield.native").touch()

    else:
        cfg = {
            "log_file": log_file,
            "exe": "SHiELD_nh.prod.64bit.x",
            "modules": modules,
            "launcher": container_launcher,
        }

    write_shield_sh(
        exit_code=state.case_dir / "exit_code",
        **cfg,
    )


def write_shield_sh(
    exe: str, log_file: Path, exit_code: Path, modules: str, launcher: str
) -> None:
    template_path = state.pyufs / "launchers" / "shield.launcher"
    output_path = state.case_dir / "shield"

    # read template
    with open(template_path, "r") as f:
        content = f.read()

    # replace placeholders
    segment_label = "Run"
    if state.resubmit > 0:
        segment_label = f"Segment {state.restart_no}"

    # replace placeholders
    content = content.replace("__MODULES__", str(modules))
    content = content.replace("__CASE_DIR__", str(state.case_dir))
    content = content.replace("__LAUNCHER__", str(launcher))
    content = content.replace("__TOTAL_PES__", str(state.total_pes))
    content = content.replace("__EXECUTABLE__", str(exe))
    content = content.replace("__LOG_FILE__", str(log_file))
    content = content.replace("__EXIT_CODE_FILE__", str(exit_code))
    content = content.replace("__SEG_LABEL__", str(segment_label))

    # write final script
    with open(output_path, "w") as f:
        f.write(content)

    # make executable
    os.chmod(output_path, 0o755)
