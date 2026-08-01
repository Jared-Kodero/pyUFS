import logging
import os
from pathlib import Path

from fv3_state import state

log = logging.getLogger("PREPROCESS")


def gen_shield_run_sh() -> None:
    slurm_mpi_launcher = [
        "srun",
        "--mpi=pmix",
        "--distribution=block:block",
        "--cpu-bind=cores",
        "-n",
    ]
    mpi_launcher = ["mpirun", "-np"]
    native_modules = state.modules
    native_launcher = " ".join(slurm_mpi_launcher)
    container_launcher = " ".join(mpi_launcher)

    gen_shield_container_scripts(native_modules, native_launcher, container_launcher)
    if state.restart_no == 0:
        log.info(f"Total PEs needed for run: {state.total_pes}")


def gen_shield_container_scripts(
    native_modules: list, native_launcher: str, container_launcher: str
) -> None:
    if state.multi_node and not state.shield_exe:
        raise RuntimeError(
            "Set `shield_exe` in run_config.yaml when running in multi-node mode."
        )
    os.system(f"mkdir -p {state.logs}/preprocess")
    os.system(f"mv {state.logs}/*.log {state.logs}/preprocess/")

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

        (state.work_dir / "shield.native").touch()

    else:
        cfg = {
            "log_file": log_file,
            "exe": "SHiELD_nh.prod.64bit.x",
            "modules": modules,
            "launcher": container_launcher,
        }

    write_shield_sh(
        exit_code=state.work_dir / "exit_code",
        **cfg,
    )


def write_shield_sh(
    exe: str, log_file: Path, exit_code: Path, modules: str, launcher: str
) -> None:
    template_path = state.configs / "shield.launcher"
    output_path = state.work_dir / "shield"

    # read template
    with open(template_path, "r") as f:
        content = f.read()

    # replace placeholders
    content = content.replace("__MODULES__", str(modules))
    content = content.replace("__WORK_DIR__", str(state.work_dir))
    content = content.replace("__LAUNCHER__", str(launcher))
    content = content.replace("__TOTAL_PES__", str(state.total_pes))
    content = content.replace("__EXECUTABLE__", str(exe))
    content = content.replace("__LOG_FILE__", str(log_file))
    content = content.replace("__EXIT_CODE_FILE__", str(exit_code))

    # write final script
    with open(output_path, "w") as f:
        f.write(content)

    # make executable
    os.chmod(output_path, 0o755)
