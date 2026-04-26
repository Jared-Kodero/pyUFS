import logging
import os

import yaml
from fv3gfs_state import state

log = logging.getLogger("UFS_UTILS")


def gen_shield_run_sh() -> None:
    machine_settings = state.configs / "machine_config.yaml"
    with open(machine_settings, "r") as f:
        mach_settings = yaml.safe_load(f)

    native_modules = mach_settings["modules"]
    native_launcher = " ".join(mach_settings["launchers"]["srun"])
    container_launcher = " ".join(mach_settings["launchers"]["mpirun"])

    gen_shield_container_scripts(native_modules, native_launcher, container_launcher)
    if state.restart_no == 0:
        log.info(f"Total PEs needed for run: {state.total_pes}")


def gen_shield_container_scripts(
    native_modules, native_launcher, container_launcher
) -> None:
    if state.multi_node and not state.shield_exe:
        raise RuntimeError(
            "Set `shield_exe` in run_config.yaml when running in multi-node mode."
        )

    restart_no = state.get("restart_no", 0)
    log_file = state.logs / f"shield_{restart_no:03d}.log"
    modules = ""

    if state.shield_exe:
        modules = "\n".join(f"module load {m}" for m in native_modules)

        cfg = dict(
            log_file=log_file,
            exe=state.shield_exe,
            modules=modules,
            launcher=native_launcher,
        )

        (state.home / "shield.native").touch()

    else:
        cfg = dict(
            log_file=log_file,
            exe="SHiELD_nh.prod.64bit.x",
            modules=modules,
            launcher=container_launcher,
        )

    write_shield_sh(
        exit_code=state.home / "exit_code",
        **cfg,
    )


def write_shield_sh(exe, log_file, exit_code, modules, launcher) -> None:
    template_path = state.configs / "shield.launcher"
    output_path = state.home / "shield"

    # read template
    with open(template_path, "r") as f:
        content = f.read()

    # replace placeholders
    content = content.replace("__MODULES__", str(modules))
    content = content.replace("__RUNDIR__", str(state.home))
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
