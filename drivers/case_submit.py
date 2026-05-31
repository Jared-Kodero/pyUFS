#!/usr/bin/python

# case_submit.py
import base64
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CASE.SUBMIT")

try:
    import yaml
except ImportError:
    logger.error("PyYAML is not installed in the current Python environment")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve()
MACHINE_CFG_PATH = SCRIPT_DIR.parent.parent / "configs" / "machine_config.yaml"
RUN_CFG_PATH = Path.cwd() / "run_config.yaml"

for f in (MACHINE_CFG_PATH, RUN_CFG_PATH):
    if not f.exists():
        logger.error(f"File not found: {f}")
        sys.exit(1)


def get_paths(cfg: dict):

    paths = cfg.get("paths", {})

    for k in (
        "jobtmp",
        "scratch",
        "case_root",
        "fix_dir",
        "ufs_utils",
        "archive_root",
        "shield",
        "fregrid",
        "preprocess",
        "containers_root",
        "container_bindpath",
    ):
        if k not in paths or not paths[k]:
            logger.error(f"Missing `paths` configuration: {k} in {MACHINE_CFG_PATH}")
            sys.exit(1)

    for k, v in paths.items():
        if k == "container_bindpath":
            if isinstance(paths[k], list):
                paths[k] = ",".join(paths[k])
                paths[k] = base64.b64encode(paths[k].encode("utf-8")).decode("utf-8")
            continue
        paths[k] = str(Path(os.path.expandvars(v)))
        if k in ("jobtmp", "scratch", "case_root", "archive_root"):
            if not Path(paths[k]).exists():
                Path(paths[k]).mkdir(parents=True, exist_ok=True)

    cfg = {
        "JOBTMP_DIR": paths["jobtmp"],
        "SCRATCH_DIR": paths["scratch"],
        "CASE_ROOT_DIR": paths["case_root"],
        "FIX_DIR": paths["fix_dir"],
        "UFS_UTILS_DIR": paths["ufs_utils"],
        "ARCHIVE_ROOT_DIR": paths["archive_root"],
        "SHIELD_SIF": paths["shield"],
        "FREGRID_SIF": paths["fregrid"],
        "PREPROCESS_SIF": paths["preprocess"],
        "CONTAINERS_DIR": paths["containers_root"],
        "CONTAINER_BINDPATH": paths["container_bindpath"],
    }
    return cfg


def get_sbatch_runtime_flags(cfg: dict) -> dict:
    nnodes = cfg["SBATCH_NNODES"]
    ntasks_per_node = cfg["SBATCH_NTASKS_PER_NODE"]
    exclusive = cfg["SBATCH_EXCLUSIVE_NODE"]
    use_constraint = cfg["SBATCH_NODE_CONSTRAINT"]
    sbatch_ntasks = cfg["SBATCH_NTASKS"]

    mem = cfg["SBATCH_MEM"]

    if mem > sbatch_ntasks * 2:  # at least 2GB per task
        mem_per_cpu = mem // sbatch_ntasks
    else:
        mem_per_cpu = None
        mem = None

    if nnodes > 1:
        if mem_per_cpu is not None:
            memory_flag = f"--mem-per-cpu={mem_per_cpu}g"
        else:
            memory_flag = ""
        multi_node = 1
    else:
        if mem is not None:
            memory_flag = f"--mem={mem}g"
        else:
            memory_flag = ""
        multi_node = 0

    node_constraint_flag = ""  # --constraint="
    if use_constraint == 1:
        tasks = (24, 32, 48, 64)
        constraint = next(
            (f"{c}core" for c in tasks if ntasks_per_node <= c), "192core"
        )

        node_constraint_flag = f"--constraint={constraint}"

        if ntasks_per_node <= 64:
            exclusive = 1

    exclusive_flag = "--exclusive" if exclusive == 1 else ""

    flags = {
        "SBATCH_MEMORY_FLAG": memory_flag,
        "SBATCH_EXCLUSIVE_NODE": exclusive,
        "SBATCH_MULTI_NODE_FLAG": multi_node,
        "SBATCH_NODE_CONSTRAINT_FLAG": node_constraint_flag,
        "SBATCH_NODE_EXCLUSIVE_FLAG": exclusive_flag,
    }
    return flags


def read_yaml(path: Path):

    def _read_yaml_txt(path: Path, line_no: int):
        with open(path, "r") as f:
            v = f.readlines()[line_no - 1].strip()
            return v, len(v)

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        if hasattr(e, "problem_mark"):
            mark = e.problem_mark
            v, n = _read_yaml_txt(path, mark.line)
            print(
                "ERROR: Bad Yaml file ! \n",
                f"File path: {path}\n",
                f"Line: {mark.line},  Column: {mark.column}, {e.problem}\n",
                f"\t-> {v}\n",
                f"\t   {'^' * n}",
            )
        else:
            logger.error(f"Invalid YAML file: {path}")
        sys.exit(1)
    return data


def get_config():
    user_cfg = read_yaml(RUN_CFG_PATH)
    mach_cfg = read_yaml(MACHINE_CFG_PATH)

    default_sbatch = mach_cfg.get("sbatch", {})
    user_sbatch = user_cfg.get("sbatch", {})
    cfg = {**default_sbatch, **user_sbatch}

    sbatch_time = cfg["time"]
    sbatch_nnodes = max(cfg["nnodes"], 1)
    sbatch_ntasks = max(cfg["ntasks"], 36)
    sbatch_output = cfg["output"]
    sbatch_partition = cfg["partition"]
    sbatch_exclusive = int(cfg["exclusive"])
    sbatch_constraint = int(cfg["constraint"])
    sbatch_cpu_per_task = max(cfg["cpus_per_task"], 1)
    sbatch_mem = cfg.get("mem", 0)

    sbatch_ntasks_per_node = sbatch_ntasks // sbatch_nnodes
    sbatch_ntasks_total = sbatch_ntasks_per_node * sbatch_nnodes

    if sbatch_time > 48:
        sbatch_time = 48

    sbatch_time = f"{sbatch_time}:00:00"
    n_ensembles = user_cfg.get("n_ensembles", 0)
    resubmit = user_cfg.get("resubmit", 0)
    archive_data = int(user_cfg.get("archive_data", False))
    env_case_name = os.environ.get("CASE_NAME", Path.cwd().name)
    case_name = user_cfg.get("case_name") or env_case_name
    skip_ensembles = user_cfg.get("skip_ensembles", None)

    if not isinstance(skip_ensembles, list):
        skip_ensembles = [skip_ensembles] if skip_ensembles is not None else []

    paths = get_paths(mach_cfg)

    env = {
        "SBATCH_MEM": sbatch_mem,
        "SBATCH_TIME_LIMIT": sbatch_time,
        "SBATCH_NNODES": sbatch_nnodes,
        "SBATCH_OUTPUT": sbatch_output,
        "SBATCH_PARTITION": sbatch_partition,
        "SBATCH_NTASKS": sbatch_ntasks_total,
        "SBATCH_EXCLUSIVE_NODE": sbatch_exclusive,
        "SBATCH_CPUS_PER_TASK": sbatch_cpu_per_task,
        "SBATCH_NODE_CONSTRAINT": sbatch_constraint,
        "SBATCH_NTASKS_PER_NODE": sbatch_ntasks_per_node,
        "CASE_ENSEMBLES": n_ensembles,
        "CASE_SKIP_ENSEMBLES": skip_ensembles,
        "CASE_RESUBMIT": resubmit,
        "CASE_ARCHIVE": archive_data,
        "CASE_NAME": case_name,
        **paths,
    }

    env.update(get_sbatch_runtime_flags(env.copy()))

    return env


def run(sbatch_script: Path, proc_env: dict, cwd: Path) -> int:
    try:
        subprocess.run(
            ["bash", str(sbatch_script)],
            env=proc_env,
            cwd=str(cwd),
            check=False,
        )

    except subprocess.SubprocessError as e:
        logger.error(f"Job submission failed! {e}")
        sys.exit(1)


def main():
    env = get_config()
    case_pwd = Path.cwd()
    case_dir = case_pwd.name
    case_parent_dir = case_pwd.parent.name
    ufs_utils_dir = SCRIPT_DIR.parent.parent  # case_submit.py lives in drivers/

    env["CASE_PWD"] = str(case_pwd)
    env["CASE_DIR"] = case_dir
    env["CASE_PARENT_DIR"] = case_parent_dir
    env["UFS_UTILS_DIR"] = str(ufs_utils_dir)
    env["CASE_NAME"] = env["CASE_NAME"] or case_dir

    n_ensembles = int(env["CASE_ENSEMBLES"])
    sbatch_output = Path(env["SBATCH_OUTPUT"])
    sbatch_script = ufs_utils_dir / "drivers" / "sbatch.sh"
    skipped_ensembles = env["CASE_SKIP_ENSEMBLES"]

    jobs = [i for i in range(n_ensembles)]

    if not jobs:
        ensemble_id = 0
        slurm_job_name = f"{case_parent_dir}.{case_dir}"
        case_name = env["CASE_NAME"]
        case_data_symlink = case_pwd / "run"
        case_log_file = sbatch_output.with_suffix(".log")

        iter_env = {
            **env,
            "CASE_ENSEMBLE_ID": ensemble_id,
            "SLURM_JOB_NAME": slurm_job_name,
            "SLURM_OPEN_MODE": "truncate",
            "CASE_NAME": case_name,
            "CASE_DATA_SYMLINK": str(case_data_symlink),
            "CASE_LOG_FILE": str(case_log_file),
        }
        proc_env = {**os.environ, **{k: str(v) for k, v in iter_env.items()}}
        run(sbatch_script, proc_env, case_pwd)
        logger.info("Success! Case Submitted")

    else:
        for i in jobs:
            ensemble_id = i + 1

            if ensemble_id in skipped_ensembles:
                logger.info(f"Skipped ensemble: {ensemble_id}")
                continue

            run_link = case_pwd / "run"
            if run_link.is_symlink() or run_link.exists():
                run_link.unlink()
            mem_id = f"{ensemble_id:02d}"
            slurm_job_name = f"{case_parent_dir}.{case_dir}.MEM{mem_id}"
            case_name = f"{env['CASE_NAME']}/mem{mem_id}"
            case_data_symlink = case_pwd / f"mem{mem_id}"
            case_log_file = sbatch_output.with_suffix(f".{mem_id}.log")

            iter_env = {
                **env,
                "CASE_ENSEMBLE_ID": ensemble_id,
                "SLURM_JOB_NAME": slurm_job_name,
                "SLURM_OPEN_MODE": "truncate",
                "CASE_NAME": case_name,
                "CASE_DATA_SYMLINK": str(case_data_symlink),
                "CASE_LOG_FILE": str(case_log_file),
            }

            proc_env = {**os.environ, **{k: str(v) for k, v in iter_env.items()}}
            run(sbatch_script, proc_env, case_pwd)
            logger.info(f"Submitted ensemble {ensemble_id}/{n_ensembles}")

        logger.info("Success! Case Submitted")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
