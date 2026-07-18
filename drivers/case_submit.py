#!/usr/bin/python

# case_submit.py
import base64
import logging
import os
import subprocess
import sys
from difflib import get_close_matches
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Case.Submit")

try:
    import yaml
except ImportError:
    logger.error("PyYAML is not installed in the current Python environment")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve()
DEFAULT_CFG_PATH = SCRIPT_DIR.parent.parent / "configs" / "run_config.yaml"
RUN_CFG_PATH = Path.cwd() / "run_config.yaml"

if not RUN_CFG_PATH.exists():
    logger.error(f"File not found: {RUN_CFG_PATH}")
    sys.exit(1)


with open(DEFAULT_CFG_PATH, "r") as f:
    DEFAULT_CFG = yaml.safe_load(f)
DEFAULT_KEYS = DEFAULT_CFG.keys()


def read_yaml(path: Path):

    def _read_yaml_txt(path: Path, line_no: int):
        with open(path, "r") as f:
            v = f.readlines()[line_no - 1].strip()
            return v, len(v)

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        valid_keys = list(DEFAULT_KEYS)

        for key in data:
            if key not in DEFAULT_KEYS:
                suggestions = get_close_matches(
                    key,
                    valid_keys,
                    n=1,
                    cutoff=0.6,
                )

                message = (
                    f"ERROR: Unknown configuration key in run_config.yaml: `{key}`."
                )

                if suggestions:
                    message += f" Did you mean {suggestions[0]!r}?"

                print(message)
                print(f"\nPlease check the configuration file: {path}")
                print(
                    f"For a full list of keys and description of each key, refer to the default configuration file:\n\t{DEFAULT_CFG_PATH}"
                )
                sys.exit(1)

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


def get_paths(cfg: dict):

    paths = {}

    path_mapping = {
        "JOBTMP_DIR": "jobtmp",
        "CASE_ROOT_DIR": "case_root",
        "FIX_SRC": "fix_src",
        "UFS_UTILS_DIR": "ufs_utils",
        "ARCHIVE_ROOT_DIR": "archive_root",
        "SHIELD_SIF": "shield_image",
        "FREGRID_SIF": "fregrid_image",
        "PREPROCESS_SIF": "preprocess_image",
        "CONTAINERS_DIR": "containers_root",
        "CONTAINER_BINDPATH": "container_bindpath",
    }

    for k, v in path_mapping.items():
        value = cfg.get(v, DEFAULT_CFG[v])

        if v == "container_bindpath" and isinstance(value, list):
            value = ",".join(value)
            value = base64.b64encode(value.encode("utf-8")).decode("utf-8")
        else:
            value = str(Path(os.path.expandvars(value)))

        if v in ("jobtmp", "case_root", "archive_root"):
            if not Path(value).exists():
                Path(value).mkdir(parents=True, exist_ok=True)

        paths[k] = value

    return paths


def get_runtime_flags(cfg: dict) -> dict:
    nnodes = cfg["CASE_NNODES"]
    ntasks_per_node = cfg["CASE_NTASKS_PER_NODE"]
    exclusive = cfg["CASE_EXCLUSIVE_NODE"]
    use_constraint = cfg["CASE_NODE_CONSTRAINT"]
    n_tasks = cfg["CASE_NTASKS"]

    mem = cfg["CASE_MEM"]

    if mem > n_tasks * 2:  # at least 2GB per task
        mem_per_cpu = mem // n_tasks
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

    exclusive_flag = "--exclusive" if exclusive == 1 else ""

    flags = {
        "CASE_MEMORY_FLAG": memory_flag,
        "CASE_EXCLUSIVE_NODE": exclusive,
        "CASE_MULTI_NODE_FLAG": multi_node,
        "CASE_NODE_CONSTRAINT_FLAG": node_constraint_flag,
        "CASE_NODE_EXCLUSIVE_FLAG": exclusive_flag,
    }
    return flags


def get_config():
    user_cfg = read_yaml(RUN_CFG_PATH)
    default_cfg = read_yaml(DEFAULT_CFG_PATH)

    preprocess_grid_only = int(user_cfg.get("preprocess_grid_only", False))
    preprocess_orog_only = int(user_cfg.get("preprocess_orog_only", False))
    preprocess_only = int(user_cfg.get("preprocess_only", False))
    if preprocess_grid_only or preprocess_orog_only:
        preprocess_only = 1

    walltime = int(user_cfg.get("walltime", default_cfg.get("walltime", 24)))
    n_nodes = int(user_cfg.get("n_nodes", default_cfg.get("n_nodes", 4)))
    n_tasks = int(user_cfg.get("n_cpus", default_cfg.get("n_cpus", 192)))
    partition = user_cfg.get("partition", default_cfg.get("partition", "batch"))

    logfile = user_cfg.get("logfile", default_cfg.get("logfile", "shield_driver"))
    exclusive = int(user_cfg.get("exclusive_node", False))
    constraint = int(user_cfg.get("constraint_node", False))
    cpu_per_task = int(
        user_cfg.get("n_cpus_per_task", default_cfg.get("n_cpus_per_task", 1))
    )
    mem = int(user_cfg.get("mem", default_cfg.get("mem", 0)))

    ntasks_per_node = n_tasks // n_nodes
    ntasks_total = ntasks_per_node * n_nodes

    ensemble_run = user_cfg.get("ensemble_run", False)
    n_ensembles = user_cfg.get("n_ensembles", 0)

    if ensemble_run and n_ensembles < 1:
        logger.error(
            "Ensemble run is enabled, but n_ensembles is not set or less than 1 in run_config.yaml"
        )
        sys.exit(1)

    if not ensemble_run and n_ensembles > 0:
        logger.error(
            "Ensemble run is disabled, but n_ensembles is set to a value greater than 0 in run_config.yaml"
        )
        sys.exit(1)

    walltime = f"{walltime}:00:00"
    resubmit_max = user_cfg.get("resubmit", 0)
    archive_data = int(user_cfg.get("archive_data", False))

    env_case_name = os.environ.get("CASE_NAME", Path.cwd().name)
    case_name = user_cfg.get("case_name") or env_case_name
    skip_ensembles = user_cfg.get("skip_ensembles", None)

    if not isinstance(skip_ensembles, list):
        skip_ensembles = [skip_ensembles] if skip_ensembles is not None else []

    paths = get_paths(user_cfg)

    env = {
        "CASE_MEM": mem,
        "CASE_TIME_LIMIT": walltime,
        "CASE_NNODES": n_nodes,
        "CASE_OUTPUT": logfile,
        "CASE_PARTITION": partition,
        "CASE_NTASKS": ntasks_total,
        "CASE_EXCLUSIVE_NODE": exclusive,
        "CASE_CPUS_PER_TASK": cpu_per_task,
        "CASE_NODE_CONSTRAINT": constraint,
        "CASE_NTASKS_PER_NODE": ntasks_per_node,
        "CASE_ENSEMBLES": n_ensembles,
        "CASE_SKIP_ENSEMBLES": skip_ensembles,
        "CASE_RESUBMIT_INDEX": 0,
        "CASE_RESUBMIT_MAX": resubmit_max,
        "CASE_ARCHIVE": archive_data,
        "CASE_PREPROCESS_ONLY": preprocess_only,
        "CASE_NAME": case_name,
        **paths,
    }

    env.update(get_runtime_flags(env.copy()))

    return env


def run(script: Path, proc_env: dict, cwd: Path) -> int:
    try:
        result = subprocess.run(
            ["bash", str(script)],
            env=proc_env,
            cwd=str(cwd),
        )

    except subprocess.SubprocessError as e:
        logger.error(f"Job submission failed! {e}")
        sys.exit(1)

    if result.returncode != 0:
        logger.error(f"Job submission failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    return result.returncode


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
    logfile = Path(env["CASE_OUTPUT"])
    script = ufs_utils_dir / "drivers" / "sbatch.sh"
    skipped_ensembles = env["CASE_SKIP_ENSEMBLES"]

    jobs = [i for i in range(n_ensembles)]

    if not jobs:
        ensemble_id = 0
        slurm_job_name = f"{case_parent_dir}.{case_dir}"
        case_name = env["CASE_NAME"]
        case_data_symlink = case_pwd / "run"
        case_log_file = logfile.with_suffix(".log")

        iter_env = {
            **env,
            "TOTAL_WALLTIME_TIME": 0,
            "CASE_ENSEMBLE_ID": ensemble_id,
            "SLURM_JOB_NAME": slurm_job_name,
            "SLURM_OPEN_MODE": "truncate",
            "CASE_NAME": case_name,
            "CASE_DATA_SYMLINK": str(case_data_symlink),
            "CASE_LOG_FILE": str(case_log_file),
        }
        proc_env = {**os.environ, **{k: str(v) for k, v in iter_env.items()}}
        run(script, proc_env, case_pwd)
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
            case_log_file = logfile.with_suffix(f".{mem_id}.log")

            iter_env = {
                **env,
                "TOTAL_WALLTIME_TIME": 0,
                "CASE_ENSEMBLE_ID": ensemble_id,
                "SLURM_JOB_NAME": slurm_job_name,
                "SLURM_OPEN_MODE": "truncate",
                "CASE_NAME": case_name,
                "CASE_DATA_SYMLINK": str(case_data_symlink),
                "CASE_LOG_FILE": str(case_log_file),
            }

            proc_env = {**os.environ, **{k: str(v) for k, v in iter_env.items()}}
            run(script, proc_env, case_pwd)
            logger.info(f"Submitted ensemble {ensemble_id}/{n_ensembles}")

        logger.info("Success! Case Submitted")


if __name__ == "__main__":
    main()
