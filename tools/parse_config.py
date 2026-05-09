#!/usr/bin/python

# parse_config.py
import base64
import os
import sys
import uuid
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is not installed in the current Python environment")
    sys.exit(0)


SCRIPT_DIR = Path(__file__).resolve()
MACHINE_CFG_PATH = SCRIPT_DIR.parent.parent / "configs" / "machine_config.yaml"
RUN_CFG_PATH = Path.cwd() / "run_config.yaml"

for f in (MACHINE_CFG_PATH, RUN_CFG_PATH):
    if not f.exists():
        print(f"ERROR: File not found: {f}")
        sys.exit(0)


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
            print(f"ERROR: Missing `paths` configuration: {k} in {MACHINE_CFG_PATH}")
            sys.exit(0)

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
    mem = cfg["SBATCH_MEM"]
    mem_per_cpu = cfg["SBATCH_MEM_PER_CPU"]
    exclusive = cfg["SBATCH_EXCLUSIVE_NODE"]
    use_constraint = cfg["SBATCH_NODE_CONSTRAINT"]

    if nnodes > 1:
        memory_flag = f"--mem-per-cpu={mem_per_cpu}g"
        multi_node = 1
    else:
        memory_flag = f"--mem={mem}g"
        multi_node = 0

    node_constraint_flag = "--constraint="
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
        "SBATCH_EXCLUSIVE_NODE": exclusive,
        "SBATCH_MULTI_NODE_FLAG": multi_node,
        "SBATCH_MEMORY_FLAG": memory_flag,
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
                " ERROR: in run_config.yaml\n",
                f"File path: {path}\n",
                f"Line: {mark.line},  Column: {mark.column}, {e.problem}\n",
                f"\t-> {v}\n",
                f"\t   {'^' * n}",
            )
        else:
            print(f"ERROR: Invalid YAML file: {path}")
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
    sbatch_mem = max(cfg["mem"], sbatch_ntasks * 2)
    sbatch_cpu_per_task = max(cfg["cpus_per_task"], 1)

    sbatch_mem_per_cpu = sbatch_mem // sbatch_ntasks
    sbatch_ntasks_per_node = sbatch_ntasks // sbatch_nnodes
    sbatch_ntasks_total = sbatch_ntasks_per_node * sbatch_nnodes

    if sbatch_time > 48:
        sbatch_time = 48

    sbatch_time = f"{sbatch_time}:00:00"
    n_ensembles = user_cfg.get("n_ensembles", 1)
    resubmit = user_cfg.get("resubmit", 0)
    archive_data = int(user_cfg.get("archive_data", False))
    env_case_name = os.environ.get("CASE_NAME", Path.cwd().name)
    case_name = user_cfg.get("case_name") or env_case_name
    paths = get_paths(mach_cfg)

    env = {
        "SBATCH_MEM": sbatch_mem,
        "SBATCH_TIME": sbatch_time,
        "SBATCH_NNODES": sbatch_nnodes,
        "SBATCH_OUTPUT": sbatch_output,
        "SBATCH_PARTITION": sbatch_partition,
        "SBATCH_NTASKS": sbatch_ntasks_total,
        "SBATCH_MEM_PER_CPU": sbatch_mem_per_cpu,
        "SBATCH_EXCLUSIVE_NODE": sbatch_exclusive,
        "SBATCH_CPUS_PER_TASK": sbatch_cpu_per_task,
        "SBATCH_NODE_CONSTRAINT": sbatch_constraint,
        "SBATCH_NTASKS_PER_NODE": sbatch_ntasks_per_node,
        "CASE_ENSEMBLES": n_ensembles,
        "CASE_RESUBMIT": resubmit,
        "CASE_ARCHIVE": archive_data,
        "CASE_NAME": case_name,
        **paths,
    }

    env.update(get_sbatch_runtime_flags(env.copy()))

    return env


def write_cfg(cfg: dict):

    temp_file = Path("/tmp") / f"{uuid.uuid4()}"
    with open(temp_file, "w") as f:
        for k, v in cfg.items():
            f.write(f"export {k}={v}\n")

    return temp_file


def main():
    cfg = get_config()
    return write_cfg(cfg)


if __name__ == "__main__":
    file = main()
    print(file, flush=True)
