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

script_dir = Path(__file__).resolve()
machine_cfg = script_dir.parent.parent / "configs" / "machine_config.yaml"
user_sbatch_file = Path.cwd() / "run_config.yaml"

if not user_sbatch_file.exists():
    print(f"ERROR: File not found: {user_sbatch_file}")
    sys.exit(0)


def read_yaml_as_file(file_path, line_no):
    with open(file_path, "r") as f:
        v = f.readlines()[line_no - 1].strip()
        return v, len(v)


def read_yaml(file_path):
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        if hasattr(e, "problem_mark"):
            mark = e.problem_mark
            v, n = read_yaml_as_file(file_path, mark.line)
            print(
                "ERROR: Error in run_config.yaml\n",
                f"File path: {file_path}\n",
                f"Line: {mark.line},  Column: {mark.column}, {e.problem}\n",
                f"\t-> {v}\n",
                f"\t   {'^' * n}",
            )
        else:
            print(f"ERROR: Invalid YAML file: {file_path}")
        sys.exit(0)
    return data


def get_sbatch_cfg():
    user_data = read_yaml(user_sbatch_file)
    default_sbatch = read_yaml(machine_cfg)["sbatch"]
    user_sbatch = user_data.get("sbatch", {})

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
    sbatch_multi_node = 1 if sbatch_nnodes > 1 else 0

    if sbatch_time > 48:
        sbatch_time = 48

    sbatch_time = f"{sbatch_time}:00:00"

    sbatch_cfg = [
        f"export SBATCH_MEM={sbatch_mem}",
        f"export SBATCH_TIME={sbatch_time}",
        f"export SBATCH_NNODES={sbatch_nnodes}",
        f"export SBATCH_OUTPUT={sbatch_output}",
        f"export SBATCH_PARTITION={sbatch_partition}",
        f"export SBATCH_NTASKS={sbatch_ntasks_total}",
        f"export SBATCH_MULTI_NODE={sbatch_multi_node}",
        f"export SBATCH_MEM_PER_CPU={sbatch_mem_per_cpu}",
        f"export SBATCH_EXCLUSIVE_NODE={sbatch_exclusive}",
        f"export SBATCH_CPUS_PER_TASK={sbatch_cpu_per_task}",
        f"export SBATCH_NODE_CONSTRAINT={sbatch_constraint}",
        f"export SBATCH_NTASKS_PER_NODE={sbatch_ntasks_per_node}",
    ]
    return sbatch_cfg


def get_run_cfg():
    user_data = read_yaml(user_sbatch_file)
    misc_cfg = [
        f"export N_ENSEMBLES={user_data.get('n_ensembles', 1)}",
        f"export RESUBMIT={user_data.get('resubmit', 0)}",
        f"export ARCHIVE_DATA={int(user_data.get('archive_data', False))}",
        f"export CASE_NAME={user_data.get('case_name') or os.environ.get('CASE_DIR', f'{Path.cwd().name}')}",
    ]

    return misc_cfg


def get_directories():
    user_data = read_yaml(machine_cfg)
    dirs = user_data.get("directories", {})

    for k in {
        "jobtmp",
        "scratch",
        "case_root",
        "shield_root",
        "fix_dir",
        "ufs_utils",
        "archive_root",
    }:
        if k not in dirs:
            print(f"ERROR: Missing `directories` configuration: {k} in {machine_cfg}")
            sys.exit(0)

    for key, value in dirs.items():
        dirs[key] = str(Path(os.path.expandvars(value)))
        try:
            Path(dirs[key]).mkdir(parents=True, exist_ok=True)
        except Exception:
            print(f"ERROR: Failed to create directory {dirs[key]}")
            sys.exit(0)

    directories = [
        f"export JOB_TMP={dirs['jobtmp']}",
        f"export SCRATCH={dirs['scratch']}",
        f"export CASE_ROOT={dirs['case_root']}",
        f"export SHIELD_ROOT={dirs['shield_root']}",
        f"export FIX_DIR={dirs['fix_dir']}",
        f"export UFS_UTILS={dirs['ufs_utils']}",
        f"export ARCHIVE_ROOT={dirs['archive_root']}",
    ]
    return directories


def get_containers():
    user_data = read_yaml(machine_cfg)
    containers = user_data.get("containers", {})

    for k in {
        "shield",
        "fregrid",
        "preprocess",
        "containers_dir",
        "container_bindpath",
    }:
        if k not in containers:
            print(f"ERROR: Missing `containers` configuration: {k} in {machine_cfg}")
            sys.exit(0)

    for key, value in containers.items():
        if key == "container_bindpath":
            continue
        containers[key] = str(Path(os.path.expandvars(value)))

    shield_path = containers["shield"]
    frehrid_path = containers["fregrid"]
    preprocess_path = containers["preprocess"]
    containers_dir = containers["containers_dir"]
    container_bindpath = containers["container_bindpath"]

    if isinstance(container_bindpath, list):
        container_bindpath = ",".join(container_bindpath)
        container_bindpath = base64.b64encode(
            container_bindpath.encode("utf-8")
        ).decode("utf-8")

    container_cfg = [
        f"export SHIELD_SIF={shield_path}",
        f"export FREGRID_SIF={frehrid_path}",
        f"export PREPROCESS_SIF={preprocess_path}",
        f"export CONTAINERS_DIR={containers_dir}",
        f"export CONTAINER_BINDPATH={container_bindpath}",
    ]

    return container_cfg


def write_cfg(sbatch_cfg):

    temp_file = Path("/tmp") / f"{uuid.uuid4()}"
    with open(temp_file, "w") as f:
        f.write("\n".join(sbatch_cfg))
    return temp_file


def main():
    sbatch_cfg = get_sbatch_cfg()
    misc_cfg = get_run_cfg()
    directory_cfg = get_directories()
    container_cfg = get_containers()
    return write_cfg([*sbatch_cfg, *misc_cfg, *directory_cfg, *container_cfg])


if __name__ == "__main__":
    file = main()
    print(file, flush=True)
