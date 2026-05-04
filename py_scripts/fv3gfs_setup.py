# config.py

from __future__ import annotations

import re
from pathlib import Path

import yaml
from fv3gfs_nesting import nest_info, validate_nests
from fv3gfs_paths import configure_directories, paths
from fv3gfs_restart_driver import check_prev_state
from fv3gfs_state import FV3State, load_state, log, logger, state
from fv3gfs_utils import (
    cres_to_deg,
    parse_datetime,
    parse_resolution,
)

run_logs = []  # Global list to accumulate log messages for the current run


def print_logs():
    for i in run_logs:
        if i.startswith("--"):
            print(i)
        elif "WARNING" in i:
            log.warning(i.replace("WARNING: ", ""))
        else:
            log.info(i)


def parse_input():

    input_params = {}

    default_config_path = state.configs / "run_config.yaml"

    with open(default_config_path, "r") as f:
        params_keys = yaml.safe_load(f)

    # --- Resolve runtime config path ---
    yml_path = paths["rundir"] / "run_config.yaml"

    if not Path(yml_path).exists():
        raise FileNotFoundError(f"Configuration file not found at: {yml_path}")

    # --- Parse runtime YAML ---
    _nml_match = re.compile(r"^tile(7|[8-9]\d*)_nml$")

    with open(yml_path, "r") as file:
        config = yaml.safe_load(file)

    for k, v in config.items():
        if k not in params_keys and not _nml_match.match(k):
            msg = f"Unknown configuration key in run_config.yaml: `{k}`"
            msg += f"\nSee {default_config_path} for valid keys."
            raise KeyError(msg)
        input_params[k] = v

    input_params["run_config"] = Path(yml_path)
    if "res" in input_params and input_params["res"] is not None:
        input_params["res"] = parse_resolution(input_params["res"])

    if "sbatch" in input_params:
        del input_params["sbatch"]  # Remove sbatch config from model params

    for k, v in params_keys.items():
        if v is None:
            continue  # skip undefined defaults

        if input_params.get(k) is None and v is not None:
            input_params[k] = v

    input_params["case_description"] = input_params.get("description", "")

    input_params["warm_start"] = input_params.get("continue_run", False)
    logger(input_params.get("debug"))
    check_prev_state(input_params)
    input_params = parse_datetime(input_params)

    return input_params


def _append_init_logs(params: FV3State) -> None:
    run_logs.append(f"Current directory: {params['rundir']}")
    run_logs.append(f"Working directory: {params['home']}")
    run_logs.append(f"Case directory: {params['case_dir']}")
    run_logs.append(f"Archive directory: {params['archive_dir']}")
    run_logs.append(f"Fixed/static directory: {params['fix']}")
    run_logs.append(f"Configuration file: {params['run_config']}")

    if "shield_exe" in params:
        run_logs.append(f"Model executable:         {params['shield_exe']}")
    else:
        run_logs.append("Model executable:         container image (SHiELD)")

    run_logs.append(f"Description: {params.description}")
    run_logs.append("Initial run mode selected")

    if params["update_nml_only"]:
        run_logs.append("`update_nml_only` flag is set to true.")
    else:
        run_logs.append("Full Grid/IC regeneration will be performed.")

    if state.ensemble_run:
        run_logs.append(f"Ensemble run [{state.ensemble_id}/{state.n_ensembles}]")

    run_logs.append(f"Model initialization time: {params['init_datetime']} UTC")
    run_logs.append(f"Forecast length: {params.run_nhours} hours")
    run_logs.append(f"Vertical levels: {params.levels}")

    run_logs.append(f"Grid type: {params.gtype}")
    run_logs.append(f"Global cubed-sphere resolution: C{params.res}")

    for i in range(1, 7):
        run_logs.append(f"Global tile {i} resolution: {params.global_res_km:.2f} km")

    if params.gtype == "nest":
        run_logs.extend(nest_info)
        run_logs.append(f"Number of nests: {params.n_nests}")
        run_logs.append(f"Refinement ratio: {params.refine_ratio}")

    run_logs.append(f"Target longitude: {params.target_lon}")
    run_logs.append(f"Target latitude: {params.target_lat}")


def _append_restart_logs(params: FV3State) -> None:
    run_logs.append(f"Restart number: {params.restart_no}")


def preprocess_input():

    load_state()
    params = FV3State(parse_input())  # Get parsed arguments

    params.n_cpus = state.n_cpus  # Update n_cpus based on available CPUs
    params.global_res_km = cres_to_deg(params.res).km

    paths = configure_directories(params)

    params.update(paths)

    if params.gtype == "nest":
        params.n_nests = len(params.refine_ratio)
        validate_nests(params)
    else:
        params.n_nests = 0
        params.refine_ratio = 1

    description = [params.datetime, state.case_name]

    params.description = ".".join([str(d).upper() for d in description if d])

    # Derive warm_start from restart_no (single source of truth)
    params.warm_start = (params.restart_no or 0) > 0

    # Enforce invariant: init runs can never be NML-only
    if not params.warm_start:
        params.update_nml_only = False

    # Now decide which block to print
    if not params.warm_start:
        _append_init_logs(params)
    else:
        _append_restart_logs(params)

    state.update(dict(params))
    print_logs()
