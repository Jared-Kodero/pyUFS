# ufs_py HPC Run Guide

`ufs_py` is the Python workflow used to configure, stage, and launch GFDL SHiELD cases on Oscar and other HPC systems that do not provide the native UFS utilities layout.

## Overview

This repository contains the scripts that prepare a case directory, build the runtime environment, generate or stage initial conditions, run SHiELD, and synchronize the results back to the case directory. The workflow is intentionally thin: the shell wrapper hands off to Python, and Python assembles the launch environment and submits the runtime job.

## Additional Documentation

For model-specific background, use the official references below:

* SHiELD Model: https://www.gfdl.noaa.gov/shield/
* NOAH MP Land Model: https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs/LAND_SURFACE/noah_mp_tech_note.pdf
* FV3 Dynamical Core: https://www.gfdl.noaa.gov/fv3/fv3-documentation-and-references/
* FV3 Namelist Guide: https://www.gfdl.noaa.gov/wp-content/uploads/2017/09/fv3_namelist_Feb2017.pdf
* UFS Utilities: https://noaa-emcufs-utils.readthedocs.io/en/latest/ufs_utils.html
* UFS Weather Model: https://ufs-weather-model.readthedocs.io/en/develop/Introduction.html
* Flexible Modeling System: https://noaa-gfdl.github.io/FMS/md_docs_doxygenGuide.html

## Workflow Flow

The repo launch path is:

1. Create a case directory.
2. Place a `run_config.yaml` in that case directory.
3. Run `case_submit.sh` from the case directory, either directly or through a small local wrapper.
4. `case_submit.sh` calls `drivers/case_submit.py`.
5. `drivers/case_submit.py` validates the config, resolves the environment, and submits `drivers/case_run.sh`.
6. `case_run.sh` stages files, runs preprocess, launches SHiELD, runs fregrid, and syncs outputs.

## Repository Layout

The repository is organized as follows:

```text
ufs_py/
├── case_submit.sh           # Thin wrapper around drivers/case_submit.py
├── configs/                 # Default runtime configuration and templates
│   ├── run_config.yaml
│   ├── env.yaml
│   ├── input_nml.yaml
│   ├── field_table.yaml
│   ├── data_table.yaml
│   ├── diag_table
│   ├── diag_field.csv
│   └── *.vars.csv
├── drivers/                 # Submission and runtime job scripts
├── fregrid/                 # Regridding stage interface
├── preprocess/              # Preprocess stage entrypoint
├── py_scripts/              # Workflow implementation
├── tests/                   # Example case and validation assets
└── README.md
```

The default configuration template lives in [configs/run_config.yaml](configs/run_config.yaml), and the launcher reads the case-local `run_config.yaml` from the current working directory.


## Static Runtime Datasets (`fix/` directory)

The `fix` tree contains the climatologies, lookup tables, orography inputs, and other static datasets needed by a run. These files are already staged on Oscar under `<path_to_dir>/gfdl_shield/fix`, so you normally do not need to download them manually.

If a dataset is missing on your system, use the NOAA fix bundle as the source of truth:

https://noaa-nws-global-pds.s3.amazonaws.com/index.html#fix/

## Case Setup

Create a case directory, then place a case-local `run_config.yaml` in that directory before launching the workflow. A common pattern is:

```bash
INIT="2026031200Z"
CASE_NAME="C96.R4N2.R2N1.CNTRL"
WORK_ROOT="$HOME/scratch/shield_cases/$INIT"
CASE_DIR="$WORK_ROOT/$CASE_NAME"

mkdir -p "$CASE_DIR"
```

The generated case directory is the working unit for the workflow. Submit the job from this directory so case-local overrides are picked up from the same place. After a successful run you should expect staged directories such as `FIXED`, `GRID`, `IC`, `INPUT`, `LOGS`, `OUTPUT`, `RESTART`, `HIST`, and `TMP` depending on the case settings.

## Initial Conditions and Preprocessing

The workflow can generate initial conditions automatically from operational GFS or HRRR data when `generate_ic_data: true`. The preprocess stage runs inside the container stack and stages the grid, IC files, and supporting metadata for the case.

This is the supported default path for new runs:

1. Set a valid initialization time in `run_config.yaml`.
2. Leave `generate_ic_data: true`.
3. Let the workflow download and convert the upstream data during preprocess.

If you only want the workflow to stage the grid and IC files, set `preprocess_only: true`. The job exits after preprocessing and does not launch the model.

### External IC Bundles

If you already have a pre-generated case bundle, set `generate_ic_data: false` and point `external_ic_dir` at that bundle. The workflow expects a staged case directory containing the files it reads at startup, not an arbitrary NetCDF file that it edits in place.

Example:

```yaml
generate_ic_data: false
external_ic_dir: /path/to/prestaged_case
```

If you are building your own external IC pipeline, use the repository code as the handoff point rather than trying to mutate the default files in place. The `py_scripts/era5_to_fv3.py` and related helpers are good starting references for a custom conversion flow.

## Config Reference

Each case directory must include a `run_config.yaml`. Start from [configs/run_config.yaml](configs/run_config.yaml) and copy only the settings you need to override.

Important fields:

* `init_datetime`: initialization cycle in `YYYYMMDDHHZ` form.
* `run_nhours`: forecast length in hours.
* `c_res`: cubed-sphere face count for the grid. The code and docs often refer to this as `C96`, `C192`, and so on.
* `gtype`: grid type such as `uniform`, `nest`, `stretch`, or a regional mode supported by the local utilities.
* `levels`: number of vertical levels.
* `generate_ic_data`: generate ICs and grid during preprocess.
* `external_ic_dir`: path to a staged bundle when `generate_ic_data` is false.
* `preprocess_only`: stop after staging the case files.
* `continue_run`: restart from existing output when the case supports it.
* `archive_data`: copy final outputs into the archive tree.
* `shield_exe`: optional path to a native SHiELD executable.

The config parser rejects unknown keys, so keep custom case files aligned with the template.

### Example Global Case

```yaml
description: C96 control run
init_datetime: "2026031200Z"
run_nhours: 6
c_res: 96
gtype: uniform
levels: 64
generate_ic_data: true
preprocess_only: false
continue_run: false
archive_data: true
shield_exe: /path/to/SHiELD_nh.prod.64bit.gnu.x
constraint_node: false
exclusive_node: false
walltime: 12
n_nodes: 2
n_cpus: 96
n_cpus_per_task: 1
partition: batch
```

### Example Nested Case

```yaml
description: Nested SHiELD case
init_datetime: "2026031200Z"
run_nhours: 24
c_res: 96
gtype: nest
levels: 64
refine_ratio: [4, 2]
lon_min: [-125, -95]
lon_max: [-47, -57]
lat_min: [25, 32]
lat_max: [60, 55]
generate_ic_data: true
preprocess_only: false
shield_exe: /path/to/SHiELD_nh.prod.64bit.gnu.x
constraint_node: false
exclusive_node: false
walltime: 12
n_nodes: 2
n_cpus: 96
n_cpus_per_task: 1
partition: batch
```

## Compiling a Custom SHiELD Executable

If you need a custom model binary, build it from the SHiELD source tree and point `shield_exe` at the resulting executable. The workflow will use the native executable when `shield_exe` is set; otherwise it falls back to the container image. For multi-node native runs, `shield_exe` is required.

Typical build outline:

```bash
git clone -b oscar https://github.com/biosphereNclimate/SHiELD_build.git
cd SHiELD_build
./CHECKOUT_code
git submodule update --init mkmf
```

On newer glibc systems, patch `SHiELD_SRC/FMS/affinity/affinity.c` so the local `gettid` helper does not conflict with glibc's definition. The usual fix is to remove the duplicate `static` qualifier from the local declaration.

Before compiling, update `SHiELD_build/site/environment.gnu.sh` so the GNU build loads the modules required on Oscar:

```bash
module load hpcx-mpi
module load netcdf-mpi
module load libyaml
module load cmake
```

Then build the executable:

```bash
./Build/COMPILE 64bit gnu pic
```

Set the resulting executable path in `run_config.yaml`:

```yaml
shield_exe: /path/to/SHiELD_nh.prod.64bit.gnu.x
```

## Minimal Quick Start

1. Create a case directory.
2. Copy or write `run_config.yaml` into that case directory.
3. Add any optional case-local overrides such as `diag_table`.
4. Run `case_submit.sh` from the case directory.

Example case-local launcher:

```bash
#!/bin/bash -l
"/path/to/ufs_py/case_submit.sh"
```

## Diagnostics

To change output frequency or the set of reported variables, place a case-local `diag_table` in the case directory where you submit the run. If no local file is present, the workflow uses the default table from `configs/diag_table`.

The variable reference list is in [configs/diag_field.csv](configs/diag_field.csv).

## RUN CONFIG OPTIONS

See [configs/run_config.yaml](configs/run_config.yaml) for the full set of supported options and inline comments.

## Submission Notes

Before submitting, deactivate any active conda environments and use a clean shell so the workflow starts from a predictable environment.

Recommended local launcher:

```bash
#!/bin/bash -l
"/path/to/ufs_py/case_submit.sh"
```

Place that script in the pwd directory and run it from there so the workflow reads the local `run_config.yaml` and any case-local files such as `diag_table`.
