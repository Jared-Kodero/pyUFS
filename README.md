# pyUFS HPC Run Guide

`pyUFS` is a Python workflow for configuring, staging, and launching GFDL SHiELD cases on
Oscar and other HPC systems that do not provide the native UFS utilities layout. A thin
shell wrapper hands off to Python, which validates the case configuration, assembles the
launch environment, generates or stages the grid and initial conditions, runs SHiELD,
regrids the output, and synchronizes results back to the case directory.

## Branch and SHiELD release

This branch (`202604`) is configured for `FV3-202604-public` (FMS `2026.01`, FMSCoupler
`full` driver). The SHiELD 2024 release (`FV3-202411-public`) is maintained on the separate
`202411` branch and is not supported here.

```bash
git clone -b 202604 https://github.com/Jared-Kodero/pyUFS.git
```

## Contents

1. Reference documentation
2. Requirements and runtime environment
3. Repository layout
4. Workflow flow
5. Case setup
6. Configuration reference (`run_config.yaml`)
7. Static runtime datasets (`fix/`)
8. Initial conditions and preprocessing
9. Modifying the grid
10. Modifying orography
11. Soil moisture perturbations
12. Time stepping
13. Process decomposition (PEs and layout)
14. Diagnostics and regridded output
15. Restarts and segmented runs
16. Ensembles
17. Archiving
18. Example cases
19. Compiling a custom SHiELD executable
20. Quick start and submission notes
21. SST and sea-ice perturbations (`tgrad_perturbations`)
22. Monthly runs

## 1. Reference documentation

For model background, use the official references below.

- SHiELD model: https://www.gfdl.noaa.gov/shield/
- FV3 dynamical core: https://www.gfdl.noaa.gov/fv3/fv3-documentation-and-references/
- FV3 namelist guide:
  https://www.gfdl.noaa.gov/wp-content/uploads/2017/09/fv3_namelist_Feb2017.pdf
- Noah-MP land model:
  https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs/LAND_SURFACE/noah_mp_tech_note.pdf
- UFS_UTILS: https://noaa-emcufs-utils.readthedocs.io/en/latest/ufs_utils.html
- UFS Weather Model: https://ufs-weather-model.readthedocs.io/en/develop/Introduction.html
- Flexible Modeling System: https://noaa-gfdl.github.io/FMS/md_docs_doxygenGuide.html

## 2. Requirements and runtime environment

The workflow uses three Apptainer images referenced from `run_config.yaml`; SHiELD itself
may instead run from a native executable:

- `preprocess_image` (`docker://gfdlfv3/preprocessing`, with UFS_UTILS at
  `/UFS_UTILS/exec`, plus the Python environment of `configs/env.yaml`) runs the
  preprocessing driver `scripts/driver.py`.
- `shield_image` (`docker://gfdlfv3/shield`) runs `SHiELD_nh.prod.64bit.x` when no
  native `shield_exe` is given.
- `fregrid_image` (`docker://gfdlfv3/fre-nctools` plus the same Python environment) runs
  the regridding stage `scripts/fv3_regrid.py`. Regridding is done in Python (xarray
  and xESMF, `scripts/pyfregrid.py`), not by the FRE-NCtools `fregrid` program.

`configs/install_images.sh` builds all three into `containers_root` as `preprocess.sif`,
`shield.sif` and `fregrid.sif`, so the three image keys must point there. If any image is
missing, `case_run.sh` runs it before the first stage (output in `image_build.log` in the
working directory). To build or rebuild by hand:

```bash
CONTAINERS_DIR=/path/to/containers TMP_DIR=/scratch bash configs/install_images.sh
```

The preprocessing and SHiELD images do not contain the Open MPI help texts
(`/opt/openmpi/share`), without which `mpirun` reports errors as missing help files; the
script copies them from the fre-nctools image, which has the same Open MPI 4.1.0. Images
are pulled from the `latest` tag, which is the newest published release of each
(`gfdlfv3/shield:latest` is the 2024-12-03 push; the older tags are earlier releases).
`SHIELD_TAG`, `PREPROCESS_TAG` and `FREGRID_TAG` select another tag. The sources, SHA-256
sums and conda package lists are written to `$CONTAINERS_DIR/manifest`. A build needs
internet access (Docker Hub, repo.anaconda.com, conda-forge) and several GB in `TMP_DIR`.

The `modules` key lists the host modules loaded before a native `shield_exe` is launched
(default `hpcx-mpi`, `netcdf-mpi`, `libyaml`, `netcdf`); it is not used with the
container image. `case_submit.sh` needs a host `python3` with PyYAML. The Python
dependencies used inside the preprocessing container are listed in `configs/env.yaml`
and include `netcdf4`, `numpy`, `pandas`, `xarray`, `xesmf`, `esmpy`, `f90nml`, `metpy`,
`cartopy`, and `wgrib2`. The scheduler is SLURM. `drivers/sbatch.sh` submits
`drivers/case_run.sh` and forwards the resolved environment. The driver creates the
configured `jobtmp` subdirectory on the allocated node when its parent directory
exists, before selecting the work directory. When `jobtmp` exists, the case runs
there and the results are synchronized back after preprocessing and after a
successful segment; on failure the
logs are copied back. Only an absent `jobtmp` falls back to the case directory; node
count does not change this preference.
The driver logs the selected paths and the reason. When work and case directories
coincide, no copy or deletion is performed for staging. The fallback runs inside the
case tree. The run directory (`case_root/<parent>/<case>`) must not contain the submission
directory, since it is mirrored with deletion and removed after archiving;
`case_submit.sh` refuses such a layout. Every `container_bindpath` entry must exist on the
compute node, or Apptainer does not start.

## 3. Repository layout

```text
pyUFS/
├── case_submit.sh           # Thin wrapper around drivers/case_submit.py
├── bins/                    # Default directory for binaries
├── containers/              # Default directory for containers
├── fix/                     # Default directory for fix files
├── configs/                 # Default configuration and templates
│   ├── run_config.yaml      # Default configuration and inline documentation
│   ├── env.yaml             # Conda environment for the preprocess container
│   ├── input_nml.yaml       # Base FV3 namelist template
│   ├── input_nestXX_nml.yaml# Nest namelist template
│   ├── field_table.yaml     # Tracer table
│   ├── chgres_cube.yaml     # Initial-condition conversion template
│   ├── diag_table           # Default diagnostic output table
│   ├── diag_field.csv       # Diagnostic field reference
│   ├── diag_table.monthly   # Default table for month or year segments
│   ├── install_images.sh    # Container build helper
│   ├── update_fix.py        # Mirror of the NOAA fix tree (Section 7)
│   ├── sync_noaa_fix.sh     # update_fix.py wrapper that loads awscli
│   └── *.vars.csv           # GFS, HRRR, and ERA5 variable maps
├── drivers/                 # Submission and runtime job scripts
│   ├── case_submit.py       # Config validation and job submission
│   ├── sbatch.sh            # sbatch submission template
│   └── case_run.sh          # Runtime driver executed on the compute nodes
├── examples/                # Example run_config.yaml files and overrides (Section 18)
├── preprocess               # Preprocess stage entrypoint
├── fregrid                  # Regridding stage entrypoint (fregrid image)
├── scripts/              # Workflow implementation
├── tests/                   # Test suite without containers or a scheduler (tests/README.md)
└── README.md
```

The default configuration and its inline comments are in
[configs/run_config.yaml](configs/run_config.yaml). The launcher reads the case-local
`run_config.yaml` from the current working directory and fills any unset key from the
default file.

## 4. Workflow flow

1. Create a case directory.
2. Place a `run_config.yaml` in that directory.
3. Run `case_submit.sh` from the case directory, directly or through a local wrapper.
4. `case_submit.sh` deactivates any active conda environment and calls
   `drivers/case_submit.py`.
5. `drivers/case_submit.py` validates the configuration against the default key set, rejects
   unknown keys with a suggested correction, resolves paths and SLURM flags, and submits
   `drivers/case_run.sh` through `drivers/sbatch.sh`. Ensemble members are submitted as
   separate jobs.
6. `case_run.sh` prepares directories, stages the case to the working directory, runs the
   preprocess container, launches SHiELD, runs the regridding stage, synchronizes outputs, and optionally
   resubmits the next segment or archives the case.
7. Inside the preprocess container, `scripts/driver.py` calls the initial driver on the
   first segment and the restart driver on later segments.

The initial driver performs grid generation, orography generation, initial condition
conversion, optional grid plotting, process decomposition, namelist assembly, soil moisture
perturbation, and generation of the model run script.

## 5. Case setup

Create a case directory and place a case-local `run_config.yaml` in it before launching. A
common pattern is:

```bash
INIT="2026031200Z"
CASE_NAME="C96.R4N2.R2N1.CNTRL"
WORK_ROOT="$HOME/scratch/shield_cases/$INIT"
CASE_DIR="$WORK_ROOT/$CASE_NAME"
mkdir -p "$CASE_DIR"
```

Submit from this directory so case-local overrides are read from the same location. After a
successful run the case directory contains staged subdirectories such as `FIXED`, `GRID`,
`IC`, `INPUT`, `LOGS`, `OUTPUT`, `RESTART`, `HIST`, and `TMP`, depending on the case
settings. A `run` symlink points at the active working directory during a single-case job,
and at the archived case after archiving. Ensemble members use `memNN` symlinks instead.

## 6. Configuration reference (`run_config.yaml`)

Each case directory must include a `run_config.yaml`. Start from
[configs/run_config.yaml](configs/run_config.yaml) and override only the keys you need. The
configuration parser rejects unknown keys, so keep the case file aligned with the template.

### 6.1 System paths

| Key | Meaning |
| --- | --- |
| `case_root` | Root of the persistent case tree. |
| `jobtmp` | Preferred work root, created on the allocated node when its parent exists. Absent scratch falls back to `case_root`; node count does not change this preference. |
| `fix_src` | Source tree for static datasets (the `fix` directory). |
| `ufs_utils` | Configured workflow path. The launcher ultimately derives the active repository from `drivers/case_submit.py`. |
| `shield_image`, `fregrid_image`, `preprocess_image` | Apptainer images for the model, regridding and preprocessing (Section 2). |
| `containers_root` | Directory where `configs/install_images.sh` writes the three images; the image keys must point into it. |
| `container_bindpath` | Host paths bound into the containers: a list or a comma-separated string; variables are expanded. Each path must exist on the compute nodes. |
| `shield_root` | Reserved configuration key; it is not consumed by the current launcher. |
| `archive_root` | Root of the archive tree. |
| `shield_exe` | Path to a native SHiELD executable. Required for any multi-node run; an empty value selects the container path for single-node runs. |
| `modules` | Host modules loaded before a native `shield_exe` is launched; unused with the container image. |

Environment variables such as `$USER` and `$HOME` are expanded.

### 6.2 SLURM submission

| Key | Default | Meaning |
| --- | --- | --- |
| `walltime` | 24 | Wall time in hours. |
| `n_nodes` | 4 | Nodes requested. |
| `n_cpus` | 192 | Requested task count. The submitted count is rounded down to a multiple of `n_nodes`. |
| `n_cpus_per_task` | 1 | CPUs per task. |
| `partition` | batch | SLURM partition. |
| `mem` | 0 | Total job memory in GB. 0 uses the scheduler default. |
| `constraint_node` | null | Optional SLURM node constraint passed as `--constraint=<value>`. |
| `exclusive_node` | false | Request exclusive node access. |
| `logfile` | shield_driver | Base name of the driver log written in the case directory. |

Tasks per node are computed as `n_cpus // n_nodes`, and the submitted task count is
`(n_cpus // n_nodes) * n_nodes`. When `mem` exceeds twice the task count, a per-CPU or
per-job memory flag is derived.

### 6.3 Case metadata

| Key | Meaning |
| --- | --- |
| `case_name` | Case identifier. Null falls back to the directory name. |
| `description` | Short experiment label. |
| `fv3_debug` | Verbose model diagnostics. |
| `archive_data` | Archive outputs after the final segment. |
| `merge_freq` | Merge frequency for regridded output across run segments (see Section 14). |

### 6.4 Execution control

| Key | Meaning |
| --- | --- |
| `init_datetime` | Cycle of the initial-condition source data, UTC, in `YYYYMMDDHHZ` form with hour 00, 06, 12 or 18, for example `2026031200Z`. The model starts `forecast_hour` hours later. |
| `run_length` | Length of one segment, in `run_length_units`. |
| `run_length_units` | `hours` (default), `days`, `months` or `years`. Months are calendar months from the segment start (`coupler_nml months`; years are passed as 12 months) and need a start day of 28 or earlier. |
| `forecast_hour` | Lead hour of the `init_datetime` forecast used for initial conditions; 0 selects the analysis. With a non-zero value the model clock (`coupler_nml`, diagnostics, climatologies, the `chgres_cube` date) starts at `init_datetime + forecast_hour`, and regional boundaries are taken from the same cycle at leads `forecast_hour`, `forecast_hour + 3`, ..., written as boundary hours 000, 003, ...; regional runs therefore need a multiple of 3, and the last lead must exist (GFS: hourly to f120, 3-hourly to f384). The state records the source cycle as `ic_cycle`. |
| `resubmit` | Number of sequential resubmissions. The run has `resubmit + 1` segments (see Section 15). |
| `continue_run` | Managed internally by the driver. The initial segment is a cold start and later segments are warm starts. |
| `preprocess_dask_scheduler` | Dask scheduler for the parallel per-tile preprocessing and regridding steps: `processes` (default, spawned worker processes) or `synchronous` (one task at a time in the driver process, for debugging). `threads` is not supported because the tasks change the working directory. |

`c_res` is a string that starts with `C`: `C48`, `C96`, `C192`, `C384`, `C768`, `C1152` or `C3072`. A plain integer such as `96` is still read as the same resolution, and checksums and staged-grid records treat the two spellings alike.

### 6.5 Initial conditions and preprocessing

| Key | Default | Meaning |
| --- | --- | --- |
| `generate_ic_data` | true | Generate the grid and initial conditions during preprocess. |
| `external_ic_dir` | null | Path to a pre-staged case bundle used when `generate_ic_data` is false. |
| `preprocess_only` | false | Stage the complete grid and initial conditions, then exit. |
| `preprocess_grid_only` | false | Generate and stage the grid, then exit for `uniform`, `stretch`, and `nest` cases (see Section 9). |
| `preprocess_orog_only` | false | Generate and stage orography, then exit for `uniform`, `stretch`, and `nest` cases (see Section 10). |

Setting `preprocess_grid_only` or `preprocess_orog_only` implies `preprocess_only`.

The UFS_UTILS surface-analysis and coupled-grid options (`run_global_cycle`,
`run_emcsfc_snow`, `run_emcsfc_ice_blend`, `run_cpld_gridgen` and their inputs) are
described in Section 8.9.

### 6.6 Horizontal grid

| Key | Meaning |
| --- | --- |
| `c_res` | Cubed-sphere face resolution. Approximate spacing: C96 \~ 100 km, C192 \~ 50 km, C384 \~ 25 km, C768 \~ 13 km, C3072 \~ 3 km. |
| `gtype` | `uniform`, `stretch`, `nest`, `regional_gfdl`, or `regional_esg`. |
| `target_lon`, `target_lat` | Grid centre for `stretch` and `regional_esg`. For `nest` and `regional_gfdl` it is replaced by the centre of the (first) box. |
| `stretch_factor` | Schmidt stretching coefficient. Values greater than 1 refine the target region. |

Nested grids, active when `gtype: nest`:

| Key | Meaning |
| --- | --- |
| `refine_ratio` | Refinement ratio for each nest relative to its parent. A list defines multiple nests. |
| `parent_tile` | Parent cubed-sphere tile, 1 to 6: one value for all nests, or a list with one entry per same-level nest. The grid is rotated so that the centre of the first box lies on tile 6, so 6 is the usual value. A telescoping chain uses the first entry for its outermost nest; each inner nest has its predecessor as parent. |
| `halo` | Halo width for the nest boundary exchange. Regional grids require 3. |
| `lon_min`, `lon_max`, `lat_min`, `lat_max` | Bounding box of each nest: one value per `refine_ratio` entry (scalars are accepted for a single nest). A different number of boxes stops preprocessing. |

When `gtype: nest`, the target longitude and latitude are set to the center of the first
bounding box. Each nest covers its box with the smallest block of whole parent cells and
must stay `halo` parent cells inside its parent tile; all nests are generated by one
`make_hgrid --nest_grids` call, and each telescoping nest is bracketed on the generated
grid of its parent nest. The nest layout is classified automatically from the bounding boxes. If each
box is contained inside its predecessor, the layout is telescoping and refinement ratios
compound. Otherwise the nests are treated as independent nests on the same parent grid. For
telescoping nests the effective refinement of nest `i` is the product of ratios up to and
including `i`. A box may cross 0 or 180 degrees longitude. `regional_gfdl` also requires
`lon_min`, `lon_max`, `lat_min`, and `lat_max`; the cube is centred on that box, as for
nests, and the domain is bracketed on tile 6 and refined by the first `refine_ratio` entry;
`regional_esg` is defined by `target_lon`, `target_lat` and the keys below. For both
regional types `c_res` is replaced by the equivalent global resolution of the generated
grid (`global_equiv_resol`).
Regional ESG grids, active when `gtype: regional_esg`, additionally use:

| Key | Meaning |
| --- | --- |
| `idim`, `jdim` | Zonal and meridional grid points. |
| `delx`, `dely` | Supergrid spacing in degrees. The model grid spacing is twice this value (0.0585 is about 13 km). |

### 6.7 Vertical grid and physics

| Key | Meaning |
| --- | --- |
| `levels` | Number of hybrid sigma-pressure levels. |
| `do_deep` | `false` (default) switches deep convection off on every domain with grid spacing of 4 km or less and keeps it elsewhere; `true` keeps it on all domains. |

### 6.8 Time stepping

| Key | Meaning |
| --- | --- |
| `dt_atmos` | Atmospheric time step in seconds. Null triggers automatic selection. |
| `dt_cpld` | Coupling time step in seconds, written as `coupler_nml dt_cpld`. Null uses `dt_atmos`. |
| `k_split` | Remap split counts per domain. Length must equal `n_nests + 1`. |
| `n_split` | Acoustic substep counts per domain. Length must equal `n_nests + 1`. |

See Section 12 for the automatic values and the relations between these quantities.

### 6.9 Surface and orography

| Key | Meaning |
| --- | --- |
| `lake_cutoff` | Land and water fractional threshold. |
| `add_lake` | Add lake fraction and depth (GLDB v2) to the orography files; `uniform` and `regional_gfdl` only. It does not switch on a lake model. |
| `make_gsl_orog` | Generate the GSL drag-suite orography files (`oro_data_ls`, `oro_data_ss`). SHiELD physics does not read them; they are staged for use outside SHiELD. |

### 6.10 Ensembles

| Key | Meaning |
| --- | --- |
| `ensemble_run` | Enable a multi-member ensemble. |
| `n_ensembles` | Number of members. Must be at least 1 when `ensemble_run` is true. |
| `skip_ensembles` | Member indices to omit, for example `[1, 3, 5]`. |

### 6.11 Land surface perturbations

The `sm_perturbations` block applies controlled perturbations to soil-state variables. The
schema and methods are documented in Section 11.

## 7. Static runtime datasets (`fix/`)

The `fix` tree holds the climatologies, lookup tables, orography inputs, and other static
datasets required by a run, resolved from `fix_src`. On Oscar these are already staged, so
manual download is normally unnecessary. If a dataset is missing, the NOAA fix bundle is the
reference source:

https://noaa-nws-global-pds.s3.amazonaws.com/index.html#fix/

Missing files are fetched at run time. `scripts/fv3_update_fix.py` looks for each file
the workflow reads in `fix_src`, then under the name NOAA uses for it in `fix_src` (for
example `am/fix_co2_update/global_co2historicaldata_2020.txt` for
`am/co2historicaldata_2020.txt`, which it links), and then in the NOAA bucket, newest
version directory first, so files dropped from the latest release (the binary orography
inputs of the preprocessing image, present only in `orog/20231027`) are still found.
Downloads go to `fix_src` under the NOAA name, with a link under the name the workflow
reads; `fix_src` must therefore be writable. The chgres_cube variable maps come from the
UFS_UTILS fork of the preprocessing image. A run stops only for a required file found
neither locally nor remotely; `mld_DR003_c1m_reg2.0.grb` and `fix/era5` are not in the
bucket and must be staged by hand. Set `UFS_PY_FIX_REMOTE=0` to disable remote lookups.

For CO2 the observed record (`fix_co2_update`) is preferred over `co2dat_4a`, and both over
the projection (`fix_co2_proj`). When no table exists for a model year, the latest earlier
year is staged with a warning; the radiation code extrapolates from it
(`radiation_gases.f`). The decadal volcanic aerosol tables (1850-1999) are linked into the
working directory, where `radiation_aerosols.f` opens them.

`configs/update_fix.py` (or `configs/sync_noaa_fix.sh`, which loads `awscli`) updates the
whole tree in advance and is independent of the run-time lookup above. It mirrors the
`am`, `orog` and `sfc_climo` directories of the NOAA bucket with `aws s3 sync`, keeps the
newest version of each in `fix/`, removes the pre-generated `C<res>` directories, recreates
the CO2 and volcanic-aerosol links without the `global_` prefix, and downloads the Cartopy
Natural Earth data into `fix/carto`. It writes to the `fix` directory two levels above the
directory it is run from (`../../fix`) and replaces the three directories there.

The soil moisture climatology used by the perturbation module is
`fix/era5/sm_monthly_1950_2025.nc`.

## 8. Initial conditions and preprocessing

When `generate_ic_data: true`, preprocessing generates the grid and orography and then
uses `chgres_cube` to convert external GRIB2 data into FV3 initial conditions. The
workflow currently supports GFS and HRRR as external source models. Source data are
retrieved with retry and multi-source fallback:

- GFS: NOAA AWS S3 and NCAR GDEX.
- HRRR: NOAA AWS S3 and Google Cloud Storage.

The source model is selected per model domain. It is **not** selected with a key in
`run_config.yaml`. To override the built-in source assignment, place a domain-specific
`chgres_cube` YAML file in the case directory from which `case_submit.sh` is launched.

### 8.1 Default GFS/HRRR source assignment

With no case-local `chgres_cube` override files, the workflow uses the following
assignments:

| Domain | Atmospheric source | Surface source | Behavior |
| --- | --- | --- | --- |
| Global | GFS | GFS | One combined GFS conversion. |
| Regional | GFS | GFS | One combined GFS conversion. |
| Nest inside HRRR coverage | HRRR | GFS | Two conversions: HRRR atmosphere, then GFS surface. |
| Nest outside HRRR coverage | GFS | GFS | One combined GFS conversion after the implicit HRRR default falls back to GFS. |

For nested domains, HRRR is therefore the default **atmospheric** source, not the
surface source. The workflow checks each nest against the supported HRRR domain before
using HRRR. If a nest has no explicit `external_model` setting and is outside HRRR
coverage, the automatic HRRR choice falls back silently to GFS.

The surface source for limited-area HRRR initialization remains GFS because the HRRR
input used by this workflow does not provide the required soil levels. Consequently,
an HRRR limited-area initialization is split into two `chgres_cube` calls:

```text
HRRR -> atmospheric fields
GFS  -> surface fields
```

A GFS limited-area initialization uses one call for both atmospheric and surface fields.

### 8.2 Override files and precedence

Overrides are supplied per domain as flat `chgres_cube` configuration mappings in the
case directory. Every override file is optional. Domains without an override retain the
built-in behavior described above.

| Domain | YAML override file |
| --- | --- |
| Global | `chgres_cube.yaml` |
| Regional | `chgres_cube.yaml` |
| Nest `NN` | `chgres_cube_nest{NN}.yaml` |

`chgres_cube.py` also lists `fort.41` and `fort_nest{NN}.41` as candidates. Keys may be
flat or nested in one `config` group, so a conventional namelist with a `&config` section
and a YAML file copied from `configs/chgres_cube.yaml` are both accepted. Every key of the
template is null, meaning "keep the workflow value" (given in its comment), so a verbatim
copy changes nothing; set only the keys to change. The land-surface defaults follow
UFS_UTILS, except `tg3_from_soil: true`: the deep-soil temperature comes from the lowest
soil layer of the source data rather than the static substrate-temperature climatology. `NN` is the
zero-padded nest index
used throughout the workflow: the first nest
is `nest02`, the second is `nest03`, and so on. For example:

```text
CASE_DIR/
├── run_config.yaml
├── chgres_cube_nest02.yaml   # overrides nest02 only
├── chgres_cube_nest03.yaml   # overrides nest03 only
└── ...
```

Override keys use the `chgres_cube` variable names represented by
`configs/chgres_cube.yaml`. Unknown keys stop preprocessing with an error that identifies
the file and offending keys.

### 8.3 Selecting GFS or HRRR

Set `external_model` in the override file for the domain you want to change. The
supported source selections are `GFS` and `HRRR`, subject to the domain restrictions
below.

#### Global domain

The global domain supports GFS initialization. The default requires no override file:

```yaml
external_model: GFS
```

Do not request `external_model: HRRR` for the global domain. HRRR initialization is
implemented only for limited-area regional and nested domains, so an explicit global
HRRR request raises an error.

For the global domain, the conversion switches may also be overridden explicitly:

```yaml
external_model: GFS
convert_atm: true
convert_sfc: true
convert_nst: false
```

#### Regional domain

Regional domains default to GFS. To use HRRR for the regional atmosphere, create
`chgres_cube.yaml` in the case directory:

```yaml
external_model: HRRR
```

The regional domain must lie within HRRR coverage. An **explicit** HRRR request outside
HRRR coverage raises `ValueError`; it is not silently downgraded to GFS. To force GFS,
use:

```yaml
external_model: GFS
```

When HRRR is selected successfully, the workflow performs an HRRR atmospheric
conversion and a separate GFS surface conversion.

#### Nested domains

Each nest can be controlled independently. For the first nest, `nest02`, create
`chgres_cube_nest02.yaml`.

To explicitly request HRRR atmospheric initialization:

```yaml
external_model: HRRR
```

To force GFS for both atmosphere and surface:

```yaml
external_model: GFS
```

To retain automatic source selection, either omit `chgres_cube_nest02.yaml` entirely or
omit `external_model` from that file. The automatic nested behavior is:

```text
nest inside HRRR coverage  -> HRRR atmosphere + GFS surface
nest outside HRRR coverage -> GFS atmosphere + GFS surface
```

The distinction between an implicit and explicit HRRR request is important:

- No `external_model` in a nest override: try HRRR; fall back to GFS if the nest is
  outside HRRR coverage.
- `external_model: HRRR`: require HRRR; raise an error if the nest is outside HRRR
  coverage.
- `external_model: GFS`: force GFS; do not attempt HRRR.

For a two-nest experiment in which `nest02` should use HRRR and `nest03` should use GFS:

`chgres_cube_nest02.yaml`:

```yaml
external_model: HRRR
```

`chgres_cube_nest03.yaml`:

```yaml
external_model: GFS
```

These choices affect only the specified nests; domains without an override continue to
use their defaults.

### 8.4 Conversion switches on limited-area domains

For regional and nested domains, `convert_atm`, `convert_sfc`, and `convert_nst` are
controlled by the source-selection planner. Users should not use these switches to
change the HRRR/GFS split for limited-area initialization.

The planner applies:

| Selected atmospheric source | `convert_atm` | `convert_sfc` | `convert_nst` | Calls |
| --- | ---: | ---: | ---: | --- |
| GFS | `true` | `true` | `false` | One GFS conversion |
| HRRR | `true` for HRRR pass | `false` for HRRR pass | `false` | HRRR atmosphere pass |
| HRRR | `false` for GFS pass | `true` for GFS pass | `false` | GFS surface pass |

If these conversion keys are present in a regional or nest override file, the planner
removes them from the user override and supplies the values required by the selected
source plan. Other recognized settings remain user-overridable.

### 8.5 Overriding other `chgres_cube` settings

A source override can be combined with other recognized `chgres_cube` settings. For
example, the first nest can explicitly require HRRR while changing the output soil-layer
count and vegetation-fraction climatology behavior:

```yaml
external_model: HRRR
nsoill_out: 9
vgfrc_from_climo: false
```

Other recognized settings, including tracer configuration, climatology switches, and
halo-related options, are applied to the corresponding domain unless they are among the
limited-area conversion switches controlled by the planner.

### 8.6 Verifying which source was used

The workflow records source provenance in `state.yaml` under a per-domain
`ic_source` mapping, keyed by domain (`global`, `regional`, `nest02`, ...), with separate
`atm`, `sfc`, and `nst` entries. State files written by earlier versions, with one
`{domain}_ic_source` key per domain, are read as before. This is the
recommended way to verify the resolved source after preprocessing.

A domain converted from multiple sources also receives separate logs for each converted
field group. For example, an HRRR-initialized first nest produces logs such as:

```text
chgres_cube_nest02_atm.log   # HRRR atmospheric conversion
chgres_cube_nest02_sfc.log   # GFS surface conversion
```

This makes the final source assignment traceable even when automatic HRRR-to-GFS
fallback is active for nests.

### 8.7 Preprocessing without launching SHiELD

To generate and stage the grid and initial conditions without running the model, set:

```yaml
generate_ic_data: true
preprocess_only: true
```

The job exits after preprocessing.

### 8.8 External initial-condition bundles

If a pre-generated case bundle already exists, bypass initial-condition generation by
setting `generate_ic_data: false` and pointing `external_ic_dir` at the staged case
bundle:

```yaml
generate_ic_data: false
external_ic_dir: /path/to/prestaged_case
```

The workflow expects a staged case directory containing non-empty `FIXED`, `GRID`, and
`INPUT` directories plus `state.yaml`, not a single NetCDF file that it modifies in place.
Use the repository code as the handoff point when building a custom conversion pipeline
rather than mutating the default files directly.

The bundle's `state.yaml` supplies only the grid and IC keys (`BUNDLE_KEYS` in
`scripts/fv3_external_ic.py`); all other settings, including segment settings and
`tgrad_perturbations`, come from the case `run_config.yaml`. The case model start
(`init_datetime` plus `forecast_hour`), `gtype` and `levels` must match the bundle.

### 8.9 Surface analysis update and coupled grids (UFS_UTILS)

Four UFS_UTILS programs of the preprocessing image can run after `chgres_cube` on a cold
start with `generate_ic_data: true`, each when its `run_*` key is true (all false by
default; `configs/run_config.yaml` section 5a lists the keys). They follow the
corresponding UFS_UTILS `ush` scripts.

- `run_global_cycle` runs `global_cycle` on `INPUT/sfc_data*` (`global_cycle_driver.sh`,
  `global_cycle.sh`): the surface fields are updated at the model start from the
  climatologies of the model `namsfc` (Section 7) and the optional analyses
  `global_cycle_sst_file` (FNTSFA), `global_cycle_ice_file` (FNACNA) and
  `global_cycle_snow_file` (FNSNOA), all GRIB1; a missing analysis keeps the first guess
  and climatology for that field. The relaxation settings are those of the scripts
  (`DELTSFC=6`, `FSMCL(2:4)=60`, `global_cycle_vars` for `CYCLVARS`, default
  `FSNOL=-2., FSNOS=99999.`). The six global tiles run as one 6-rank job and each nest as
  a 1-rank job, since the program takes one tile size per call. Global, stretched and
  nested grids only. Logs: `LOGS/preprocess/global_cycle_<domain>.log`.
- `run_emcsfc_snow` builds the snow analysis from `ims_snow_file` (and the optional AFWA
  files) with `emcsfc_snow2mdl` on the T1534 Gaussian grid (`emcsfc_snow.sh`) and passes
  it to `global_cycle`.
- `run_emcsfc_ice_blend` blends `ims_ice_file` into the MMAB 5-minute concentration
  `five_min_ice_file` with `emcsfc_ice_blend` (`emcsfc_ice_blend.sh`) and passes the GRIB1
  result to `global_cycle`. The script's `cnvgrib`, `copygb2` and `copygb` (NCEP
  grib_util) must be on `PATH`; they are not part of `configs/env.yaml`.
- `run_cpld_gridgen` writes the MOM6/CICE6 grids, SCRIP files and the ocean mask mapped to
  this cubed sphere with `cpld_gridgen` (`cpld_gridgen.sh`) into `IC/cpld_gridgen`, for
  MOM6 resolution `cpld_gridgen_res` (`500`, `100`, `050`, `025`). The MOM6 inputs are
  taken from `fix_src/mom6/<res>` (fetched from the NOAA fix bucket when missing). Other
  resolutions need `Ct.mx025_SCRIP.nc`, so the 025 grid is generated first.
  `ESMF_Scrip2Unstruct` must be available; `cpld_gridgen_postwgts: true` also needs
  `ncremap` (NCO). SHiELD does not read these files; they serve coupled UFS
  configurations.

The analyses are not in the NOAA GFS buckets used for the initial conditions and must be
staged by the user; the emcsfc outputs are kept in `IC/surface_analysis`. The settings
are checked before the grid is generated.

## 9. Modifying the grid

For `uniform`, `stretch`, and `nest` cases, grid generation is driven from
`scripts/fv3_make_grid.py` and staged through a modification directory. The generator
copies user-supplied files verbatim when a non-empty modification directory is present, so
the procedure is stage, edit, and re-inject. The regional branches do not currently stage
and terminate through this same grid-only path.

1. Set `preprocess_grid_only: true` and submit. The workflow generates the grid and mosaic,
   stages them into the case-local `IC/grid` directory, and exits. The driver log reports
   the staging path.
2. Copy the staged files to a backup directory. The staged content includes
   `C{c_res}_grid.tile*.nc` and the corresponding `C{c_res}_*mosaic*.nc` files.
3. Edit the tile files. If you change tile geometry, keep the mosaic consistent, because the
   mosaic is staged and re-injected from the same directory. Preserve filenames exactly.
4. Set `preprocess_grid_only: false`, keep `generate_ic_data: true`, and resubmit. The
   edited grid is copied through without regeneration.

## 10. Modifying orography

For `uniform`, `stretch`, and `nest` cases, orography generation is driven from
`scripts/fv3_make_orog.py` and follows the same stage, edit, and re-inject pattern as the
grid. Orography is generated after the grid, so a grid must exist first. The regional
branches do not currently stage and terminate through this same orography-only path.

1. Set `preprocess_orog_only: true` and submit. The workflow stages the orography into the
   case-local `IC/orography` directory and exits. The `shield_driver*.log` file reports the
   staging path.
2. Copy the staged files to a backup directory. The staged content is
   `oro.C{c_res}.tile*.nc`, and the GSL variants when `make_gsl_orog: true`.
3. Edit the orography. Modify both the `orog_raw` and `orog_filt` variables inside each tile
   file, and update them together to preserve consistency. `orog_raw` is the unfiltered
   surface height and `orog_filt` is the filtered height used by the dynamical core.
   Preserve filenames exactly.
4. Set `preprocess_orog_only: false`, keep `generate_ic_data: true`, and resubmit.

The topography filter runs after orography generation on the global tiles of uniform,
stretched, and nested grids. Nest tiles keep the unfiltered orography.
Inject edited orography as the staged tile files rather than relying on the filter to
preserve raw edits.

## 11. Soil moisture perturbations

Soil moisture perturbations are applied at model initialization and at the start of each
restart segment by `scripts/sm_perturbations.py`. They act on the surface restart files
`sfc_data.tile*.nc`, `sfc_data.nest{NN}.tile*.nc` for nested tiles, and `sfc_data.nc` for a
regional domain. Target variables are `smc` (total volumetric soil moisture), `slc`
(liquid volumetric soil moisture), and `stc` (soil temperature). All three are perturbed on
soil points, identified by valid `smc`. Volumetric soil moisture is clipped to the interval $[0.01, 0.99]\,\mathrm{m^3\,m^{-3}}$. The frozen fraction is held fixed by keeping the ice content
$\mathrm{ice} = \mathrm{smc} - \mathrm{slc}$ constant and reconstructing `slc` after any
`smc` edit. The workflow writes both an original and a perturbed copy of each file into
`IC/perts`, so the unperturbed state is recoverable.

### Schema

```yaml
sm_perturbations:
  target_var: smc          # smc, slc, or stc
  soil_layers: [0, 1, 2]   # integer or list of integers
  tiles: [1, 2, 3, 4, 5, 6]
  method: mean_shift       # string or list of strings
  apply_on_restarts: 0     # None, "all", int, or list of ints
```

Required keys are `target_var`, `soil_layers`, `tiles`, and `method`. If `apply_on_restarts`
is absent, no perturbation is applied. Multiple methods in a list are applied in order.

### Methods

Let $X$ be the soil field in a layer, $\mu$ the mean over valid points, and $\sigma$ the
standard deviation.

Standard deviation shift, `std_shift`, with $k$ from `n_sigma`:

$$X' = X + k\,\sigma$$

Mean scaling, `mean_shift`, with $s$ from `mean_scale`:

$$X' = X\,(1 + s)$$

Anomaly scaling, `anom_shift`, with $a$ from `anom_scale`:

$$X' = \mu + (1 + a)\,(X - \mu)$$

Constant fill, `constant_fill`, with $c$ from `fill_value`, or the field mean when
`fill_value` is the string `mean`:

$$X' = c$$

Climatological replacement, `climo_mean`, replaces valid points with the monthly
climatological mean regridded to the cubed sphere. The month is that of the start of the
current segment. Climatology points outside the valid soil-moisture range are excluded when
regridding, and model points left without a valid regridded value keep their own value.
The climatology layers are matched to the model soil layers by index, so the climatology
file must already be on the model soil layers (Noah: 0-10, 10-40, 40-100 and 100-200 cm).

### Cross-segment behavior

Two behaviors act across restart segments and are mutually exclusive.

Nudging, `do_nudge: true`, relaxes the field toward a reference with weight $\alpha = \Delta
t / \tau$ clipped to $[0, 1]$:

$$X' = (1 - \alpha)\,X + \alpha\,X_\mathrm{ref}$$

Here $\Delta t$ is the length in hours of the previous segment and $\tau$ is `tau_hours`, default 24 hours. The reference
is the previous perturbed segment, or the climatological mean of `climo_file` (default
`fix/era5/sm_monthly_1950_2025.nc`) when `use_climo: true`. Holding, `do_hold: true`,
carries the perturbed `target_var` layers of the previous segment forward without
recomputing; all other surface fields continue from the model restart.

### Optional keys

`n_sigma`, `mean_scale`, `anom_scale`, `fill_value`, `use_climo`, `do_nudge`, `do_hold`,
`climo_file`, `tau_hours`, `apply_on_restarts`. A method that requires a parameter raises an
error if the parameter is missing.

## 12. Time stepping

When `dt_atmos`, `k_split`, or `n_split` are null, the workflow derives them from a base
table indexed by resolution.

| `c_res` | `dt_atmos` (s) | `k_split` | `n_split` |
| --- | --- | --- | --- |
| 48 | 1200 | 2 | 6 |
| 96 | 720 | 2 | 6 |
| 192 | 450 | 2 | 6 |
| 384 | 360 | 2 | 6 |
| 768 | 180 | 2 | 8 |
| 1152 | 120 | 2 | 8 |
| 3072 | 90 | 2 | 10 |

Resolutions outside the table are estimated by a log-log fit of `dt_atmos` against `c_res`
and snapped to a value that divides 3600 seconds. `dt_cpld` defaults to `dt_atmos`. Before
the model is launched, `dt_cpld` is checked to be a multiple of `dt_atmos` and the segment
length a multiple of `dt_cpld`, the checks the SHiELD coupler applies. For nested runs the
finest domain sets
`dt_atmos`, and each domain receives split counts sized to its resolution. The dynamics and
acoustic time steps follow

$$\Delta t_\mathrm{dyn} = \frac{\Delta t_\mathrm{atmos}}{k_\mathrm{split}}$$

$$\Delta t_\mathrm{acoustic} = \frac{\Delta t_\mathrm{atmos}}{k_\mathrm{split}\,n_\mathrm{split}}$$

where $\Delta t_\mathrm{atmos}$ is the atmospheric time step in seconds, $k_\mathrm{split}$
is the remap split count, and $n_\mathrm{split}$ is the acoustic substep count per remap
split. Supplied `k_split` and `n_split` must be lists of length `n_nests + 1`, with the
first entry for the global grid and the remaining entries for the nests in order. The
atmospheric time step is CFL constrained and should decrease as horizontal resolution or
refinement increases.

## 13. Process decomposition (PEs and layout)

Process counts and domain layouts are computed automatically from the grid. For a uniform
grid the total process count is the largest multiple of 6 not exceeding `n_cpus` whose
per-tile count has a valid layout, distributed equally across the six tiles. A regional
grid uses the largest valid count not exceeding `n_cpus`. For nested runs the workflow distributes processes
across the global grid and each nest by minimizing the largest estimated per-domain time,

$$T_g \sim \frac{w_g}{P_g}$$

$$w_g = N_g\,k_{\mathrm{split},g}\,n_{\mathrm{split},g}$$

where $T_g$ is the estimated time for domain $g$, $P_g$ is its process count, $w_g$ is a
work weight, $N_g$ is the number of horizontal cells, and $k_{\mathrm{split},g}$ and
$n_{\mathrm{split},g}$ are its split counts. Global process counts are multiples of 6. Nest
process counts are drawn from a set that keeps subdomain aspect ratios no more elongated
than 2 to 1. An allocation that uses all `n_cpus` is preferred; among those, the smallest
bottleneck time, then the smallest spread. The per-domain layout requires at least 4 cells
per subdomain edge (FV3 exchanges 3-point halos), then prefers a layout that divides the
cells of the domain evenly in both directions, so every PE holds the same subdomain, then
the most nearly square subdomains. The I/O layout is set to one by one and the physics
block size to 32.

To override the automatic allocation, place a case-local `input.nml` or `input.yaml` in the
case directory that sets `grid_pes` under `fv_nest_nml` (the first of `input.nml`,
`input.yaml` and `input.yml` found is used, as for namelist overrides). The listed values
become the per-domain process counts; there must be one per domain, the global count must
be a multiple of 6, and their sum must not exceed `n_cpus`.

## 14. Diagnostics and regridded output

Output frequency and the reported variable set are controlled by a `diag_table`. Place a
case-local `diag_table` in the case directory to override the default in
`configs/diag_table` (`configs/diag_table.monthly` for month and year segments). Its first
two entries, the title and the base date (literal or the `DESCRIPTION` and `DATETIME`
placeholders), are replaced by the case description and the initialization time. A
case-local `field_table.yaml`, or a legacy ASCII `field_table`, replaces
`configs/field_table.yaml`; `field_manager_nml use_field_table_yaml` follows the format. A
case-local `data_table` (`data_table.yaml` with `use_modern_diag: true`) is staged when
present. The default defines three streams: `grid_spec` and `atmos_static` written once, and `fv3_hist`
written hourly (monthly means in the monthly table). The variable reference list is in
[configs/diag_field.csv](configs/diag_field.csv). After the model runs, the regridding stage
remaps native history to a latitude-longitude grid: the global grid at the resolution of
`c_res`, each nest over its box (a box across 180 degrees gives longitudes above 180), and a
regional domain over the bounding box of its compute domain, with missing values outside
the domain. Regridded files are named by domain: `global`, `regional`, and `nest02`,
`nest03`, and so on, where nest tile 7 maps to `nest02`. Conservative remapping averages
only valid source values, so missing values (for example pressure levels below the
surface) are neither filled nor counted as zero, and fields without horizontal dimensions
(`average_T1`, `average_T2`, `average_DT`, `time_bnds`) are copied unchanged. Weights are
computed once per grid and stream set.

The `merge_freq` key controls how per-segment regridded files are combined.

- `-1` merges the whole run into one file per stream and grid on the final segment.
- `0` disables merging and retains one file per segment.
- `n` merges every `n` segments and flushes any remainder on the final segment.

Merging concatenates records along time; it does not average (Section 22).

### 14.1 Grid visualization

The initial driver optionally renders the generated grid through
`scripts/fv3_plot_grid.py`. Plotting is a diagnostic step: it reads the
`C*_grid.tile*.nc` supergrid files from `state.grid`, requires `cartopy` and its Natural
Earth cache under `fix_src/carto`, and writes to `state.run_dir`. Two figures are produced.

- `grid_faces.png` places one panel per global cubed-sphere face. Each panel is an
  orthographic view centred on the centroid of its own face, with every nest hosted by that
  face drawn on it and the host mesh masked out beneath each nest, so a mesh is shown only
  where it is the finest grid present. Nest hosting follows `state.parent_tile` up the
  parent chain, so a telescoping chain resolves to the single global face at its root. Line
  density is set by physical grid spacing rather than array size, so a fine nest and its
  coarse host read at a comparable spacing on the page.
- `nest_grids.png` draws the nest bounding boxes from `state.lon_min`, `state.lon_max`,
  `state.lat_min`, and `state.lat_max` on a map whose projection is selected from the union
  of those bounds: Robinson for whole-world spans, cylindrical equidistant (PlateCarree) for
  low-latitude or equator-straddling domains, polar stereographic for high-latitude domains,
  and Lambert conformal conic for mid-latitudes. Bounds are read in degrees east in the
  range [-180, 180] and each nest is assumed not to cross the antimeridian.

Coordinate units are taken from the `units` attribute of the supergrid `x` and `y`
variables; absent that attribute, degrees are assumed, in line with the FV3 supergrid
convention. Both figures are written to disk and no interactive window is opened, so
plotting is safe on a headless compute node. Failures in the plotting stage are logged and
do not interrupt the run.

## 15. Restarts and segmented runs

A run is divided into `resubmit + 1` segments, each `run_length` `run_length_units` long. The first
segment is a cold start produced by the initial driver. Each later segment is a warm start
produced by the restart driver, which resumes from the previous segment restart files.
Restart segments load the persisted `state.yaml`; their settings come from it, not from
`run_config.yaml`. The cold start records a checksum of the grid geometry, vertical
levels, initialization time and, when set, `tgrad_perturbations` of `run_config.yaml`;
each restart compares it with the current `run_config.yaml` before moving any directory
and stops if these settings were edited. Other keys (for example `run_length` or
`resubmit`) are not checked. Regional restart segments must start on multiples of 3 h, the
interval of the boundary files, because the model reads the boundary file of its restart
hour. Forcing files in `FIXED/MODS`
and the cold-start namelists are reused by later segments; the case `diag_table` is
restaged every segment. When `RESTART` is promoted to `INPUT`, the static inputs of the
cold-start `INPUT` (archived as `IC/INPUT`) that the model also reads at a warm start are
linked back: `oro_data*`, grid and mosaic links, `gfs_ctrl.nc` and fix files. The initial
conditions and boundary files are not. Segments are resubmitted automatically until the maximum index
is reached.

## 16. Ensembles

Set `ensemble_run: true` and `n_ensembles` to submit an ensemble. Each member is submitted
as an independent job with its own working directory `memNN` and its own log. Use
`skip_ensembles` to omit specific members. Ensemble members can be combined with soil
moisture perturbations to build spread.

Member 1 is unperturbed; other members get temperature noise seeded from the case
checksum (`scripts/fv3_ensemble_driver.py`), which includes `tgrad_perturbations`
when set. Perturbations are generated only when ICs are generated; to pair members
across experiments, stage the same member bundle through `external_ic_dir`.

## 17. Archiving

When `archive_data: true`, the final segment copies the regridded output to the archive tree
under `archive_root`, writes a copy of `state.yaml` and the model log, and compresses the
persistent working case tree into `case.tar.gz`. After archiving, that working tree is
removed and the case-local `run` (or ensemble `memNN`) symlink points to the archived
location.

## 18. Example cases

Example case files are in [examples/](examples). Copy a `run_config.<case>.yaml` into the
case directory as `run_config.yaml`; unset keys come from the defaults.

| File | Case |
| --- | --- |
| `run_config.uniform.yaml` | Global uniform C96, hourly output |
| `run_config.stretch.yaml` | Stretched global grid |
| `run_config.nest.yaml` | Two telescoping nests; `chgres_cube_nest02.yaml` forces GFS ICs |
| `run_config.regional.yaml` | Regional ESG domain |
| `run_config.external_ic.yaml` | Start from a staged grid and IC bundle |
| `run_config.sm_perturbations.yaml` | Soil moisture perturbation (Section 11) |
| `run_config.tgrad.yaml` | SST and sea-ice perturbation (Section 21) |
| `run_config.monthly.yaml` | Ten years of monthly segments, yearly files (Section 22) |

## 19. Compiling a custom SHiELD executable

If you need a custom binary, build it from the SHiELD source tree and point `shield_exe` at
the result. The workflow uses the native executable when the resolved `shield_exe` value is
non-empty.
An empty value selects the container image for a single-node run; all multi-node runs require
a native `shield_exe`. Because the non-null repository default is restored when a case sets
`shield_exe: null`, use `shield_exe: ""` to force the single-node container path.

Use `scripts/build_shield.py` rather than running `CHECKOUT_code` and `COMPILE` by hand:

```bash
git checkout 202604
python scripts/build_shield.py --root /path/to/build_root
```

The script builds `FV3-202604-public`. It reads
`shield_exe` and `modules` from `configs/run_config.yaml` (the executable is built with the
modules it is launched with), checks that every module exists on the host, clones
`SHiELD_build` into `build_root/FV3-<profile>-public/`, runs `CHECKOUT_code`, verifies the
tagged sources, writes `site/environment.gnu.sh` for the machine (module loads, `mpif90`,
`mpicc`, `mpicxx`, netCDF and HDF5 locations from `nc-config`, `FMS_CPPDEFS=-DHAVE_GETTID`
when glibc is 2.30 or later), tests the toolchain (gfortran 10 or later, MPI wrappers,
`nc-config`, `nf-config`, `cmake`, libyaml), runs
`COMPILE shield nh prod 64bit gnu pic cleanall`, and copies the executable to `shield_exe`
with `shield_exe.manifest`, which lists the commit of every component, the SHA-256, the
toolchain and the build tree. `--no-compile` stops after the environment file, `--modules a,b`
and `--march` override the module list and `AVX_LEVEL` (empty by default, so the code runs on
every node), `--exe` overrides the install path and `--force` replaces an existing tree. The
script does not change `run_config.yaml`. The upstream steps are not safe to repeat for a
second release, which is why each release gets its own tree and a clean build:

- `SHiELD_build/CHECKOUT_code` clones into `../SHiELD_SRC` with `git clone`. When that
  directory already holds another release the clones fail, the sources stay at the old
  release, and the script still writes the new release name to `SHiELD_SRC/release`. Two
  `SHiELD_build` clones in the same parent directory share one `SHiELD_SRC`.
- `COMPILE` reuses `Build/libFMS/gnu` and `Build/nceplibs/gnu` whenever the library files exist,
  whichever FMS version made them, and
  `make` does not rebuild an object when only a library module file changed. Switching
  releases in one tree therefore links new sources against the old FMS modules and objects.
  The `cleanall` option removes the libraries, the NCEP libraries and the objects first.
- `COMPILE` tests the exit status of the `mv` that follows `make`, not of `make`. The script
  checks that a new, non-empty executable exists and otherwise reports the errors in
  `Build/build_shield_nh.prod.64bit.gnu.out`.
- The script compares every tagged repository with the tag that `CHECKOUT_code` requests and
  stops on a difference.

FMS `affinity.c` needs no source patch: `-DHAVE_GETTID` makes it use the `gettid` of glibc, as
in the upstream Gaea environment. The compile options are the upstream ones for the production
GNU build (`shield nh prod 64bit gnu`). The upstream regression tests use Intel with `repro`,
so a GNU `prod` build is a different configuration from the one NOAA verifies. For a first run
of a new profile, also build `repro` and `debug` (edit the `COMPILE` line in a copy of the
script, or run it in `build_root/FV3-<profile>-public/SHiELD_build/Build`) and run the same
case with each.

Set the executable path in `run_config.yaml`:

```yaml
shield_exe: /path/to/FV3-202604-public_SHiELD_nh.prod.64bit.gnu.x
```

`case_submit.sh` stops when the configured `shield_exe` does not exist.

### 19.1 Build profile

| Component | Version |
| --- | --- |
| Core, physics, drivers | `FV3-202604-public` |
| FMS / FMSCoupler | `2026.01` / `2026.01` |
| `ice_param`, `*_null` | as cloned (untagged) |
| Executable (`shield_exe` default) | `FV3-202604-public_SHiELD_nh.prod.64bit.gnu.x` |

The `shield` build uses `FMSCoupler/full`, which reads `coupler_nml dt_cpld` and rejects
`dt_ocean`. The workflow key `dt_cpld` is written directly to that variable. For custom
builds, check `Build/exec/*/pathnames_driver`. Case-local `input.nml` or `input.yaml`
overrides are passed through normally; the model reports incompatible namelist keys.

In the full-driver template, `coupler_nml do_land = false` skips the separate
land component linked from `land_null`. Noah still runs inside SHiELD physics
with `gfs_physics_nml lsm = 1`; the standard `shield` build defines
`_USE_LEGACY_LAND_`, and `atmos_model_nml fullcoupler_fluxes` defaults to 0.
Likewise, `coupler_nml do_ocean = false` skips the external ocean component,
while `gfs_physics_nml do_ocean = true` retains the internal slab ocean.
`do_flux = false` skips the coupler's component-exchange flux calculation;
surface fluxes are still calculated by the atmospheric physics. The `ice_npes`
and `land_npes` settings follow the upstream SHiELD regression cases. These defaults describe the standard
`shield` build, not a coupled `shiemom_lm4` experiment.

The namelist templates use variables declared in `FV3-202604-public`, except
`interpolator_nml interp_method` and the C3072 template's `cloud_diagnosis_nml`, which the
SHiELD_build test cases also set. `gfs_physics_nml sfc_coupled` and `fms_io_nml` are not
namelist variables in this release and are not set. Record the commits in `SHiELD_SRC`, the
executable checksum, the modules (`modules` key) and the container digests, and validate any
newer release separately before use.

### 19.2 UFS_UTILS and containers

`UFS_UTILS` provides the compiled preprocessing utilities (`chgres_cube`, `make_hgrid`,
`orog`, `global_cycle`, ...). `pyUFS` is this workflow; its `ufs_utils` key names the
workflow checkout. Preprocessing runs the utilities from `/UFS_UTILS/exec` inside
`preprocess_image` (built from `docker://gfdlfv3/preprocessing:latest` by
`configs/install_images.sh`). Building `UFS_UTILS` on the host does not change these
binaries; rebuild the image or bind the host `exec` directory to `/UFS_UTILS/exec`.
The image is built from the `gaea` branch of `kaiyuan-cheng/UFS_UTILS` (2024-10-18;
Containerized_SHiELD_Workflow), and the workflow follows its program interfaces: the
`orog` program reads the binary terrain inputs (`gmted2010.30sec.int`,
`landcover30.fixed`, `thirty.second.antarctic.new.bin`) and a nine-number `INPS` line, and
`lakefrac` takes four arguments with GLDB v2 data. Upstream UFS_UTILS releases of 2024 and
later read NetCDF terrain (`topography.gmted2010.30s.nc`, ...) with a three-line `INPS`
and a five- or seven-argument `lakefrac`, so an upstream `exec` directory is not a drop-in
replacement.
`build_all.sh` supports listed machines only; other sites configure dependencies and CMake
themselves.

## 20. Quick start and submission notes

1. Create a case directory.
2. Write `run_config.yaml` into it.
3. Add optional case-local overrides such as `diag_table`, `chgres_cube.yaml`, or
   `input.nml`.
4. Run `case_submit.sh` from the case directory.

Recommended case-local launcher:

```bash
#!/bin/bash -l
"/path/to/pyUFS/case_submit.sh"
```

Deactivate any active conda environment and use a clean shell before submitting, so the
workflow starts from a predictable environment. Place the launcher in the case directory and
run it there so the workflow reads the local `run_config.yaml` and any case-local files.

## 21. SST and sea-ice perturbations (`tgrad_perturbations`)

`scripts/tgrad_perturbations.py` writes perturbed copies of the monthly SST and sea-ice
climatologies to `FIXED/MODS` once, at the cold start, and points `namsfc` at them. Restart
segments keep these files; `preprocess_only` exits before they are written. Any block,
including `method: none`, also prescribes SST and sea ice from the climatologies. The block
is validated when the case starts. See `examples/run_config.tgrad.yaml`, the comments in
`configs/run_config.yaml` (its section 11) and the module for methods and limits.

## 22. Monthly runs

Use one-month segments starting at 00 UTC on day 1 (`examples/run_config.monthly.yaml`):
`resubmit: 119` gives 120 segments (ten years) and `merge_freq: 12` writes one file per
year; merging only concatenates segments. When the case has no `diag_table`, month and
year segments use `configs/diag_table.monthly`, which averages the hourly samples
(`.true.`) over each month. `fhzero` and `fdiag` are both 1 h by default, so each sample
of a bucket field such as `totprcpb_ave` covers one hour; keep them equal in any override.
The diag tables write time in seconds for every output frequency; the regridded files
keep the time handling of `post_process` unchanged. Fields without horizontal axes, such as `average_T1`, `average_T2` and `average_DT`, are copied to the regridded files.
