# ufs_py HPC Run Guide

`ufs_py` is a Python workflow for configuring, staging, and launching GFDL SHiELD
cases on Oscar and other HPC systems that do not provide the native UFS utilities
layout. A thin shell wrapper hands off to Python, which validates the case
configuration, assembles the launch environment, generates or stages the grid and
initial conditions, runs SHiELD, regrids the output, and synchronizes results back
to the case directory.

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

## 1. Reference documentation

For model background, use the official references below.

* SHiELD model: https://www.gfdl.noaa.gov/shield/
* FV3 dynamical core: https://www.gfdl.noaa.gov/fv3/fv3-documentation-and-references/
* FV3 namelist guide: https://www.gfdl.noaa.gov/wp-content/uploads/2017/09/fv3_namelist_Feb2017.pdf
* Noah-MP land model: https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs/LAND_SURFACE/noah_mp_tech_note.pdf
* UFS_UTILS: https://noaa-emcufs-utils.readthedocs.io/en/latest/ufs_utils.html
* UFS Weather Model: https://ufs-weather-model.readthedocs.io/en/develop/Introduction.html
* Flexible Modeling System: https://noaa-gfdl.github.io/FMS/md_docs_doxygenGuide.html

## 2. Requirements and runtime environment

The workflow runs inside three Apptainer containers, referenced from
`run_config.yaml`:

* `preprocess_image` runs the preprocessing driver `py_scripts/driver.py`.
* `shield_image` runs the SHiELD executable when a native binary is not supplied.
* `fregrid_image` runs the regridding stage `py_scripts/fv3_regrid.py`.

If the container directory is empty, `case_run.sh` builds the images through
`configs/install_images.sh` before the first stage. Host modules loaded for the
job are set by the `modules` key, with a default of `hpcx-mpi`, `netcdf-mpi`,
`libyaml`, and `netcdf`. The Python dependencies used inside the preprocessing
container are listed in `configs/env.yaml` and include `netcdf4`, `numpy`,
`pandas`, `xarray`, `xesmf`, `esmpy`, `f90nml`, `metpy`, `cartopy`, and `wgrib2`.

The scheduler is SLURM. `drivers/sbatch.sh` submits `drivers/case_run.sh` and
forwards the resolved environment. Node-local scratch is used when `jobtmp`
exists: the case directory is copied to the working directory on the compute
node, the model runs there, and outputs are synchronized back to the case
directory. When `jobtmp` is absent the workflow runs in place inside the case
tree.

## 3. Repository layout

```text
ufs_py/
├── case_submit.sh           # Thin wrapper around drivers/case_submit.py
├── configs/                 # Default configuration and templates
│   ├── run_config.yaml      # Default configuration and inline documentation
│   ├── env.yaml             # Conda environment for the preprocess container
│   ├── input_nml.yaml       # Base FV3 namelist template
│   ├── input_nestXX_nml.yaml# Nest namelist template
│   ├── field_table.yaml     # Tracer table
│   ├── data_table.yaml      # Data override table
│   ├── chgres_cube.yaml     # Initial-condition conversion template
│   ├── diag_table           # Default diagnostic output table
│   ├── diag_field.csv       # Diagnostic field reference
│   ├── install_images.sh    # Container build helper
│   └── *.vars.csv           # GFS, HRRR, and ERA5 variable maps
├── drivers/                 # Submission and runtime job scripts
│   ├── case_submit.py       # Config validation and job submission
│   ├── sbatch.sh            # sbatch submission template
│   └── case_run.sh          # Runtime driver executed on the compute node
├── fregrid                  # Regridding stage entrypoint
├── preprocess               # Preprocess stage entrypoint
├── py_scripts/              # Workflow implementation
└── README.md
```

The default configuration and its inline comments are in
[configs/run_config.yaml](configs/run_config.yaml). The launcher reads the
case-local `run_config.yaml` from the current working directory and fills any
unset key from the default file.

## 4. Workflow flow

1. Create a case directory.
2. Place a `run_config.yaml` in that directory.
3. Run `case_submit.sh` from the case directory, directly or through a local
   wrapper.
4. `case_submit.sh` deactivates any active conda environment and calls
   `drivers/case_submit.py`.
5. `drivers/case_submit.py` validates the configuration against the default key
   set, rejects unknown keys with a suggested correction, resolves paths and
   SLURM flags, and submits `drivers/case_run.sh` through `drivers/sbatch.sh`.
   Ensemble members are submitted as separate jobs.
6. `case_run.sh` prepares directories, stages the case to the working directory,
   runs the preprocess container, launches SHiELD, runs fregrid, synchronizes
   outputs, and optionally resubmits the next segment or archives the case.
7. Inside the preprocess container, `py_scripts/driver.py` calls the initial
   driver on the first segment and the restart driver on later segments.

The initial driver performs grid generation, orography generation, initial
condition conversion, optional grid plotting, process decomposition, namelist
assembly, soil moisture perturbation, and generation of the model run script.

## 5. Case setup

Create a case directory and place a case-local `run_config.yaml` in it before
launching. A common pattern is:

```bash
INIT="2026031200Z"
CASE_NAME="C96.R4N2.R2N1.CNTRL"
WORK_ROOT="$HOME/scratch/shield_cases/$INIT"
CASE_DIR="$WORK_ROOT/$CASE_NAME"

mkdir -p "$CASE_DIR"
```

Submit from this directory so case-local overrides are read from the same
location. After a successful run the case directory contains staged
subdirectories such as `FIXED`, `GRID`, `IC`, `INPUT`, `LOGS`, `OUTPUT`,
`RESTART`, `HIST`, and `TMP`, depending on the case settings. A `run` symlink
points at the active working directory during the job, and at the archived case
after archiving.

## 6. Configuration reference (`run_config.yaml`)

Each case directory must include a `run_config.yaml`. Start from
[configs/run_config.yaml](configs/run_config.yaml) and override only the keys you
need. The configuration parser rejects unknown keys, so keep the case file
aligned with the template.

### 6.1 System paths

| Key | Meaning |
| --- | --- |
| `case_root` | Root of the persistent case tree. |
| `jobtmp` | Node-local scratch root. When present, the model runs here and syncs back to `case_root`. |
| `fix_src` | Source tree for static datasets (the `fix` directory). |
| `ufs_utils` | Path to this workflow on the host. |
| `shield_image`, `fregrid_image`, `preprocess_image` | Apptainer images for each stage. |
| `containers_root` | Directory holding the container images. |
| `shield_root` | Root of the shared SHiELD installation. |
| `archive_root` | Root of the archive tree. |
| `shield_exe` | Optional path to a native SHiELD executable. Required for multi-node native runs. |
| `container_bindpath` | Host paths bound into the containers. |
| `modules` | Host modules loaded for the job. |

Environment variables such as `$USER` and `$HOME` are expanded.

### 6.2 SLURM submission

| Key | Default | Meaning |
| --- | --- | --- |
| `walltime` | 24 | Wall time in hours. |
| `n_nodes` | 4 | Nodes requested. |
| `n_cpus` | 192 | Total tasks requested. |
| `n_cpus_per_task` | 1 | CPUs per task. |
| `partition` | batch | SLURM partition. |
| `mem` | 0 | Total job memory in GB. 0 uses the scheduler default. |
| `constraint_node` | false | Apply a core-count node constraint. |
| `exclusive_node` | false | Request exclusive node access. |
| `logfile` | shield_driver | Base name of the driver log written in the case directory. |

Tasks per node are computed as `n_cpus // n_nodes`. When `mem` exceeds twice the
task count, a per-CPU or per-job memory flag is derived.

### 6.3 Case metadata

| Key | Meaning |
| --- | --- |
| `case_name` | Case identifier. Null falls back to the directory name. |
| `description` | Short experiment label. |
| `fv3_debug` | Verbose model diagnostics. |
| `archive_data` | Archive outputs after the final segment. |
| `merge_freq` | Merge frequency for regridded output in restart segments (see Section 14). |

### 6.4 Execution control

| Key | Meaning |
| --- | --- |
| `init_datetime` | Initialization cycle, UTC, in `YYYYMMDDHHZ` form, for example `2026031200Z`. |
| `run_nhours` | Integration length of one segment in hours. |
| `forecast_hour` | Lead hour of the source dataset used for initial conditions. 0 selects the analysis. |
| `resubmit` | Number of sequential resubmissions. The run has `resubmit + 1` segments (see Section 15). |
| `continue_run` | Managed internally by the driver. The initial segment is a cold start and later segments are warm starts. |

`c_res` accepts either an integer such as `96` or the labeled form `C96`.

### 6.5 Initial conditions and preprocessing

| Key | Default | Meaning |
| --- | --- | --- |
| `generate_ic_data` | true | Generate the grid and initial conditions during preprocess. |
| `external_ic_dir` | null | Path to a pre-staged case bundle used when `generate_ic_data` is false. |
| `preprocess_only` | false | Stage the complete grid and initial conditions, then exit. |
| `preprocess_grid_only` | false | Generate the grid only, then exit (see Section 9). |
| `preprocess_orog_only` | false | Generate orography only, then exit (see Section 10). |

Setting `preprocess_grid_only` or `preprocess_orog_only` implies
`preprocess_only`.

### 6.6 Horizontal grid

| Key | Meaning |
| --- | --- |
| `c_res` | Cubed-sphere face resolution. Approximate spacing: C96 ~ 100 km, C192 ~ 50 km, C384 ~ 25 km, C768 ~ 13 km, C3072 ~ 3 km. |
| `gtype` | `uniform`, `stretch`, `nest`, `regional_gfdl`, or `regional_esg`. |
| `target_lon`, `target_lat` | Grid center used for stretched and regional grids. |
| `stretch_factor` | Schmidt stretching coefficient. Values greater than 1 refine the target region. |

Nested grids, active when `gtype: nest`:

| Key | Meaning |
| --- | --- |
| `refine_ratio` | Refinement ratio for each nest relative to its parent. A list defines multiple nests. |
| `parent_tile` | Parent cubed-sphere tile, 1 to 6. |
| `halo` | Halo width for the nest boundary exchange. |
| `lon_min`, `lon_max`, `lat_min`, `lat_max` | Bounding box for each nest. Lists are required for multiple nests. |

When `gtype: nest`, the target longitude and latitude are set to the center of
the first bounding box. The nest layout is classified automatically from the
bounding boxes. If each box is contained inside its predecessor, the layout is
telescoping and refinement ratios compound. Otherwise the nests are treated as
independent nests on the same parent grid. For telescoping nests the effective
refinement of nest `i` is the product of ratios up to and including `i`.

Regional ESG grids, active when `gtype: regional_esg`:

| Key | Meaning |
| --- | --- |
| `idim`, `jdim` | Zonal and meridional grid points. |
| `delx`, `dely` | Supergrid spacing in degrees. |

### 6.7 Vertical grid and physics

| Key | Meaning |
| --- | --- |
| `levels` | Number of hybrid sigma-pressure levels. |
| `do_deep` | Deep convection parameterization. Disable for grid spacing below about 4 km. |

### 6.8 Time stepping

| Key | Meaning |
| --- | --- |
| `dt_atmos` | Atmospheric time step in seconds. Null triggers automatic selection. |
| `dt_ocean` | Ocean coupling time step in seconds. |
| `k_split` | Remap split counts per domain. Length must equal `n_nests + 1`. |
| `n_split` | Acoustic substep counts per domain. Length must equal `n_nests + 1`. |

See Section 12 for the automatic values and the relations between these
quantities.

### 6.9 Surface and orography

| Key | Meaning |
| --- | --- |
| `lake_cutoff` | Land and water fractional threshold. |
| `add_lake` | Activate the lake model. |
| `make_gsl_orog` | Generate the GSL orography fields used by the gravity wave drag scheme. |

### 6.10 Ensembles

| Key | Meaning |
| --- | --- |
| `ensemble_run` | Enable a multi-member ensemble. |
| `n_ensembles` | Number of members. Must be at least 1 when `ensemble_run` is true. |
| `skip_ensembles` | Member indices to omit, for example `[1, 3, 5]`. |

### 6.11 Land surface perturbations

The `sm_perturbations` block applies controlled perturbations to soil-state
variables. The schema and methods are documented in Section 11.

## 7. Static runtime datasets (`fix/`)

The `fix` tree holds the climatologies, lookup tables, orography inputs, and
other static datasets required by a run, resolved from `fix_src`. On Oscar these
are already staged, so manual download is normally unnecessary. If a dataset is
missing, the NOAA fix bundle is the reference source:

https://noaa-nws-global-pds.s3.amazonaws.com/index.html#fix/

The soil moisture climatology used by the perturbation module is
`fix/era5/sm_monthly_1950_2025.nc`.

## 8. Initial conditions and preprocessing

When `generate_ic_data: true`, the preprocess stage generates the grid and
orography, then converts external model data to FV3 initial conditions with
`chgres_cube`. Atmospheric and surface fields are drawn from GFS and HRRR GRIB2
data. Source data are downloaded with retry and multi-source fallback:

* GFS: NOAA AWS S3 and NCAR GDEX.
* HRRR: NOAA AWS S3 and Google Cloud Storage.

The default source assignment converts the global domain from GFS for both
atmospheric and surface fields. For nested domains the default converts
atmospheric fields from HRRR and surface fields from GFS. HRRR covers the CONUS
region only, and the workflow checks whether a nest lies inside the HRRR domain
before assigning it. A nest outside HRRR coverage falls back to another model.

To override the source assignment, place a case-local `chgres_cube.yaml` in the
case directory. Supported top-level keys are `global`, `regional`, and
`nestXX`, where `nestXX` is a template expanded across nests, or explicit
`nest02`, `nest03`, and so on. Each entry sets `external_model` and the
`convert_atm`, `convert_sfc`, and `convert_nst` switches. A field category
supplied by two models for the same domain is expressed with an
`external_models` list. See [configs/chgres_cube.yaml](configs/chgres_cube.yaml)
for annotated examples.

To stage only the grid and initial conditions without running the model, set
`preprocess_only: true`. The job exits after preprocessing.

### External initial condition bundles

If you already have a pre-generated case bundle, set `generate_ic_data: false`
and point `external_ic_dir` at that bundle. The workflow expects a staged case
directory containing the files it reads at startup, not a single NetCDF file
that it edits in place. Use the repository code as the handoff point when
building a custom conversion pipeline rather than mutating the default files
directly.

```yaml
generate_ic_data: false
external_ic_dir: /path/to/prestaged_case
```

## 9. Modifying the grid

Grid generation is driven from `py_scripts/fv3_make_grid.py` and staged through
a modification directory. The generator copies user-supplied files verbatim when
a non-empty modification directory is present, so the procedure is stage, edit,
and re-inject.

1. Set `preprocess_grid_only: true` and submit. The workflow generates the grid
   and mosaic, stages them into the case-local `IC/grid` directory, and exits.
   The driver log reports the staging path.
2. Copy the staged files to a backup directory. The staged content is
   `C{c_res}_grid.tile*.nc` and `C{c_res}_mosaic.nc`.
3. Edit the tile files. If you change tile geometry, keep the mosaic consistent,
   because the mosaic is staged and re-injected from the same directory.
   Preserve filenames exactly.
4. Set `preprocess_grid_only: false`, keep `generate_ic_data: true`, and
   resubmit. The edited grid is copied through without regeneration.

## 10. Modifying orography

Orography generation is driven from `py_scripts/fv3_make_orog.py` and follows the
same stage, edit, and re-inject pattern as the grid. Orography is generated after
the grid, so a grid must exist first.

1. Set `preprocess_orog_only: true` and submit. The workflow stages the
   orography into the case-local `IC/orography` directory and exits. The
   `shield_driver*.log` file reports the staging path.
2. Copy the staged files to a backup directory. The staged content is
   `oro.C{c_res}.tile*.nc`, and the GSL variants when `make_gsl_orog: true`.
3. Edit the orography. Modify both the `orog_raw` and `orog_filt` variables
   inside each tile file, and update them together to preserve consistency.
   `orog_raw` is the unfiltered surface height and `orog_filt` is the filtered
   height used by the dynamical core. Preserve filenames exactly.
4. Set `preprocess_orog_only: false`, keep `generate_ic_data: true`, and
   resubmit.

The topography filter runs after orography generation for uniform and stretched
grids. Inject edited orography as the staged tile files rather than relying on
the filter to preserve raw edits.

## 11. Soil moisture perturbations

Soil moisture perturbations are applied at model initialization and at the start
of each restart segment by `py_scripts/sm_perturbations.py`. They act on the
surface restart files `sfc_data.tile*.nc`, and `sfc_data.nest{NN}.tile*.nc` for
nested tiles. Target variables are `smc` (total volumetric soil moisture), `slc`
(liquid volumetric soil moisture), and `stc` (soil temperature). Volumetric soil
moisture is clipped to the interval $[0.01, 0.99]\ \mathrm{m^3\,m^{-3}}$. The
frozen fraction is held fixed by keeping the ice content
$\mathrm{ice} = \mathrm{smc} - \mathrm{slc}$ constant and reconstructing `slc`
after any `smc` edit. The workflow writes both an original and a perturbed copy
of each file into `IC/perts`, so the unperturbed state is recoverable.

### Schema

```yaml
sm_perturbations:
  target_var: smc          # smc, slc, or stc
  soil_layers: [0, 1, 2]   # integer or list of integers
  tiles: [1, 2, 3, 4, 5, 6]
  method: mean_shift       # string or list of strings
  apply_on_restarts: 0     # None, "all", int, or list of ints
```

Required keys are `target_var`, `soil_layers`, `tiles`, and `method`. If
`apply_on_restarts` is absent, no perturbation is applied. Multiple methods in a
list are applied in order.

### Methods

Let $X$ be the soil field in a layer, $\mu$ the mean over valid points, and
$\sigma$ the standard deviation.

Standard deviation shift, `std_shift`, with $k$ from `n_sigma`:
$$X' = X + k\,\sigma$$

Mean scaling, `mean_shift`, with $s$ from `mean_scale`:
$$X' = X\,(1 + s)$$

Anomaly scaling, `anom_shift`, with $a$ from `anom_scale`:
$$X' = \mu + (1 + a)\,(X - \mu)$$

Constant fill, `constant_fill`, with $c$ from `fill_value`, or the field mean
when `fill_value` is the string `mean`:
$$X' = c$$

Climatological replacement, `climo_mean`, replaces valid points with the monthly
climatological mean regridded to the cubed sphere. The month is selected from
`init_datetime`.

### Cross-segment behavior

Two behaviors act across restart segments and are mutually exclusive.

Nudging, `do_nudge: true`, relaxes the field toward a reference with weight
$\alpha = \Delta t / \tau$ clipped to $[0, 1]$:
$$X' = (1 - \alpha)\,X + \alpha\,X_\mathrm{ref}$$

Here $\Delta t$ is `run_nhours` and $\tau$ is `tau_hours`, default 24 hours. The
reference is the previous perturbed segment, or the climatological mean when
`use_climo: true`. Holding, `do_hold: true`, carries the perturbed state forward
from the previous segment without recomputing.

### Optional keys

`n_sigma`, `mean_scale`, `anom_scale`, `fill_value`, `use_climo`, `do_nudge`,
`do_hold`, `climo_file`, `tau_hours`, `apply_on_restarts`. A method that requires
a parameter raises an error if the parameter is missing.

## 12. Time stepping

When `dt_atmos`, `k_split`, or `n_split` are null, the workflow derives them from
a base table indexed by resolution.

| `c_res` | `dt_atmos` (s) | `k_split` | `n_split` |
| --- | --- | --- | --- |
| 48 | 1200 | 2 | 6 |
| 96 | 720 | 2 | 6 |
| 192 | 450 | 2 | 6 |
| 384 | 360 | 2 | 6 |
| 768 | 180 | 2 | 8 |
| 1152 | 120 | 2 | 8 |
| 3072 | 90 | 2 | 10 |

Resolutions outside the table are estimated by a log-log fit of `dt_atmos`
against `c_res` and snapped to a value that divides 3600 seconds. For nested
runs the finest domain sets `dt_atmos`, and each domain receives split counts
sized to its resolution. The dynamics and acoustic time steps follow

$$\Delta t_\mathrm{dyn} = \frac{\Delta t_\mathrm{atmos}}{k_\mathrm{split}},
\qquad
\Delta t_\mathrm{acoustic} = \frac{\Delta t_\mathrm{atmos}}{k_\mathrm{split}\,n_\mathrm{split}}$$

where $\Delta t_\mathrm{atmos}$ is the atmospheric time step in seconds,
$k_\mathrm{split}$ is the remap split count, and $n_\mathrm{split}$ is the
acoustic substep count per remap split. Supplied `k_split` and `n_split` must be
lists of length `n_nests + 1`, with the first entry for the global grid and the
remaining entries for the nests in order. The atmospheric time step is CFL
constrained and should decrease as horizontal resolution or refinement
increases.

## 13. Process decomposition (PEs and layout)

Process counts and domain layouts are computed automatically from the grid.
For a uniform grid the total process count is the largest multiple of 6 not
exceeding `n_cpus`, distributed equally across the six tiles. For nested runs
the workflow distributes processes across the global grid and each nest by
minimizing the largest estimated per-domain time,

$$T_g \sim \frac{w_g}{P_g},
\qquad
w_g = N_g\,k_{\mathrm{split},g}\,n_{\mathrm{split},g}$$

where $T_g$ is the estimated time for domain $g$, $P_g$ is its process count,
$w_g$ is a work weight, $N_g$ is the number of horizontal cells, and
$k_{\mathrm{split},g}$ and $n_{\mathrm{split},g}$ are its split counts. Global
process counts are multiples of 6. Nest process counts are drawn from a set that
keeps subdomain aspect ratios no more elongated than 2 to 1. The per-domain
layout is chosen to make local subdomains as close to square as possible. The
I/O layout is set to one by one and the physics block size to 32.

To override the automatic allocation, place a case-local `input.nml` or
`input.yaml` in the case directory that sets `grid_pes` under `fv_nest_nml`. The
listed values become the per-domain process counts.

## 14. Diagnostics and regridded output

Output frequency and the reported variable set are controlled by a `diag_table`.
Place a case-local `diag_table` in the case directory to override the default in
`configs/diag_table`. The default defines three streams: `grid_spec` and
`atmos_static` written once, and `fv3_hist` written hourly. The variable
reference list is in [configs/diag_field.csv](configs/diag_field.csv).

After the model runs, `fregrid` remaps native cubed-sphere history to a
latitude-longitude grid. Regridded files are named by domain, `global` for the
global grid and `nest02`, `nest03`, and so on for nests, where nest tile 7 maps
to `nest02`.

The `merge_freq` key controls how per-segment regridded files are combined.

* `-1` merges the whole run into one file per stream and grid on the final
  segment.
* `0` disables merging and retains one file per segment.
* `n` merges every `n` segments and flushes any remainder on the final segment.

## 15. Restarts and segmented runs

A run is divided into `resubmit + 1` segments, each of length `run_nhours`. The
first segment is a cold start produced by the initial driver. Each later segment
is a warm start produced by the restart driver, which resumes from the previous
segment restart files. The driver records a configuration checksum in
`state.yaml` and verifies it at the start of every segment, so a restart cannot
proceed against a changed grid or initial time. Segments are resubmitted
automatically until the maximum index is reached.

## 16. Ensembles

Set `ensemble_run: true` and `n_ensembles` to submit an ensemble. Each member is
submitted as an independent job with its own working directory `memNN` and its
own log. Use `skip_ensembles` to omit specific members. Ensemble members can be
combined with soil moisture perturbations to build spread.

## 17. Archiving

When `archive_data: true`, the final segment copies the regridded output to the
archive tree under `archive_root`, writes a copy of `state.yaml` and the model
log, and compresses the case directory into `case.tar.gz`. After archiving, the
case directory is replaced by a symlink to the archived location.

## 18. Example cases

### Global uniform case

```yaml
description: C96 control run
init_datetime: "2026031200Z"
run_nhours: 6
c_res: 96
gtype: uniform
levels: 64
generate_ic_data: true
preprocess_only: false
archive_data: true
shield_exe: /path/to/SHiELD_nh.prod.64bit.gnu.x
walltime: 12
n_nodes: 2
n_cpus: 96
n_cpus_per_task: 1
partition: batch
```

### Stretched global case

```yaml
description: Stretched C96 over central North America
init_datetime: "2026031200Z"
run_nhours: 24
c_res: 96
gtype: stretch
stretch_factor: 2.5
target_lon: -96
target_lat: 39
levels: 64
generate_ic_data: true
shield_exe: /path/to/SHiELD_nh.prod.64bit.gnu.x
walltime: 12
n_nodes: 2
n_cpus: 96
partition: batch
```

### Nested case

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
parent_tile: 6
halo: 3
generate_ic_data: true
shield_exe: /path/to/SHiELD_nh.prod.64bit.gnu.x
walltime: 12
n_nodes: 2
n_cpus: 96
partition: batch
```

### Regional ESG case

```yaml
description: Regional ESG domain
init_datetime: "2026031200Z"
run_nhours: 12
c_res: 3072
gtype: regional_esg
target_lon: -96
target_lat: 39
idim: 200
jdim: 200
delx: 0.0585
dely: 0.0585
halo: 3
levels: 64
generate_ic_data: true
shield_exe: /path/to/SHiELD_nh.prod.64bit.gnu.x
walltime: 12
n_nodes: 4
n_cpus: 192
partition: batch
```

## 19. Compiling a custom SHiELD executable

If you need a custom binary, build it from the SHiELD source tree and point
`shield_exe` at the result. The workflow uses the native executable when
`shield_exe` is set and otherwise falls back to the container image. Multi-node
native runs require `shield_exe`.

```bash
git clone -b oscar https://github.com/biosphereNclimate/SHiELD_build.git
cd SHiELD_build
./CHECKOUT_code
git submodule update --init mkmf
```

On newer glibc systems, patch `SHiELD_SRC/FMS/affinity/affinity.c` so the local
`gettid` helper does not conflict with the glibc definition. Remove the
duplicate `static` qualifier from the local declaration.

Before compiling, update `SHiELD_build/site/environment.gnu.sh` so the GNU build
loads the required modules:

```bash
module load hpcx-mpi
module load netcdf-mpi
module load libyaml
module load cmake
```

Then build:

```bash
./Build/COMPILE 64bit gnu pic
```

Set the executable path in `run_config.yaml`:

```yaml
shield_exe: /path/to/SHiELD_nh.prod.64bit.gnu.x
```

## 20. Quick start and submission notes

1. Create a case directory.
2. Write `run_config.yaml` into it.
3. Add optional case-local overrides such as `diag_table`, `chgres_cube.yaml`,
   or `input.nml`.
4. Run `case_submit.sh` from the case directory.

Recommended case-local launcher:

```bash
#!/bin/bash -l
"/path/to/ufs_py/case_submit.sh"
```

Deactivate any active conda environment and use a clean shell before submitting,
so the workflow starts from a predictable environment. Place the launcher in the
case directory and run it there so the workflow reads the local `run_config.yaml`
and any case-local files.