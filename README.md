# pyUFS HPC Run Guide

`pyUFS` is a Python workflow for setting up and running GFDL SHiELD cases on Oscar and similar HPC systems that lack the native UFS utilities layout. A thin shell wrapper hands off to Python, which checks the case configuration, builds the launch environment, generates or stages the grid and initial conditions, runs SHiELD, regrids the output and copies results back to the case directory.

## SHiELD releases

One checkout supports two releases. `202411` is the default.

| | `202411` (default) | `202604` |
| --- | --- | --- |
| Core, physics, drivers | `FV3-202411-public` | `FV3-202604-public` |
| FMS / FMSCoupler | `2024.03` / `2024.03.01` | `2026.01` / `2026.01` |
| Coupler driver | `FMSCoupler/SHiELD` (`coupler_main.F90`) | `FMSCoupler/full` |
| `coupler_nml` time-step key | `dt_ocean` | `dt_cpld` |
| Default configuration | `config/202411/run_config.yaml` | `config/202604/run_config.yaml` |
| Namelist templates | `templates/202411/` | `templates/202604/` |

Set `shield_release: "202604"` in the case `run_config.yaml`, or omit it for the default. The key selects the default configuration, the namelist templates, the `coupler_nml` time-step key and the restart checksum overrides (all resolved in `src/release.py`). In `run_config.yaml` the time-step key is always `dt_cpld`; it is written as `dt_ocean` for `202411`. The release must match the executable (Section 19). Case files written for the old `202411` branch need `dt_ocean` renamed to `dt_cpld`.

## Contents

1. Reference documentation
2. Requirements and runtime environment
3. Repository layout
4. Workflow
5. Case setup
6. Configuration reference
7. Static runtime datasets (`fix/`)
8. Initial conditions and preprocessing
9. Modifying the grid
10. Modifying orography
11. Soil moisture perturbations
12. Time stepping
13. Process decomposition
14. Diagnostics and regridded output
15. Restarts
16. Ensembles
17. Archiving
18. Example cases
19. Compiling a SHiELD executable
20. Quick start
21. PAMIP SST and sea-ice boundary conditions
22. Monthly runs
23. Plugins

## 1. Reference documentation

- SHiELD: https://www.gfdl.noaa.gov/shield/
- FV3: https://www.gfdl.noaa.gov/fv3/fv3-documentation-and-references/
- FV3 namelist guide: https://www.gfdl.noaa.gov/wp-content/uploads/2017/09/fv3_namelist_Feb2017.pdf
- UFS_UTILS: https://noaa-emcufs-utils.readthedocs.io/en/latest/ufs_utils.html
- UFS Weather Model: https://ufs-weather-model.readthedocs.io/en/develop/Introduction.html
- FMS: https://noaa-gfdl.github.io/FMS/md_docs_doxygenGuide.html

## 2. Requirements and runtime environment

Three Apptainer images are set in `run_config.yaml`. SHiELD can instead run from a native executable.

- `preprocess.sif` (`docker://gfdlfv3/preprocessing`) holds UFS_UTILS at `/UFS_UTILS/exec` and the Python environment of `tools/env.yaml`. It runs `src/driver.py`.
- `shield.sif` (`docker://gfdlfv3/shield`, the 202411 build) runs the model when no native `shield_exe` is used.
- `fregrid.sif` (`docker://gfdlfv3/fre-nctools` plus the same Python environment) runs `src/fv3_regrid.py`. Regridding uses xarray and xESMF (`src/pyfregrid.py`), not the FRE-NCtools `fregrid` program.

`tools/install_images.sh` builds all three into `containers_root`; the image keys must point there. If an image is missing, `case_run.sh` builds it first (log: `image_build.log` in the case directory). To build by hand:

```bash
CONTAINERS_DIR=/path/to/containers TMP_DIR=/scratch bash tools/install_images.sh
```

The script copies the Open MPI help texts from the fre-nctools image into the other two. Images use the `latest` tag; set `SHIELD_TAG`, `PREPROCESS_TAG` or `FREGRID_TAG` to change it. Sources, SHA-256 sums and conda package lists go to `$CONTAINERS_DIR/manifest`. A build needs internet access (Docker Hub, repo.anaconda.com, conda-forge) and several GB in `TMP_DIR`.

`modules` lists the host modules loaded before a native `shield_exe` runs (default `hpcx-mpi`, `netcdf-mpi`, `libyaml`); it is unused with the container. `case_submit.sh` needs a host `python3` with PyYAML. The container Python packages are listed in `tools/env.yaml`.

The scheduler is SLURM. `launchers/sbatch.sh` submits `launchers/case_run.sh`. The case runs directly in `case_root/<parent>/<case_name>`; set `case_root` to run on scratch. The run directory must not contain the submission directory, because it is removed after archiving (`case_submit.sh` refuses that layout). Every `container_bindpath` entry must exist on the compute node, or Apptainer does not start.

## 3. Repository layout

| Path | Contents |
| --- | --- |
| `case_submit.sh` | Entrypoint; calls `launchers/case_submit.py`. |
| `case_continue.sh` | Entrypoint for resuming a failed case; calls `launchers/case_continue.py`. |
| `config/` | `<release>/run_config.yaml` (defaults and valid keys) and `examples/`. |
| `launchers/` | Submission and runtime scripts, `shield.launcher`. |
| `src/` | Core Python workflow code; `release.py` resolves the release. `fv3_plugins.py` is the only module that imports `plugins/`. |
| `plugins/` | Optional modules: `pamip` (PAMIP and external climatologies), `sm_perturbations` and `derived` (derived variables). See Section 23. |
| `templates/` | `<release>/` namelist templates (`input_nml.yaml`, `input_nestXX_nml.yaml`, `C3072.template.yaml`), plus shared tracer, diagnostic and chgres templates. |
| `tables/` | Diagnostic field reference and variable maps. |
| `tools/` | Build, dataset download (`get_pamip.py`, `get_amip.py`, `esa_sst.py`, `get_era5.py`), fix update and container helpers, `env.yaml`. |
| `tests/cases/` | HPC test cases, submitted with `tests/submit.sh`. |
| `bin/` | Link to `../bin`, the compiled executables (created by `tools/build_shield.py`). |
| `fix/` | Link to `../fix`, the static datasets (created by `tools/update_fix.py`). If `fix_src` is missing, `case_submit.sh` stops; link it with `ln -s /path/to/fix ../fix`. |
| `containers/` | Container images, ignored by Git. |

## 4. Workflow

1. Create a case directory with a `run_config.yaml`.
2. Run `case_submit.sh` from that directory. It deactivates any active conda environment and calls `launchers/case_submit.py`.
3. `case_submit.py` checks the configuration against the default key set, rejects unknown keys with a suggested fix, fills unset keys from the default file, resolves paths and SLURM flags into `CASE_*` variables (listed in its docstring) and submits `case_run.sh` through `sbatch.sh`. Ensemble members are separate jobs (`<parent>.<case>.MEMnn`).
4. `case_run.sh` prepares the case directory, runs the preprocess container, launches SHiELD, runs the regridding stage, then resubmits the next segment or archives the case. It keeps `TMP` after a failure and clears it when the next job starts.
5. In the preprocess container, `src/driver.py` calls the initial driver for the first segment and the restart driver for later ones. The initial driver generates the grid and orography, converts initial conditions, plots the grid, decomposes processes, writes the namelists, tables and fixed files, writes the run script and then runs the plugins (Section 23: soil moisture perturbations, PAMIP and external boundary files), which edit the written namelists where they need to.

## 5. Case setup

Make a case directory and put a `run_config.yaml` in it. Submit from this directory so case-local overrides are picked up. After a run it holds `FIXED`, `GRID`, `IC`, `INPUT`, `LOGS`, `OUTPUT`, `RESTART`, `HIST` and `TMP`, depending on the settings. A `run` symlink points to the case directory during a job and to the archive afterwards (`memNN` links for ensemble members).

## 6. Configuration reference

Start from [config/202411/run_config.yaml](config/202411/run_config.yaml) (or `config/202604/`) and override only what you need. Unknown keys are rejected. Variables such as `$USER` and `$HOME` are expanded in paths.

### 6.1 System paths

| Key | Description |
| --- | --- |
| `fix_src` | Static dataset tree (the `fix` directory); must be writable (Section 7). |
| `shield_image`, `fregrid_image`, `preprocess_image` | Full paths of the images (Section 2). |
| `containers_root` | Where `tools/install_images.sh` writes images. |
| `case_root` | Root of the case tree, `case_root/<parent>/<case_name>`. Default `/jobtmp/$USER/gfdl_shield`. |
| `container_bindpath` | Host paths bound into the containers (list or comma-separated string). Each must exist on the compute nodes. |
| `archive_root` | Archive base; outputs go to `<archive_root>/<parent>/<case_name>`. |
| `shield_release` | `"202411"` (default) or `"202604"`. An unquoted `202604` is accepted; any other value stops the submission. |
| `shield_exe` | Full path of a native executable, used as written. Required for multi-node runs; `""` selects the single-node container. If the file is missing, a single-node `202411` case uses the container with a warning; any other case stops. `null` uses the repository default. |
| `modules` | Host modules loaded before a native executable runs. |

### 6.2 SLURM submission

| Key | Default | Description |
| --- | --- | --- |
| `walltime` | 24 | Wall time in hours (`--time=<h>:00:00`). |
| `n_nodes` | 4 | Nodes (`--nodes`). |
| `n_cpus` | 192 | Tasks, rounded down to a multiple of `n_nodes` (`--ntasks`, `--ntasks-per-node`). |
| `n_cpus_per_task` | 1 | `--cpus-per-task`. |
| `partition` | batch | `--partition`. |
| `mem` | 0 | Total memory in GB. `--mem=<mem>g` on one node, `--mem-per-cpu=<mem // tasks>g` on several; omitted when not above 2 GB per task. |
| `constraint_node` | null | `--constraint=<value>`; omitted when null. |
| `exclusive_node` | false | `--exclusive`. |
| `sbatch_options` | "" | Extra raw options, appended unchanged, e.g. `"--account=myproj --mail-type=END"`. |
| `logfile` | shield_driver | Base name of the driver log. `.log` is added (a trailing `.log` is ignored); ensembles insert the member number, `<logfile>.NN.log`. |

The job name is `<parent directory>.<case directory>`, with `.MEMnn` for ensemble members. The job writes its own log, so `sbatch` gets `--output=/dev/null`. A multi-node job runs the native `shield_exe`; a single-node job may use the container. Several `tests/cases/*/run_config.yaml` files show the SLURM keys in a `# SBATCH OPTIONS` block.

### 6.3 Case metadata and execution control

| Key | Description |
| --- | --- |
| `case_name` | Case identifier; null uses the directory name. |
| `description` | Optional short experiment label. |
| `fv3_debug` | Verbose model diagnostics. |
| `archive_data` | Archive outputs after the final segment (Section 17). |
| `merge_freq` | Merging of regridded output across segments (Section 14). |
| `init_datetime` | Cycle of the initial-condition data, UTC, `YYYYMMDDHHZ` with hour 00, 06, 12 or 18, e.g. `2026031200Z`. |
| `forecast_hour` | Lead hour of the `init_datetime` forecast used for initial conditions (0 is the analysis). If non-zero, the model clock starts at `init_datetime + forecast_hour`, and regional boundaries come from the same cycle at leads `forecast_hour`, `+3`, ... Regional runs therefore need a multiple of 3, and the last lead must exist (GFS: hourly to f120, 3-hourly to f384). The state records `ic_cycle`. |
| `run_length`, `run_length_units` | Segment length; units are `hours` (default), `days`, `months` or `years`. Months are calendar months (`coupler_nml months`; years as 12 months) and need a start day of 28 or earlier. |
| `resubmit` | Number of resubmissions; the run has `resubmit + 1` segments. |
| `continue_run` | Set internally (first segment cold, later ones warm). |
| `preprocess_dask_scheduler` | `processes` (default) or `synchronous` (for debugging). `threads` is not supported. |

### 6.4 Initial conditions and preprocessing

| Key | Default | Description |
| --- | --- | --- |
| `generate_ic_data` | true | Generate the grid and initial conditions. |
| `external_ic_dir` | null | Pre-staged bundle, used when `generate_ic_data` is false (Section 8.5). |
| `preprocess_only` | false | Stage the grid and initial conditions, then exit. |
| `preprocess_grid_only` | false | Generate and stage the grid, then exit (Section 9). |
| `preprocess_orog_only` | false | Generate and stage orography, then exit (Section 10). |

The grid-only and orography-only options imply `preprocess_only`. The UFS_UTILS options (`run_*`) are in Section 8.6.

### 6.5 Horizontal grid

| Key | Description |
| --- | --- |
| `c_res` | Cubed-sphere resolution: `C48`, `C96`, `C192`, `C384`, `C768`, `C1152` or `C3072` (an integer such as `96` is the same). Spacing: C96 ~100 km, C192 ~50 km, C384 ~25 km, C768 ~13 km, C3072 ~3 km. |
| `gtype` | `uniform`, `stretch`, `nest`, `regional_gfdl` or `regional_esg`. |
| `target_lon`, `target_lat` | Grid centre for `stretch` and `regional_esg`. For `nest` and `regional_gfdl` it is the centre of the (first) box. |
| `stretch_factor` | Schmidt stretching; values above 1 refine the target region. |
| `refine_ratio` | Refinement ratio of each nest relative to its parent; a list defines several nests. |
| `parent_tile` | Parent tile, 1 to 6: one value for all nests, or one per same-level nest. The grid is rotated so the centre of the first box lies on tile 6 (the usual value). A telescoping chain uses the first entry for its outermost nest. |
| `halo` | Halo width for the nest boundary exchange; regional grids need 3. |
| `lon_min`, `lon_max`, `lat_min`, `lat_max` | Bounding box of each nest, one value per `refine_ratio` entry. A different count stops preprocessing. |
| `idim`, `jdim`, `delx`, `dely` | `regional_esg` only: zonal and meridional points and supergrid spacing in degrees (model spacing is twice `delx`; 0.0585 is about 13 km). |

All nests are generated by one `make_hgrid --nest_grids` call. Each nest covers its box with the smallest block of whole parent cells and must stay `halo` parent cells inside its parent tile. If each box lies inside its predecessor the layout is telescoping, and the effective refinement of nest `i` is the product of the ratios up to `i`; otherwise the nests are independent on the same parent grid. A box may cross 0 or 180 degrees longitude. `regional_gfdl` needs the box keys, is centred on the box and refined by the first `refine_ratio`. For both regional types `c_res` is replaced by the equivalent global resolution (`global_equiv_resol`).

### 6.6 Vertical grid, physics, time stepping, surface

| Key | Description |
| --- | --- |
| `levels` | Number of hybrid sigma-pressure levels. |
| `do_deep` | `false` (default) turns deep convection off on domains with spacing of 4 km or less; `true` keeps it everywhere. |
| `dt_atmos`, `dt_cpld`, `k_split`, `n_split` | Time stepping (Section 12). |
| `lake_cutoff` | Land and water fraction threshold. |
| `add_lake` | Add lake fraction and depth (GLDB v2) to the orography; `uniform` and `regional_gfdl` only. It does not switch on a lake model. |
| `make_gsl_orog` | Generate `oro_data_ls` and `oro_data_ss`. SHiELD physics does not read them. |
| `use_modern_diag` | Use a case-local `diag_table.yaml` (required); `data_table.yaml` is optional. |
| `pack_output` | Pack final output as int16 with `scale_factor` and `add_offset`. |
| `derived_vars_module` | Path to a Python file whose `derive` list adds derived variables (Section 14). |

### 6.7 Ensembles and perturbations

| Key | Description |
| --- | --- |
| `ensemble_run`, `n_ensembles`, `skip_ensembles` | Ensemble switch, member count (at least 1) and member indices to omit (Section 16). |
| `sm_perturbations` | Soil-state perturbation block (Section 11). |
| `pamip_experiment`, `pamip_baseline`, `pamip_fields`, `external_sst`, `external_sic`, `external_mask` | PAMIP and external climatologies (Section 21). |

## 7. Static runtime datasets (`fix/`)

The `fix` tree holds climatologies, lookup tables, orography inputs and other static datasets, resolved from `fix_src`. The reference source is https://noaa-nws-global-pds.s3.amazonaws.com/index.html#fix/. Missing files are fetched at run time: `src/fv3_update_fix.py` looks for each file

1. in `fix_src` under the workflow name;
2. in `fix_src` under the NOAA name, which it then links (e.g. `am/fix_co2_update/global_co2historicaldata_2020.txt` for `am/co2historicaldata_2020.txt`);
3. in the NOAA bucket, newest version directory first (so the binary orography inputs, present only in `orog/20231027`, are found).

Downloads go to `fix_src` under the NOAA name with a link under the workflow name, so `fix_src` must be writable. A run stops only for a required file found nowhere. `mld_DR003_c1m_reg2.0.grb` and `fix/external/ERA5` are not in the bucket and must be staged by hand. `UFS_PY_FIX_REMOTE=0` disables remote lookups. The chgres_cube variable maps are fetched from a pinned UFS_UTILS commit (override with `UFS_PY_FIX_VARMAP_URL`).

For CO2 the observed record (`fix_co2_update`) is preferred over `co2dat_4a`, and both over the projection (`fix_co2_proj`). If no table exists for a model year, the latest earlier year is staged with a warning. The decadal volcanic aerosol tables (1850-1999) are linked into the working directory.

To update the whole tree in advance, use `tools/update_fix.py` (or `tools/sync_noaa_fix.sh`, which loads `awscli`). It mirrors `am`, `orog` and `sfc_climo` with `aws s3 sync`, keeps the newest version of each, removes pre-generated `C<res>` directories, recreates the CO2 and aerosol links without the `global_` prefix and downloads Cartopy Natural Earth data into `fix/external/CARTOPY`. It writes to `fix` in the parent of the repository by default; use `--dest` for another path and `--skip-carto` to skip Cartopy.

Data not from the bucket is in `fix/external` (`ERA5`, `PAMIP`, `AMIP`, `CARTOPY`); `update_fix.py` writes only `CARTOPY`. The soil moisture climatology is `fix/external/ERA5/sm_monthly_1950_2025.nc`.

## 8. Initial conditions and preprocessing

With `generate_ic_data: true`, preprocessing generates the grid and orography, then runs `chgres_cube` on GRIB2 data. GFS and HRRR are supported sources; downloads retry and fall back between sources (GFS: NOAA AWS S3, NCAR GDEX; HRRR: NOAA AWS S3, Google Cloud Storage). The source is chosen per domain, not in `run_config.yaml`.

### 8.1 Default source assignment

| Domain | Atmosphere | Surface | Conversions |
| --- | --- | --- | --- |
| Global | GFS | GFS | One |
| Regional | GFS | GFS | One |
| Nest inside HRRR coverage | HRRR | GFS | Two (HRRR atmosphere, GFS surface) |
| Nest outside HRRR coverage | GFS | GFS | One (implicit HRRR default falls back to GFS) |

The surface source stays GFS because HRRR lacks the required soil levels.

### 8.2 Override files

Overrides are flat `chgres_cube` mappings, one optional file per domain in the submission directory: `chgres_cube.yaml` (global or regional) and `chgres_cube_nest{NN}.yaml` for nests (`nest02` is the first nest, `nest03` the second). `fort.41` and `fort_nest{NN}.41` are also read. Keys may be flat or inside one `config` group. Copy `templates/chgres_cube.yaml` as a start: every key is null, meaning "keep the workflow value", so set only what you change. An unknown key stops preprocessing with an error naming the file and keys. Land-surface defaults follow UFS_UTILS except `tg3_from_soil: true` (deep-soil temperature from the lowest soil layer of the source).

`external_model` (`GFS` or `HRRR`) selects the source:

- No `external_model` in a nest file: try HRRR, fall back to GFS outside HRRR coverage.
- `external_model: HRRR`: require HRRR; outside coverage this raises `ValueError`. Regional domains may use it with `chgres_cube.yaml`; for a global domain it is an error.
- `external_model: GFS`: force GFS.

The conversion switches `convert_atm`, `convert_sfc` and `convert_nst` are set by the planner on regional and nested domains and are dropped if given there. Other settings stay overridable, for example:

```yaml
external_model: HRRR
nsoill_out: 9
vgfrc_from_climo: false
```

### 8.3 Verifying the source

The resolved source is recorded in `state.yaml` under `ic_source` (keys `global`, `regional`, `nest02`, ..., each with `atm`, `sfc` and `nst`). A domain converted from several sources also gets one log per field group, e.g. `chgres_cube_nest02_atm.log` and `chgres_cube_nest02_sfc.log`.

### 8.4 Preprocessing only

`preprocess_only: true` stages the grid and initial conditions and exits.

### 8.5 External bundles

```yaml
generate_ic_data: false
external_ic_dir: /path/to/prestaged_case
```

The bundle must be a staged case directory with non-empty `FIXED`, `GRID` and `INPUT` plus `state.yaml`. Its `state.yaml` supplies only the grid and IC keys (`BUNDLE_KEYS` in `src/fv3_external_ic.py`); everything else, including segment settings and `pamip_experiment`, comes from the case `run_config.yaml`. The model start (`init_datetime` plus `forecast_hour`), `gtype` and `levels` must match the bundle.

### 8.6 Surface analysis update and coupled grids (UFS_UTILS)

Four UFS_UTILS programs can run after `chgres_cube` on a cold start with `generate_ic_data: true`. Each runs when its `run_*` key is true (all false by default; see section 5a of `config/<release>/run_config.yaml`) and follows the matching UFS_UTILS `ush` script.

- `run_global_cycle` runs `global_cycle` on `INPUT/sfc_data*`. It updates surface fields from the `namsfc` climatologies and the optional GRIB1 analyses `global_cycle_sst_file` (FNTSFA), `global_cycle_ice_file` (FNACNA) and `global_cycle_snow_file` (FNSNOA); a missing analysis keeps the first guess and climatology. Settings are those of the scripts (`DELTSFC=6`, `FSMCL(2:4)=60`, default `FSNOL=-2., FSNOS=99999.`; `global_cycle_vars` sets `CYCLVARS`). The six global tiles run as one 6-rank job and each nest as a 1-rank job. Global, stretched and nested grids only. Logs: `LOGS/preprocess/global_cycle_<domain>.log`.
- `run_emcsfc_snow` builds the snow analysis from `ims_snow_file` and the optional AFWA files (`afwa_snow_nh_file`, `afwa_snow_sh_file`, `afwa_snow_global_file`) with `emcsfc_snow2mdl` on the T1534 Gaussian grid and passes it to `global_cycle`.
- `run_emcsfc_ice_blend` blends `ims_ice_file` into the MMAB 5-minute concentration `five_min_ice_file` with `emcsfc_ice_blend` and passes the GRIB1 result to `global_cycle`. `cnvgrib`, `copygb2` and `copygb` (NCEP grib_util) must be on `PATH`; they are not in `tools/env.yaml`.
- `run_cpld_gridgen` writes the MOM6/CICE6 grids, SCRIP files and ocean mask to `IC/cpld_gridgen` for `cpld_gridgen_res` (`500`, `100`, `050`, `025`). MOM6 inputs come from `fix_src/mom6/<res>`. Other resolutions need `Ct.mx025_SCRIP.nc`, so generate 025 first. `ESMF_Scrip2Unstruct` must be available, and `cpld_gridgen_postwgts: true` also needs `ncremap` (NCO). SHiELD does not read these files.

The analyses are not in the NOAA GFS buckets, so stage them yourself. emcsfc outputs are kept in `IC/surface_analysis`. Settings are checked before the grid is generated.

## 9. Modifying the grid

For `uniform`, `stretch` and `nest` cases, a non-empty staged grid directory is copied in verbatim, so the procedure is stage, edit, re-inject. Regional grids do not stage.

1. Set `preprocess_grid_only: true` and submit. The grid and mosaic are staged in the case-local `IC/grid` and the job exits (the driver log gives the path).
2. Back up `C{c_res}_grid.tile*.nc` and `C{c_res}_*mosaic*.nc`.
3. Edit the tile files, keeping the mosaic consistent and the filenames unchanged.
4. Set `preprocess_grid_only: false`, keep `generate_ic_data: true` and resubmit.

Staged files are reused only while the grid settings recorded in `.grid_settings.yaml` are unchanged; otherwise the run stops and asks you to remove the directory or restore the settings.

## 10. Modifying orography

The same pattern applies, after the grid exists (`uniform`, `stretch`, `nest` only).

1. Set `preprocess_orog_only: true` and submit. Orography is staged in `IC/orography` and the job exits.
2. Back up `oro.C{c_res}.tile*.nc` (and the GSL variants if `make_gsl_orog: true`).
3. Edit `orog_raw` (unfiltered height) and `orog_filt` (filtered height used by the dynamical core) together, keeping the filenames.
4. Set `preprocess_orog_only: false`, keep `generate_ic_data: true` and resubmit.

The topography filter runs after generation on the global tiles; nest tiles keep the unfiltered orography. Inject edits as the staged tile files and do not rely on the filter to preserve raw edits.

## 11. Soil moisture perturbations

`plugins/sm_perturbations/soil_moisture.py` perturbs surface restart files (`sfc_data.tile*.nc`, `sfc_data.nest{NN}.tile*.nc`, `sfc_data.nc` for regional) at initialization and at the start of each restart segment. Targets are `smc` (total volumetric soil moisture), `slc` (liquid) and `stc` (soil temperature) on soil points. Volumetric moisture is clipped to $[0.01, 0.99]\,\mathrm{m^3\,m^{-3}}$. The ice content $\mathrm{smc} - \mathrm{slc}$ is held fixed and `slc` is rebuilt after any `smc` edit. Original and perturbed copies are written to `IC/perts`.

```yaml
sm_perturbations:
  target_var: smc          # smc, slc, or stc
  soil_layers: [0, 1, 2]   # integer or list
  tiles: [1, 2, 3, 4, 5, 6]
  method: mean_shift       # string or list (applied in order)
  apply_on_restarts: 0     # None, "all", int, or list of ints
```

`target_var`, `soil_layers`, `tiles` and `method` are required; without `apply_on_restarts` nothing is perturbed. Optional keys: `n_sigma`, `mean_scale`, `anom_scale`, `fill_value`, `use_climo`, `do_nudge`, `do_hold`, `climo_file`, `tau_hours`. A method that needs a missing parameter raises an error.

With $X$ the soil field in a layer, $\mu$ its mean over valid points and $\sigma$ its standard deviation:

| Method | Result | Parameter |
| --- | --- | --- |
| `std_shift` | $X' = X + k\,\sigma$ | `n_sigma` ($k$) |
| `mean_shift` | $X' = X\,(1 + s)$ | `mean_scale` ($s$) |
| `anom_shift` | $X' = \mu + (1 + a)\,(X - \mu)$ | `anom_scale` ($a$) |
| `constant_fill` | $X' = c$ | `fill_value` ($c$, or `mean` for the field mean) |
| `climo_mean` | monthly climatological mean regridded to the cubed sphere, for the month of the segment start | `climo_file` |

For `climo_mean`, climatology points outside the valid range are excluded when regridding, model points without a valid value keep their own, and climatology layers are matched to model layers by index (Noah: 0-10, 10-40, 40-100, 100-200 cm).

Two mutually exclusive options act across segments. Nudging (`do_nudge: true`) relaxes toward a reference with weight $\alpha = \Delta t/\tau$ clipped to $[0,1]$:

$$X' = (1 - \alpha)\,X + \alpha\,X_\mathrm{ref}$$

where $\Delta t$ is the previous segment length (h) and $\tau$ is `tau_hours` (default 24 h). The reference is the previous perturbed segment or, with `use_climo: true`, the mean of `climo_file` (default `fix/external/ERA5/sm_monthly_1950_2025.nc`). Holding (`do_hold: true`) carries the previous perturbed `target_var` layers forward without recomputing.

## 12. Time stepping

Null `dt_atmos`, `k_split` or `n_split` come from a base table by resolution.

| `c_res` | `dt_atmos` (s) | `k_split` | `n_split` |
| --- | --- | --- | --- |
| 48 | 1200 | 2 | 6 |
| 96 | 720 | 2 | 6 |
| 192 | 450 | 2 | 6 |
| 384 | 360 | 2 | 6 |
| 768 | 180 | 2 | 8 |
| 1152 | 120 | 2 | 8 |
| 3072 | 90 | 2 | 10 |

Other resolutions use a log-log fit of `dt_atmos` against `c_res`, snapped to a divisor of 3600 s. `dt_cpld` defaults to `dt_atmos`; before launch it must be a multiple of `dt_atmos`, and the segment length a multiple of `dt_cpld`. For nested runs the finest domain sets `dt_atmos` and each domain gets split counts sized to its resolution. Supplied `k_split` and `n_split` are lists of length `n_nests + 1` (global grid first). With $\Delta t_\mathrm{atmos}$ the atmospheric step (s), $k_\mathrm{split}$ the remap split count and $n_\mathrm{split}$ the acoustic substeps per remap split:

$$\Delta t_\mathrm{dyn} = \frac{\Delta t_\mathrm{atmos}}{k_\mathrm{split}}, \qquad \Delta t_\mathrm{acoustic} = \frac{\Delta t_\mathrm{atmos}}{k_\mathrm{split}\,n_\mathrm{split}}$$

The step is CFL constrained and should shrink as resolution or refinement increases.

## 13. Process decomposition

- **Uniform grid:** the largest multiple of 6 not above `n_cpus`, split equally over six tiles. It is not checked against the tile size, so choose `n_cpus` so the per-tile layout divides the tile dimensions.
- **Regional grid:** all `n_cpus`.
- **Nested runs:** processes are split between the global grid and each nest by minimizing the largest estimated per-domain time

$$T_g \sim \frac{w_g}{P_g}, \qquad w_g = N_g\,k_{\mathrm{split},g}\,n_{\mathrm{split},g}$$

where $P_g$ is the process count of domain $g$, $N_g$ its number of horizontal cells and $k_{\mathrm{split},g}$, $n_{\mathrm{split},g}$ its split counts. Global counts are multiples of 6; nest counts keep subdomain aspect ratios within 2 to 1. Allocations using all `n_cpus` are preferred, then the smallest bottleneck time, then the smallest spread.

Each domain's layout (per tile for the global grid) is the factor pair of its process count giving the most nearly square subdomains. Divisibility and a minimum subdomain size are not checked. The I/O layout is 1 by 1 and the physics block size 32. To override the allocation, set `grid_pes` under `fv_nest_nml` in a case-local `input.nml`, `input.yaml` or `input.yml` (first found), one value per domain; the global count must be a multiple of 6 and the sum must not exceed `n_cpus`.

## 14. Diagnostics and regridded output

A case-local `diag_table` replaces `templates/diag_table` (`templates/diag_table.monthly` for month and year segments). Its title and base date (literal or the `DESCRIPTION` and `DATETIME` placeholders) are replaced by the case description and initialization time. A case-local `field_table.yaml` (or legacy `field_table`) replaces `templates/field_table.yaml`; a case-local `data_table` (`data_table.yaml` with `use_modern_diag: true`) is staged when present. The default has three streams: `grid_spec` and `atmos_static` written once, and `fv3_hist` hourly (monthly means in the monthly table). Variables are listed in [tables/diag_field.csv](tables/diag_field.csv).

Each variable line (or `varlist` entry) names a native field (module and field, e.g. `"gfs_sfc", "t2m"`) and a freely chosen output name. At staging, every pair is checked against `tables/diag_field.csv`; an unknown pair, or an output name used twice in one stream, stops the case before the run (with close matches suggested), since FMS would otherwise skip the field silently.

The derived variables of `plugins/derived/derived_vars.py` (`ivt`, `ivtu`, `ivtv`, `vimfc`, `brn`, `mse`, `evf`, `vpd2m`) find their inputs by native field and are written under fixed names. `fv3_diag_files.variables` maps native field names to output names (`variables["t_plev"]` is `"t"` by default); a calculation is skipped with a warning if an input is not in the stream. A field written under several names takes the first table entry. Set `derived_vars_module` to a Python file that defines `derive = [func, ...]` to add your own: each `func(ds, *, state, variables, **_)` returns `ds`, runs after the built-ins once per output stream (skip a stream that lacks its inputs), and the file is copied to `derived_vars_user.py` in the case directory. The file is copied to `derived_vars_user.py` in the case directory at the cold start. See [plugins/README.md](plugins/README.md) and [config/examples/derived_vars_example.py](config/examples/derived_vars_example.py). `src/fv3_diag_files.py` provides `AsciiDiagTable` and `YamlDiagTable`.

After the model runs, native history is remapped to latitude-longitude: the global grid at the resolution of `c_res`, each nest over its box (a box across 180 degrees gives longitudes above 180) and a regional domain over its bounding box, with missing values outside. Files are named `global`, `regional`, `nest02`, `nest03`, ... Conservative remapping averages only valid source values. Fields without horizontal dimensions (`average_T1`, `average_T2`, `average_DT`, `time_bnds`) are copied unchanged.

`merge_freq` combines per-segment regridded files by concatenation along time (no averaging): `-1` merges the whole run into one file per stream and grid on the final segment, `0` keeps one file per segment, and `n` merges every `n` segments, flushing the remainder at the end.

### 14.1 Grid visualization

The initial driver plots the grid with `src/fv3_plot_grid.py` (needs `cartopy` and its cache in `fix_src/external/CARTOPY`) and writes to `state.run_dir`:

- `grid_faces.png`: one orthographic panel per cubed-sphere face, with nests drawn on their host face and the host mesh masked beneath each nest. Line density follows physical grid spacing.
- `nest_grids.png`: nest bounding boxes from `state.lon_min`, `lon_max`, `lat_min`, `lat_max` (degrees east in [-180, 180]; nests must not cross the antimeridian), in a projection chosen from the bounds (Robinson, PlateCarree, polar stereographic or Lambert conformal conic).

Units come from the supergrid `x` and `y` attributes (degrees if absent). No window is opened, and a plotting failure is logged without stopping the run.

## 15. Restarts

A run has `resubmit + 1` segments of `run_length` `run_length_units`. The first is a cold start from the initial driver; later segments are warm starts that load the saved `state.yaml`, not `run_config.yaml`. The cold start records a checksum of the grid geometry, vertical levels, initialization time and, when set, the hashed options of the `plugins/` plugins (`pamip_experiment`, `pamip_baseline`, `pamip_fields`, `pamip_forcing_year`). Each restart compares it with the current `run_config.yaml` and stops if those settings changed; other keys (e.g. `run_length`, `resubmit`) are not checked.

Regional segments must start on multiples of 3 h, the boundary file interval. Forcing files in `FIXED/PAMIP` and the cold-start namelists are reused; the case `diag_table` is restaged every segment. When `RESTART` is promoted to `INPUT`, the static inputs archived as `IC/INPUT` that the model also reads at a warm start (`oro_data*`, grid and mosaic links, `gfs_ctrl.nc`, fix files) are linked back; initial conditions and boundary files are not.

### 15.1 Continuing a failed case

Segments follow one another automatically; `case_continue.sh` is for a case that stopped partway. Fix the cause (a missing module, a bad node, a walltime), then run `case_continue.sh` from the case directory, the one holding `run_config.yaml`. It reads the model time of the newest restart on disk (`RESTART/coupler.res` if `RESTART` holds files, else `INPUT/coupler.res`), takes it as the start of the segment to run, and submits from there with the usual chain for the later segments. Segments already run are not repeated. The segment is taken from the files, not from the counters in `state.yaml`, so a failure before or after the restart was promoted to `INPUT` is handled alike.

The continued segment clears nothing (`RESTART`, `HIST`, `OUTPUT`, `IC`) and accepts a restart that was already promoted. A normal cold start, by contrast, refuses to run while `OUTPUT/seg` holds regridded output; use `case_continue.sh`, or remove the case directory to start over. If the model finished but the regridding failed, check `OUTPUT/seg` before continuing.

For an ensemble, `case_continue.sh --ensembles` resumes every member except those in `skip_ensembles`, and `case_continue.sh --mem 2 5` resumes only members 2 and 5. Each member is resumed from its own restart in `<case>/memNN`, so members that stopped at different segments each continue from their own. A member with no case directory (not started or already archived) or whose last restart is the final segment is reported and skipped; the others are still submitted. With no arguments an ensemble case is refused, and `--ensembles` or `--mem` on a single case is refused. The exit status is 1 if any member fails its checks or nothing was submitted.

`state.yaml` records `node_list` (the nodes of the latest attempt of each segment) and `regrid_progress` (`last` regridded segment and `total` segments).

### 15.2 Starting from an archived restart

With `archive_data: true` the archive holds `restart.tar.gz`, which holds `RESTART/` (the final restart files), `state.yaml` and `run_config.yaml` side by side. Unpack it into a directory `DIR` and run `case_continue.sh --restart-dir DIR` from a new case directory whose `run_config.yaml` keeps the grid and start-time settings of the archived case (checked against `DIR/state.yaml`). The preprocess rebuilds the grid, orography and initial conditions, then the run continues from the model time in `RESTART/coupler.res`. All other settings come from `state.yaml`, so only `resubmit` and the resource settings can change; raise `resubmit` to extend a finished case. The directory must be under a `container_bindpath` directory. With `--ensembles` or `--mem`, member N reads `DIR/memNN`.

## 16. Ensembles

Set `ensemble_run: true` and `n_ensembles`. Each member is an independent job with its own `memNN` directory and log; `skip_ensembles` omits members. Member 1 is unperturbed; the others get temperature noise seeded from the case checksum (`src/fv3_ensemble_driver.py`), which includes the PAMIP keys when set, so two PAMIP experiments get different perturbations (only member 1 is the same). Perturbations are made only when ICs are generated. To pair members across experiments, stage the same bundle through `external_ic_dir`. Members can be combined with soil moisture perturbations.

## 17. Archiving

With `archive_data: true`, the final segment copies regridded output to `archive_root`, writes `state.yaml` and the model log, compresses the final restart files, `state.yaml` and `run_config.yaml` to `restart.tar.gz` (Section 15.2), copies the driver log to `shield_log`, removes the working tree except `STATIC` and points the `run` (or `memNN`) symlink at the archive.

## 18. Example cases

Copy a file from [config/examples/](config/examples) into the case directory as `run_config.yaml`; unset keys come from the defaults.

| File | Case |
| --- | --- |
| `run_config.uniform.yaml` | Global uniform C96, hourly output |
| `run_config.stretch.yaml` | Stretched global grid |
| `run_config.nest.yaml` | Two telescoping nests (`chgres_cube_nest02.yaml` forces GFS ICs) |
| `run_config.regional.yaml` | Regional ESG domain |
| `run_config.external_ic.yaml` | Staged grid and IC bundle |
| `run_config.sm_perturbations.yaml` | Soil moisture perturbation (Section 11) |
| `run_config.pamip.yaml` | PAMIP experiment (Section 21) |
| `run_config.derived.yaml` | User derived variable (Section 14) |
| `run_config.monthly.yaml` | Ten years of monthly segments (Section 22) |

## 19. Compiling a SHiELD executable

`tools/build_shield.py` builds a native executable. It reads `modules` and `shield_exe` from the repository default `config/<release>/run_config.yaml` (not a case-local one); `--release` picks the release regardless of the checked-out branch.

```bash
python tools/build_shield.py                                  # 202411, repro, 64bit, pic, cleanall
python tools/build_shield.py --release 202604 --comp prod
python tools/build_shield.py --release 202604 --no-compile    # source and toolchain checks only
python tools/build_shield.py --release 202604 --force         # replace an existing build tree
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--release` | `202411` | `202411` or `202604`. |
| `--comp` | `repro` | `repro`, `prod` or `debug`. |
| `--bit` | `64bit` | `64bit` or `32bit`. |
| `--pic` / `--no-pic` | enabled | Position-independent code. |
| `--clean` | `cleanall` | `cleanall`, `clean` or `noclean`. |
| `--avx-level` | `-march=native` | Sets `AVX_LEVEL`; use `--avx-level=<value>` for values starting with `-`. |
| `--fflags` | empty | Extra Fortran flags. |
| `--no-compile` | off | Stop after fetching sources and checking the toolchain. |

The script works in `/tmp/$USER/FV3-<release>-public` (stops if it exists, unless `--force`), clones `SHiELD_build`, checks out the release, runs `CHECKOUT_code`, writes `site/environment.gnu.sh` from the host modules and checks the GNU/MPI/netCDF/libyaml toolchain (gfortran 10 or later). The duplicate `static` on `gettid` in the FMS source is removed for newer glibc. It then runs `COMPILE shield nh <comp> <bit> gnu [pic] <clean>`, checks for a new non-empty executable and installs it with the build mode and precision in the name. A `.manifest` beside it records component commits, SHA-256, toolchain and build directory. The script does not edit `run_config.yaml`; set `shield_exe` yourself:

```yaml
shield_exe: /path/to/FV3-<release>-public_SHiELD_nh.repro.64bit.gnu.x
```

A missing file falls back to the container only for a single-node `202411` case (with a warning); a multi-node run or a `202604` case stops at `case_submit.sh`. Test any new build on your target system.

### 19.1 Build profiles

| Component | `202411` | `202604` |
| --- | --- | --- |
| Core, physics, drivers | `FV3-202411-public` | `FV3-202604-public` |
| FMS / FMSCoupler | `2024.03` / `2024.03.01` | `2026.01` / `2026.01` |
| `ice_param`, `*_null` | as cloned (untagged) | as cloned (untagged) |
| Default executable | `FV3-202411-public_SHiELD_nh.repro.64bit.gnu.x` | `FV3-202604-public_SHiELD_nh.repro.64bit.gnu.x` |

`202411` builds `FMSCoupler/SHiELD/coupler_main.F90` (reads `dt_ocean`); `202604` builds `FMSCoupler/full` (reads `dt_cpld`, rejects `dt_ocean`). For a custom build check `Build/exec/*/pathnames_driver`, since the executable name does not show the coupler interface. Case-local `input.nml` or `input.yaml` overrides pass through unchanged.

`templates/202411/` sets `fms_io_nml` and `gfs_physics_nml sfc_coupled`, and restarts force `checksum_required` and `restart_checksums_required` to false. `templates/202604/` has neither and sets the full-coupler flags: `coupler_nml do_land = false` (no separate land component; Noah still runs in SHiELD physics, `lsm = 1`), `do_ocean = false` (no external ocean; `gfs_physics_nml do_ocean = true` keeps the slab ocean), `do_flux = false` (surface fluxes come from the atmospheric physics), and `ice_npes`, `land_npes` as in the upstream SHiELD regression cases. These describe the standard `shield` build, not a coupled `shiemom_lm4` experiment. The `202604` templates use only variables declared in `FV3-202604-public`, apart from `interpolator_nml interp_method` and the C3072 template's `cloud_diagnosis_nml`, which the SHiELD_build tests also set. Validate any newer release separately.

### 19.2 UFS_UTILS and containers

Preprocessing runs the programs in `/UFS_UTILS/exec` inside `preprocess.sif`; building UFS_UTILS on the host does not change them (rebuild the image, or bind your `exec` directory to `/UFS_UTILS/exec`). The image comes from the `gaea` branch of `kaiyuan-cheng/UFS_UTILS` (2024-10-18, Containerized_SHiELD_Workflow), and the workflow follows its interfaces: `orog` reads the binary terrain inputs (`gmted2010.30sec.int`, `landcover30.fixed`, `thirty.second.antarctic.new.bin`) with a nine-number `INPS` line, and `lakefrac` takes four arguments with GLDB v2 data. UFS_UTILS releases from 2024 on read NetCDF terrain with a three-line `INPS` and a five- or seven-argument `lakefrac`, so an upstream `exec` directory is not a drop-in replacement.

## 20. Quick start

1. Create a case directory and put `run_config.yaml` in it.
2. Add any overrides (`diag_table`, `chgres_cube.yaml`, `input.nml`).
3. Run `case_submit.sh` from that directory, directly or through a wrapper:

```bash
#!/bin/bash -l
"/path/to/pyUFS/case_submit.sh"
```

Submit from a clean shell (the script deactivates conda for its own process). A submission directory inside the run directory (`case_root/<parent>/<case_name>`) is refused. `tests/submit.sh` submits the test cases, each from its own directory.

## 21. PAMIP SST and sea-ice boundary conditions

The `pamip` plugin (`plugins/pamip`, which includes the external files) runs these steps (hooks called by `fv3_plugins`; `validate` first, the rest in `apply` at the cold start only):

1. `validate` checks `pamip_experiment`, `pamip_baseline`, `pamip_fields` and the `external_*` files before anything is written.
2. `plugins/pamip/external_files.py` copies the `external_*` files to `FIXED/external` and collects the `namsfc` entries pointing at the copies.
3. `write_boundary_files` (`plugins/pamip/boundary.py`) regrids the PAMIP fields to the baseline grids and writes SST and sea-ice climatologies to `FIXED/PAMIP`.
4. `set_cold_start_surface` sets SST and sea ice in every tile's surface file to the fields of the start date.
5. `patch_nml` writes the file entries and the settings below into the global and nest namelists, then `update_fixed_files` runs again so that the CO2 table follows `ictm`.

### 21.1 Experiments

`pamip_experiment` is the PAMIP experiment number (Smith et al., 2019, GMD 12, 1139) as quoted text, e.g. `"1.6"` (present-day SST, future Arctic sea ice), or `null` for the SHiELD climatologies. Experiments with SST and sea-ice files are 1.1 (control), 1.6, 1.5, 1.3 and 1.4 (`PAMIP_EXPERIMENTS` in `plugins/pamip/boundary.py`; input4MIPs identifiers in `plugins/pamip/registry.yaml`). Monthly climatologies come from `fix_src/external/PAMIP` (`python tools/get_pamip.py --root fix_src`, optionally `--experiment 1.6`, `--list`) and are written to `FIXED/PAMIP` as `RTGSST.PAMIP_<exp>.monthly.clim.grb` and `CFSR.SEAICE.PAMIP_<exp>.monthly.clim.grb`, keeping the headers, packing, month stamps and land flags of the baseline.

Treatment of land, ice and lakes:

- **Sea ice** changes only at ocean points of the baseline land mask (`seaice_newland.grb` or `external_mask`, thresholded at 0.5 as in `sfcsub.F`) and keeps its land flag (values above 1).
- **SST over land.** PAMIP files hold filled values over land (ocean values spread inland, 271 to 306 K). The `pamip` baseline keeps them; `shield` and `external` change ocean points only.
- **Lakes and inland seas** (water bodies of the 0.5 degree mask under 500 cells, e.g. Great Lakes, Baikal, Caspian) keep the baseline SST and ice in all baselines.
- **Surface cycle.** At ocean points it weights source fields by the land mask; at model land points it reads the SST file limited to 173 to 353 K.

The `pamip` plugin writes the file paths (`fntsfc`, `fnaisc`) and the settings below into the namelists with `patch_nml`, after the core has written them: `faiss = fsics = fsicl = 0` (sea ice follows the climatology), `use_ext_sst = false`, `nstf_name = [0, 0, 1, 0, 5]` and a global slab ocean: `do_ocean = true`, `disable_radiation_quasi_sea_ice = true` and `ocean_nml` with `ocean_option = SOM`, `mld_option = obs`, `restore_method = 1`, `sst_restore_tscale = 90` days (`SOM_RESTORE_DAYS` in `plugins/pamip/boundary.py`) and `start_lat`, `end_lat`, `maxlat` of -90, 90, 90. The slab restores the SST toward the PAMIP (or perturbed external) SST climatology and needs `fix/am/mld_DR003_c1m_reg2.0.grb`. To get the fixed-SST protocol back, set `gfs_physics_nml do_ocean: false` and `namsfc ftsfs: 0` in `input.yaml` (and in the nest file); `patch_nml` applies the case-local files again after the plugin, so they win. The surface cycle runs every `fhcyc` = 24 h, interpolating linearly between mid-month values; the AMIP II monthly-mean adjustment of the protocol is not applied.

At the cold start the fields interpolated to the start date are regridded onto every tile (`to_fv3_grid` in `src/latlon_to_fv3.py`, xESMF) and written to `slmsk`, `tsea`, `fice`, `hice` and `tisfc` over non-land points. The `newice` resets of `sfcsub.F` are applied directly: snow (`sheleg`, `snwdph`) 0, `zorl` 1 cm, sea-ice `stc` 271.2 K, new ice thickness 0.2 m (existing ice keeps its thickness). The PAMIP grid is regridded to the GRIB grids with `ll_regrid` (bilinear xESMF, NaN-skipping weights, periodic longitude, nearest valid value at ocean points with none). For the 0.083 degree SST grid the weights take about 3 minutes and 15 GB, and the land mask needs a second regridder of the same size.

At `restart_no` 0, `pamip_<tag>_sst_change.png` and `pamip_<tag>_sic_change.png` are written to `state.run_dir` (twelve Robinson panels of written file minus baseline; K for SST, fraction for ice; symmetric colour scale at the 99.9th percentile of the absolute change). The first process draws them, the others skip if the figure or its lock exists. They need `cartopy`; failure is logged and does not stop the run. Restart segments keep these files, and `preprocess_only` exits before they are written. See `config/examples/run_config.pamip.yaml`.

### 21.2 Baselines (`pamip_baseline`)

`null` or `pamip` (default), `shield`, or `external`.

- **`pamip`**: the experiment is applied to the PAMIP fields as they are.
- **`shield`**: the perturbation (experiment minus 1.1) is regridded from 1 degree and added to the SHiELD climatologies (`RTGSST.1982.2012`, 0.083 degree; `CFSR.SEAICE.1982.2012`, 0.5 degree). Sea ice is clipped to 0 to 1 with land flags kept. SST is perturbed where the SHiELD value lies in 271.2 to 313.0 K and kept between the SHiELD freezing value and 313.0 K. Files are named `...PAMIP_shield_<exp>...`. The control is experiment 1.1; `pamip_experiment: null` keeps the SHiELD settings and is not a control for it.
- **`external`**: the same perturbation added to the `external_*` copies (Section 21.3). Files are named `...PAMIP_external_<exp>...`. SST and ice change at ocean points of `external_mask` only (regridded and thresholded at 0.5 if on a different grid). All three `external_*` files must be set and exist, otherwise the case stops before grid generation.

`pamip_fields` (`null` or `both`, `sst`, `sic`) takes only the SST or only the sea ice from the experiment; the other field is the control (PAMIP 1.1 with the `pamip` baseline, the unperturbed field otherwise), and `_sst` or `_sic` is appended to the file names. This leaves out the ice-edge SST adjustment of the protocol pairs. `pamip_experiment`, `pamip_baseline` and `pamip_fields` are part of the restart checksum.

### 21.3 External climatologies

`external_sst`, `external_sic` and `external_mask` (default `null`) replace namsfc `FNTSFC`, `FNAISC` and `FNMSKH`. They are paths to GRIB files (variables and `~` expanded; absolute recommended, since a relative path is resolved from `CASE_DIR`), in the format of the files they replace (`am/RTGSST.1982.2012`, `am/CFSR.SEAICE.1982.2012`, `am/seaice_newland.grb`). They work without a PAMIP experiment. Each is copied (never linked or modified) to `FIXED/external`. Give all three together, since the land mask decides which source cells the surface cycle uses and the ice land flags must agree with it. The paths are not part of the restart checksum.

With baseline `null`, `pamip` or `shield`, the PAMIP files replace `external_sst` and `external_sic` without a warning. A set `external_mask` still replaces `FNMSKH`, so leave it `null` unless the baseline is `external`.

`python tools/esa_sst.py` builds a set from the ESA SST CCI v3.0 climatology in `fix/external/ESA` (details in its docstring), e.g. `external_sst: /path/to/fix/external/ESA/sst.ESACCI_1991_2020.monthly.clim.grb`. ESA has no values over land, so land and lake cells take the 1991 to 2020 monthly ERA5-Land skin temperature (0.1 degree), except land cells within three cells of water, which take the nearest ocean value. Lakes of the SHiELD mask stay water with the ERA5-Land lake skin temperature and the SHiELD ice (land flag 1.57, so a lake at freezing is ice covered). The sea-ice file keeps flag 1.57 over land. Regridding the 0.05 degree fields to 1/12 degree needs a large-memory node (about 15 GB). `tools/get_era5.py` downloads the ERA5 data once (Copernicus CDS account and `~/.cdsapirc` needed) to `fix/external/ERA5/monthly_skin_1991-2020.nc` and `monthly_sst_1991-2020.nc`.

### 21.4 Running the experiments

Run 1.1 and 1.6 first (1.6 minus 1.1 isolates the Arctic sea-ice forcing), then 1.5 and 1.3; 1.4 is Tier 2. The protocol is a 14-month run from 1 April 2000, discarding two months, over at least 100 members. Initial conditions come from GFS, which is unavailable for 2000, so start on 1 April of an available year; the climatologies do not depend on the year. CO2 is the year 2020 value (`ictm` 20200, reading `co2historicaldata_2020.txt` for every year); volcanic aerosol and solar constant are unchanged. Per-experiment settings are in `PAMIP_EXPERIMENTS` and `plugins/pamip/registry.yaml`.

Three test cases run twelve one-month segments from 1 January 2023 (`dt_atmos` 1200 s, `k_split` 3, `n_split` 6), all listed in `tests/submit.sh`:

- `tests/cases/pamip_1.1_pdsst_pdsic` (control) and `tests/cases/pamip_1.6_pdsst_futarcsic` use `pamip_baseline: external` with the ESA set from `tools/esa_sst.py`. They differ only in `pamip_experiment` and need the PAMIP files (`tools/get_pamip.py`) and the ESA files.
- `tests/cases/pamip_ctrl` has no PAMIP option: the reference for the SHiELD method.

## 22. Monthly runs

Use one-month segments starting at 00 UTC on day 1 (`config/examples/run_config.monthly.yaml`): `resubmit: 119` gives ten years and `merge_freq: 12` one file per year.

- **Diag table.** Without a case `diag_table`, `templates/diag_table.monthly` averages samples over each month. Precipitation (`pr`, `prc`, `prsn`) is a monthly mean rate (kg m-2 s-1); runoff buckets are summed by FMS into `mrro_sum` and `mrros_sum` (kg m-2). Time is written in seconds, and regridded files keep the `post_process` time handling.
- **Namelist.** Same as other segments (`common_configs`, `set_radiation` in `src/fv3_namelists.py`) except `fhswr` and `fhlwr` are at least 3600 s. `fdiag` and `fhzero` both take `diagnostic_interval_hours` (default 1 h), `print_freq` takes `fv3_print_freq` (default 0), `fprint` is off and `ldiag3d` keeps the `templates/<release>/input_nml.yaml` value (true).
- **Diagnostic interval.** The model computes diagnostics every `fdiag` hours, so each sample of a bucket field is a full `fhzero` mean and cost scales with `1/fdiag`. Use `diagnostic_interval_hours: 6` for monthly means: the windows tile each day, so bucket means are exact, and four samples a day alias only the fourth and higher diurnal harmonics of instantaneous fields. Use 3 for a finer diurnal sample at twice the cost; intervals above 6 h alias the diurnal cycle. In any override keep `fdiag = fhzero`, both multiples of `dt_atmos`; a weekly `fhzero` would misalign buckets with month ends.
- **Time step.** For C96 use `dt_atmos` 600 s, `k_split` 2, `n_split` 6, keeping `dt_atmos` at or below `fv_sg_adj` (600 s) as in the SHiELD_build RTS.

The test `tests/cases/monthly` has no case `diag_table`, so it exercises the template, and keeps the physics of `templates/<release>/input_nml.yaml`, changing only the timing.

## 23. Plugins

`src/` is the core. Everything that is not needed by every run is a package in [plugins/](plugins); the core calls it only through `src/fv3_plugins.py`, and a plugin is active when one of its `ACTIVE_KEYS` is set in `run_config.yaml`. The hook contract and how to add a plugin are in [plugins/README.md](plugins/README.md).

| Plugin | Active keys | Hooks | What it does |
| --- | --- | --- | --- |
| `derived` | `derived_vars_module` | `validate`, `apply`; `add_derived_variables` for every output stream | Built-in derived variables (`derived_vars.py`) plus the `derive` list of the user file (`use_derived.py`); the user file is copied to the case directory. Section 14. |
| `sm_perturbations` | `sm_perturbations` | `apply` | Perturbs the surface restart files at the cold start and at each restart segment. Section 11. |
| `pamip` | `pamip_experiment`, `pamip_baseline`, `pamip_fields`, `external_sst`, `external_sic`, `external_mask` | `validate`, `apply` | PAMIP boundary files, slab ocean and CO2 settings, and the external climatologies. Section 21. |

The core calls the hooks here:

| When | Core call | Hook |
| --- | --- | --- |
| Cold start, before grid and IC generation | `fv3_plugins.validate()` in `fv3_init_driver.py` | `validate` |
| Cold start, last step before `save_fv3_state()` | `fv3_plugins.do_plugins(restart=False)` in `fv3_init_driver.py` | `apply` |
| Each restart segment, last step before `save_fv3_state()` | `fv3_plugins.do_plugins(restart=True)` in `fv3_restart_driver.py` | `apply` |
| Each regridded output stream | `fv3_plugins.add_derived_variables(ds)` in `fv3_regrid.py` | `plugins.derived.add_derived_variables` |
| Restart checksum | `compute_checksum` in `fv3_state.py` | `HASH_KEYS` of each plugin |

`do_plugins` runs after the namelists, tables, fixed files and run script have been written. A plugin changes the namelists with `patch_nml(settings)` (`settings` is `{group: {key: value}}`), which merges them into the global and nest namelists and then applies the case-local `input.yaml` and `input.nml` override files again, so those win. A `namsfc` file key (`fntsfc`, `fnaisc`, `fnmskh`) replaces the SHiELD default file name in the namelist. A plugin that changes a value that `update_fixed_files` reads (pamip sets `ictm` for the CO2 table) is passed `update_fixed_files` and calls it after patching. A plugin gets the live `state` and the core functions it needs as keyword arguments (listed in `src/fv3_plugins.py`), and imports nothing from `src/`.
