# pyUFS HPC Run Guide

`pyUFS` runs GFDL SHiELD on SLURM clusters such as Oscar without the standard UFS directory layout. It generates or stages grids and initial conditions, runs SHiELD, regrids history files, and manages restarts, ensembles, and archives. The case configuration is `run_config.yaml`; case-local namelists and preprocessing override files customize the generated inputs.

## 1. Quick start

**Native setup** is required for multi-node SHiELD runs. From the pyUFS repository root, in an interactive session (`interact`) because the builds are heavy:

```bash
./setup_all.sh --all                   # SHiELD 202411 (default)
./setup_all.sh --all --release 202604  # SHiELD 202604
```

This creates the `pyufs` conda environment, builds SHiELD and UFS_UTILS, and updates the `fix/` datasets. `--release` affects only the SHiELD build and `-j` only the UFS_UTILS build. The native executables are used unless `shield_use_container` or `preprocess_use_container` is set to `true` (see Section 2). Set `shield_exe` (and `ufs_bin_dir`, if used) in your case configuration to the installed paths reported by the scripts. The default `shield_exe` is the `202411` build, so after building `202604` set both `shield_release: "202604"` and `shield_exe`.

The defaults point to lab paths under `/oscar/data/deeps/...`. Outside the lab, override `fix_src`, `shield_exe`, `containers_root`, `shield_image`, `preprocess_image`, `case_root` and `archive_root` in your case configuration.

Create and submit a case:

```bash
mkdir -p ~/cases/test
cp config/examples/run_config.uniform.yaml ~/cases/test/run_config.yaml
cd ~/cases/test
# Edit run_config.yaml (paths, grid, dates, resources, run length).
/path/to/pyUFS/case_submit.sh
```

Submit **from the case directory**. `case_submit.sh` validates the YAML, applies defaults from `config/run_config.yaml`, and submits through `sbatch`. The driver log defaults to `shield_driver.log`.

Missing images are built automatically (requires internet access and several GB of scratch). `SHIELD_TAG` and `PREPROCESS_TAG` select non-default image tags; the image manifest records sources and checksums. To build them separately (existing images are skipped; `FORCE=1` rebuilds, and names select one image):

```bash
CONTAINERS_DIR=/path/to/containers TMP_DIR=/scratch bash tools/install_images.sh [preprocess] [shield]
```

A host `python3` with PyYAML is required for submission. The `run` symlink points to the working case while a job is active and to its archive after archiving.

## 2. SHiELD releases and execution modes

One checkout supports both releases; **`202411` is the default**.

| | `202411` | `202604` |
| --- | --- | --- |
| Model source | `FV3-202411-public` | `FV3-202604-public` |
| FMS / FMSCoupler | `2024.03` / `2024.03.01` | `2026.01` / `2026.01` |
| Coupler driver | `FMSCoupler/SHiELD` | `FMSCoupler/full` |
| Coupler namelist key | `dt_ocean` | `dt_cpld` |
| Namelist templates | `templates/202411.*.yaml` | `templates/202604.*.yaml` |

Select the newer release with `shield_release: "202604"`; the defaults in `config/run_config.yaml` are the same for both releases. Set `shield_exe` to an executable built for the selected release (the default path is the `202411` build). **Use `dt_cpld` in `run_config.yaml` for either release**: pyUFS writes `dt_ocean` for `202411`. Rename `dt_ocean` to `dt_cpld` in older case configurations. The release must match the executable. Case-local `input.nml`/`input.yaml` overrides are passed through unchanged, including any coupler keys; use the key expected by that release in these files. The `202604` full-coupler templates disable coupled land, ocean, and flux components while SHiELD surface physics remains active.

Execution mode is selected explicitly and independently for the model and preprocessing. Both switches default to `false` (native); a path key is required, and checked, only when its switch is `false`, and is ignored when the switch is `true`.

| Setting | Behavior |
| --- | --- |
| `shield_use_container: false` | Run the native SHiELD in `shield_exe`; required for multiple nodes. |
| `shield_use_container: true` | Run SHiELD from `shield_image`; single node only. |
| `preprocess_use_container: false` | Native UFS_UTILS in `ufs_bin_dir` using the `conda_env/pyufs` environment; MPI programs run as SLURM steps. |
| `preprocess_use_container: true` | Run UFS_UTILS and regridding in `preprocess.sif`. |

A null value in a case file takes the repository default (`false`), so state the switches explicitly. Values other than `true` or `false`, such as the string `"false"`, are rejected at submission.

Host `modules` (default: `hpcx-mpi`, `netcdf-mpi`, `libyaml`) are loaded for native execution. In container mode, every `container_bindpath` must exist on the compute nodes. `shield.sif` uses the `202411` build, so use a matching native executable for `202604`.

## 3. Configuration

Copy an example from [`config/examples/`](config/examples/) and override only the required settings. The full list of accepted keys and defaults is in [`config/run_config.yaml`](config/run_config.yaml). Unknown keys cause submission to fail. Relative paths resolve from the submission directory; `~` and environment variables are expanded in paths. A `run_config.yaml` or plugins file can pull in shared settings with `include: path` or a list of paths, relative to its own directory; its own keys take precedence, and two included files that set the same key to different values are an error. The cold start records the resolved absolute path of every included file in `state.yaml`, under `run_config_includes` and `plugins_config_includes`; an include that does not exist is skipped with a warning and not recorded.

### Paths and SLURM resources

| Key | Description / default |
| --- | --- |
| `fix_src` | Writable static-data tree (`fix/`); must exist. |
| `case_root` | Scratch root, default `/jobtmp/$USER/gfdl_shield`. Must not contain the submission directory. |
| `archive_root`, `archive_subdir` | Archive destination; `archive_subdir` required if `archive_data: true`. |
| `containers_root`, `shield_image`, `preprocess_image`, `container_bindpath` | Container storage, image paths, and bind mounts. |
| `n_nodes`, `n_cpus`, `n_cpus_per_task` | Default: 4 nodes, 192 total MPI tasks, 1 CPU per task. Tasks are rounded down to a multiple of nodes. |
| `walltime`, `partition` | Default: 24 hours, `batch`. |
| `job_mem_gb`, `constraint_node`, `exclusive_node` | Optional memory, node constraint, and exclusive allocation; `job_mem_gb` is total GB, translated to per-CPU memory on multi-node jobs. |
| `sbatch_options` | Extra raw `sbatch` arguments, e.g. `"--account=myproj"`. |
| `logfile` | Driver log basename, default `shield_driver`. |

Jobs are named `<parent>.<case>` (`.MEMnn` appended for ensembles). Working cases live at `<case_root>/<parent>/<case>`. After preprocessing and running, the case may contain `FIXED/`, `GRID/`, `IC/`, `INPUT/`, `LOGS/`, `OUTPUT/`, `RESTART/`, `HIST/`, and `TMP/`.

### Experiment and model settings

| Key | Description |
| --- | --- |
| `case_name`, `description`, `fv3_debug` | Identifier (defaults to directory name), experiment label, and verbose model diagnostics. |
| `init_datetime` | Initial-condition cycle in UTC, `YYYYMMDDHHZ`, hour 00, 06, 12, or 18. |
| `forecast_hour` | Lead time of the selected GFS/HRRR cycle; model starts at cycle plus lead. |
| `run_length`, `run_length_units` | Segment duration in `hours` (default), `days`, `months`, or `years`. |
| `resubmit` | Additional segments: total segments = `resubmit + 1`. |
| `generate_ic_data`, `model_input_dir` | Generate ICs or reuse a staged case (Section 5). |
| `ic_grib_files` | Use downloaded GFS or HRRR GRIB2 files instead of downloading them (Section 5.2). |
| `preprocess_only`, `preprocess_grid_only`, `preprocess_orog_only` | Stop after preparing the specified inputs. |
| `c_res`, `gtype`, `levels` | Horizontal resolution, grid type, hybrid vertical level count. |
| `dt_atmos`, `dt_cpld`, `k_split`, `n_split` | Time steps and dynamical subcycling (Section 7). |
| `do_deep` | Keep deep convection active on grids ≤4 km when `true`; otherwise disabled there. |
| `lake_cutoff`, `add_lake`, `make_gsl_orog` | Surface fractions, lake data, optional GSL orography. |
| `merge_freq`, `pack_output`, `use_modern_diag` | Regridding merge interval, int16 packing, modern diagnostic tables. |
| `archive_data` | Archive the final segment and its restart files. |
| `ensemble_run`, `n_ensembles`, `skip_ensembles` | Ensemble controls (Section 10). |
| `plugins_config` | Path to a separate plugin YAML (Section 11). |

Month/year lengths use calendar months (one year = 12 months). For calendar-month segments, the start day must be 28 or earlier. Regional runs also require **segment start times divisible by 3 hours** for lateral boundary updates. `continue_run` is set by the driver and should not be set manually.

## 4. Horizontal grids

Supported resolutions: `C48`, `C96`, `C192`, `C384`, `C768`, `C1152`, `C3072` (integers are also accepted). Approximate global spacing is 100 km at C96, 50 km at C192, 25 km at C384, 13 km at C768, and 3 km at C3072.

| `gtype` | Required or relevant settings |
| --- | --- |
| `uniform` | Standard six-face cubed sphere. |
| `stretch` | `target_lon`, `target_lat`, `stretch_factor` (>1 refines the target). |
| `nest` | `refine_ratio`, `parent_tile`, `halo`, and `lon_min`, `lon_max`, `lat_min`, `lat_max` for each nest. |
| `regional_gfdl` | Bounding box, `refine_ratio`, target center; equivalent global resolution is computed. |
| `regional_esg` | `target_lon/lat`, `idim`, `jdim`, `delx`, `dely` (supergrid degrees). |

For nests, give one bounding box per `refine_ratio` entry; mismatched list lengths fail preprocessing. All nests are generated in one `make_hgrid --nest_grids` call. Boxes contained within their predecessors form a telescoping chain (effective refinement is the product of ratios); otherwise they are same-level nests. Bounds are rounded outward to whole parent cells, and each nest must remain `halo` parent cells inside its parent tile. A list of `parent_tile` values selects the host for independent nests; the first value applies to the outer nest in a telescoping chain. The grid rotates so the first box is centred on tile 6 (the usual `parent_tile`). Boxes may cross 0° or 180° longitude. Regional grids require `halo: 3`; `regional_esg` model spacing is twice `delx` and `dely`.

### Editing generated grid or orography files

For `uniform`, `stretch`, and `nest` cases, staged files can be edited and reused:

1. Set `preprocess_grid_only: true`, submit, and edit the NetCDF tiles in `IC/grid/`. Keep grid/mosaic filenames and geometry consistent. Set the option back to `false` and resubmit.
2. To edit terrain, set `preprocess_orog_only: true`, submit, and edit `IC/orography/oro.C<c_res>.tile*.nc`. Modify `orog_raw` and `orog_filt` consistently; the latter is read by the dynamical core. Then disable the option and resubmit.
3. Keep `generate_ic_data: true`. Staged grids are accepted only if settings in `.grid_settings.yaml` still match. Back up originals before editing.

The grid-only and orography-only modes imply `preprocess_only`. Regional grids do not use this staging workflow. The topography filter is applied during generation on global tiles; edited staged files are reused directly.

## 5. Static datasets and initial conditions

### 5.1 Static datasets (`fix/`)

`fix_src` points to a writable static-data tree holding terrain, climatologies, lookup tables, and surface inputs. Missing files are searched by NOAA filename in `fix_src` and then in the NOAA public bucket, newest version first. Successful downloads are stored in `fix_src`. A run stops if a required file is unavailable. `UFS_PY_FIX_REMOTE=0` disables remote lookups.

To populate `fix/` in advance:

```bash
python tools/update_fix.py --dest /path/to/fix
# Use --skip-carto to omit Cartopy data.
```

The updater mirrors `am/`, `orog/`, and `sfc_climo/`, fixes CO2/aerosol links, and fetches Cartopy data. The repository `fix/` entry is a symlink; if it does not resolve on your system, repoint it with `ln -sfn /path/to/fix fix` from the repository root. Non-bucket data such as `mld_DR003_c1m_reg2.0.grb` and external ERA5/PAMIP datasets must be staged separately under `fix/external/` or their expected locations. If the selected model year has no CO2 table, the latest earlier year is used with a warning.

### 5.2 GRIB2 sources and `chgres_cube`

With `generate_ic_data: true` (default), pyUFS creates the grid and orography, retrieves GRIB2 fields, then runs `chgres_cube` for each domain. Sources are selected **per domain**:

| Domain | Atmosphere | Surface | `chgres_cube` conversions |
| --- | --- | --- | --- |
| Global | GFS | GFS | One |
| Regional | GFS | GFS | One |
| Nest within HRRR coverage | HRRR | GFS | Two: atmosphere and surface |
| Nest outside HRRR coverage | GFS (automatic fallback) | GFS | One |

GFS always provides surface fields because HRRR lacks the required soil levels. Downloads retry using NOAA AWS S3 or NCAR GDEX for GFS, and NOAA AWS S3 or Google Cloud Storage for HRRR.

To use GRIB2 files you have already downloaded, list them by model in `ic_grib_files`; paths and globs are allowed, and any file names:

```yaml
ic_grib_files:
  GFS: /data/gfs/gfs.t00z.pgrb2.0p25.f0*
  HRRR: [/data/hrrr/hrrr.t00z.wrfnatf00.grib2]
```

A listed model is never downloaded. pyUFS picks, for each conversion and regional boundary lead, the file whose first message has the required cycle and lead, and stops with the list of files found if there is none or more than one. Models that are not listed are downloaded as usual.

`tools/ecmwf_ic.py` writes ERA5 initial conditions into a GFS file of the same cycle (`python tools/ecmwf_ic.py --cycle 2000060100 --fh 0 --out DIR`, CDS key in `~/.cdsapirc`). List its output under `GFS`; `chgres_cube` reads it as GFS. The ERA5 model levels come from the MARS tape archive and can queue for hours, so run the tool before submitting the case. Regional lateral boundaries after hour 0 and HRRR nest atmospheres still come from their own files.

`init_datetime` selects the source forecast cycle; `forecast_hour: 0` uses its analysis. A nonzero `forecast_hour` starts the model at `init_datetime + forecast_hour`. Regional lateral boundaries use that **same cycle** at leads `forecast_hour`, `+3`, `+6`, and so forth, so `forecast_hour` must be divisible by 3 and data must exist through the final boundary lead (GFS: hourly through f120, three-hourly through f384).

### 5.3 `chgres_cube` overrides

Add optional domain-specific files **in the submission directory**:

| Domain | YAML override | Also recognized |
| --- | --- | --- |
| Global or regional | `chgres_cube.yaml` | `fort.41` |
| First nest | `chgres_cube_nest02.yaml` | `fort_nest02.41` |
| Second nest | `chgres_cube_nest03.yaml` | `fort_nest03.41` |

Defaults for every key live in [`config/chgres_cube.yaml`](config/chgres_cube.yaml); it also defines the complete set of accepted keys. Write only the keys you want to change, since the workflow merges them over the defaults (precedence: defaults, then workflow-derived values, then your file). An unset or `null` value leaves the workflow-generated value unchanged. YAML keys can be flat or nested inside a single `config:` mapping; unknown keys fail preprocessing and identify the offending file. [`config/examples/chgres_cube_nest02.yaml`](config/examples/chgres_cube_nest02.yaml) is a minimal per-nest example. For example:

```yaml
external_model: HRRR
nsoill_out: 9
vgfrc_from_climo: false
```

`external_model` controls **atmospheric** input selection:

| Value | Behavior |
| --- | --- |
| Omitted on a nest | Try HRRR, fall back to GFS outside HRRR coverage. |
| `GFS` | Force GFS for that domain. |
| `HRRR` | Require HRRR; error outside its coverage. Also allowed for regional domains, but invalid for a global domain. |

Surface fields remain GFS. The workflow controls `convert_atm`, `convert_sfc`, and `convert_nst` for nested and regional grids, and ignores user-supplied values for these switches; other settings remain overridable. The default land-surface options follow UFS_UTILS except `tg3_from_soil: true`, which initializes deep-soil temperature from the lowest source soil layer.

The selected atmosphere, surface, and NST source for every domain is recorded in `state.yaml` under `ic_source` (`global`, `regional`, `nest02`, ...). When a domain uses multiple sources, logs are separated by conversion, for example `chgres_cube_nest02_atm.log` and `chgres_cube_nest02_sfc.log`. Inspect these before changing source overrides.

### 5.4 Preprocessing only or external ICs

Use `preprocess_only: true` to prepare the grid and ICs without running SHiELD. For grid-only and terrain-only staging, see Section 4.

To reuse an already staged case:

```yaml
generate_ic_data: false
model_input_dir: /path/to/prestaged_case
```

The bundle must contain nonempty `FIXED/`, `GRID/`, and `INPUT/` directories plus `state.yaml`. Only grid/IC state is taken from the bundle; run settings and plugins come from the new case. Its grid type, vertical levels, and model start (`init_datetime + forecast_hour`) must match the new case.

### 5.5 Optional UFS_UTILS preprocessing

These programs are **off by default** and run after `chgres_cube` on a cold start when `generate_ic_data: true`:

| Key | Purpose and requirements |
| --- | --- |
| `run_global_cycle` | Update surface fields using `namsfc` climatologies and optional `global_cycle_sst_file`, `global_cycle_ice_file`, `global_cycle_snow_file` (GRIB1). Missing analyses retain the first guess. Global, stretched, and nested grids only. |
| `run_emcsfc_snow` | Build snow analysis from `ims_snow_file` and optional AFWA snow inputs, then pass it to `global_cycle`. |
| `run_emcsfc_ice_blend` | Blend `ims_ice_file` with `five_min_ice_file` for `global_cycle`. Requires NCEP `cnvgrib`, `copygb2`, and `copygb` on `PATH`. |
| `run_cpld_gridgen` | Generate MOM6/CICE6 grids and ocean mask in `IC/cpld_gridgen` for `cpld_gridgen_res` (`500`, `100`, `050`, `025`). Requires `fix_src/mom6/<res>` and `ESMF_Scrip2Unstruct`; `cpld_gridgen_postwgts: true` also needs `ncremap`. |

Source analyses are not included in the NOAA GFS buckets and must be provided separately. `run_cpld_gridgen` output is **not read by SHiELD**. Generate resolution `025` before other resolutions requiring `Ct.mx025_SCRIP.nc`. The `global_cycle` logs are in `LOGS/preprocess/global_cycle_<domain>.log`; emcsfc files remain under `IC/surface_analysis/`. See the release defaults for the complete file-key list.

## 6. Case workflow

A new case runs one cold-start segment, then `resubmit` warm-start segments. `case_run.sh` handles grid/IC preprocessing, namelists and plugins, the model, output regridding, and resubmission or final archiving. A case-local `input.yaml`, `input.yml`, or `input.nml` overrides generated namelists; case-local `diag_table`, `field_table.yaml`, and `data_table` override the defaults.

On failure, `TMP/` is retained for inspection. The next submitted job clears it. The model state and numbered outputs are preserved during continuation.

## 7. Time stepping and MPI decomposition

When unset, `dt_atmos`, `k_split`, and `n_split` follow these resolution defaults:

| `c_res` | `dt_atmos` (s) | `k_split` | `n_split` |
| --- | ---: | ---: | ---: |
| 48 | 1200 | 2 | 6 |
| 96 | 720 | 2 | 6 |
| 192 | 450 | 2 | 6 |
| 384 | 360 | 2 | 6 |
| 768 | 180 | 2 | 8 |
| 1152 | 120 | 2 | 8 |
| 3072 | 90 | 2 | 10 |

`dt_cpld` defaults to `dt_atmos`; it must be a multiple of `dt_atmos`, and the segment length must be a multiple of `dt_cpld`. The FV3 remapping and acoustic substeps are `dt_atmos / k_split` and `dt_atmos / (k_split * n_split)`, respectively. Reduce the time step when resolution or refinement makes the CFL constraint more restrictive.

In nested cases, the finest grid sets `dt_atmos`. Provide `k_split`/`n_split` lists of length `n_nests + 1` when overriding them. MPI tasks are placed by the size of the largest rank, since every step waits for it: FMS gives a rank `floor` or `ceil` of the tile size over the layout, so a layout need not divide the tile, but the cost is `ceil(nx/lx) * ceil(ny/ly)` cells. For a uniform cubed sphere, a regional grid or a nested case, `n_cpus` is the upper limit; a count that forces thin strips (for example a prime number of tasks per face) is replaced by a smaller one with a smaller largest rank, so `total_pes` can be below `n_cpus`. A rank keeps at least 8 cells in each direction; a grid too small for that gets the best layout available and the run continues. In nested cases the tasks are split to minimise the largest `w_g * (cells of the largest rank of grid g)`, with `w_g = k_split,g * n_split,g`, where the global grid takes a multiple of 6 and the grids run concurrently; the search is exact for any number of nests. The layout orientation follows the grid shape, and the physics `blocksize` is the size nearest 32 that divides every rank's domain, or -1 when none does. To override allocation, set `fv_nest_nml.grid_pes` in a case-local namelist, one count per domain, with a global count divisible by 6 and total no greater than `n_cpus`.

## 8. Diagnostics, regridding, and monthly runs

The default `diag_table` writes `grid_spec` and `atmos_static` once and `fv3_hist` hourly. To customize streams, place `diag_table` in the case directory. Its title and base date are replaced by the case description and initialization time. Each variable names a native module/field pair (for example `"gfs_sfc", "t2m"`) and an output name. Pairs are checked against [`tables/diag_field.csv`](tables/diag_field.csv); unknown pairs and duplicate output names **within a stream** stop the run. `use_modern_diag: true` requires a case-local `diag_table.yaml` (optionally `data_table.yaml`). A case-local `field_table.yaml` or `field_table` is also supported.

History is remapped to latitude-longitude in `OUTPUT/`: `global`, `regional`, `nest02`, `nest03`, etc. Native fields without horizontal dimensions are copied unchanged. During execution, outputs are numbered by segment. `merge_freq` controls concatenation **without temporal averaging**:

| `merge_freq` | Result |
| --- | --- |
| `-1` | Merge the full run at the last segment. |
| `0` | Keep one regridded file per segment. |
| Positive integer `n` | Merge groups of `n` segments. |

Merged filenames use `<stream>.<grid>_<nn>.nc` (no suffix for a single chunk). `pack_output: true` packs final output to int16 with scaling and offset. The initial driver also attempts `grid_faces.png` and `nest_grids.png` in the submission directory; plotting needs Cartopy and does not block a run; `nest_grids.png` cannot plot nests crossing the antimeridian.

For **monthly climatological experiments**, use one-month segments starting at 00 UTC on day 1:

```yaml
run_length: 1
run_length_units: months
resubmit: 119                 # 120 months = 10 years
merge_freq: 12                # one output per year
diagnostic_interval_hours: 6
```

Without a case-local `diag_table`, monthly/year segments use `templates/diag_table.monthly`: precipitation `pr`, `prc`, and `prsn` is a monthly mean rate (`kg m-2 s-1`), while runoff buckets `mrro_sum` and `mrros_sum` are monthly totals (`kg m-2`). `diagnostic_interval_hours` sets both `fdiag` and `fhzero` (keep them equal and divisible by `dt_atmos`). A 6-hour interval reduces diagnostic cost; use 3 hours for finer diurnal sampling. For C96 monthly runs, the documented example uses `dt_atmos: 600`, `k_split: 2`, `n_split: 6`.

Optional derived diagnostics are computed after regridding when their input fields are present; see [`plugins/README.md`](plugins/README.md).

## 9. Restarts, continuation, and archiving

Warm starts read `state.yaml`, not a freshly generated configuration. Each restart verifies that model-defining settings (grid, levels, initialization, and relevant plugin settings) still agree with `run_config.yaml`; changes are rejected. Other values such as `run_length` and `resubmit` are not part of this restart check. The driver stores the plugin settings at cold start, so editing a plugin file later does not alter an existing case.

To resume a failed case, fix the failure and run from the original submission directory:

```bash
/path/to/pyUFS/case_continue.sh
```

The command finds the newest model restart, skips completed segments, and preserves `RESTART/`, `HIST/`, `OUTPUT/`, and `IC/`. **Do not rerun `case_submit.sh` over existing numbered outputs**. If SHiELD finished but regridding failed, inspect the numbered `OUTPUT/` files before continuing. For ensembles, specify `--ensembles` or `--member`; an unqualified continuation is rejected. Missing or completed members are skipped, and the command fails if no member is submitted.

Ensemble continuation:

```bash
/path/to/pyUFS/case_continue.sh --ensembles
/path/to/pyUFS/case_continue.sh --member 2 5
```

With `archive_data: true`, the final segment (`core/fv3_finalize_run.py`, run by `launchers/finalize` after the regrid) archives regridded output, `state.yaml`, logs, and `restart.tar.gz` under `<archive_root>/<archive_subdir>/<parent>/<case>`. The archive retains `STATIC/` in the working tree and removes other working data after copying. The restart archive contains `RESTART/`, `state.yaml`, and `run_config.yaml`. To continue from an archived restart, unpack it to `DIR`, configure a new case with matching grid and start-time settings, and run:

```bash
/path/to/pyUFS/case_continue.sh --restart-dir DIR
```

The grid and ICs are rebuilt and model time comes from `RESTART/coupler.res`. For archive continuation, existing state controls the physical setup; only `resubmit` and resource settings can be changed. Increase `resubmit` to extend a completed run. `DIR` must be accessible through `container_bindpath` when using containers.

## 10. Ensembles

```yaml
ensemble_run: true
n_ensembles: 10
skip_ensembles: []
```

Each member is an independent SLURM job with its own `memNN/` case directory and log. Member 1 is unperturbed; subsequent members receive deterministic temperature noise seeded by the case checksum, which incorporates relevant plugin settings. To use identical starting perturbations across cases, stage the same IC bundle through `model_input_dir`. See Section 9 for member-specific continuation.

## 11. Plugins

Plugins extend the core workflow without changing `core/`. Set `plugins_config` in the case `run_config.yaml` to a separate YAML file. The path can be absolute or relative to the submission directory; `~` and environment variables are expanded:

```yaml
# run_config.yaml
plugins_config: plugins.yaml
```

Each top-level section in `plugins.yaml` enables one plugin and supplies its options; the plugins run in the order of their sections. `plugin_dirs` lists folders with your own plugins, searched before `plugins/`; relative paths are read from the folder of `plugins.yaml`:

```yaml
# plugins.yaml
plugin_dirs: [~/pyufs_plugins]   # optional
pamip: {experiment: "1.6"}
t2m_celsius: true                # a plugin without settings
```

`plugins_config: null` disables plugins. The plugins in `plugins/` are `pamip`, `sst_perturbation` and `sm_perturbations`; their keys are in [`plugins/README.md`](plugins/README.md). A section `name: true` enables a plugin with its defaults. Your own derived variables are a plugin with a `regrid` hook in a `plugin_dirs` folder; see [`config/examples/plugins/t2m_celsius`](config/examples/plugins/t2m_celsius/__init__.py). Unknown plugin names or keys stop preprocessing before grid generation.

`core/fv3_plugins.py` is the **only core module that loads plugins**. It finds each plugin by name, in the `plugin_dirs` listed in the plugins file and then in `plugins/`, and calls its hooks at the relevant workflow stages in the order the plugins file lists them. The cold start stores plugin settings in `state.yaml`; later segments reuse the stored settings, so editing `plugins.yaml` does not change a running case. For plugins that modify namelists, case-local `input.yaml` or `input.nml` overrides take precedence where supported.

To add a plugin, create a folder with an `__init__.py` under `plugins/` or a `plugin_dirs` folder, following [`plugins/README.md`](plugins/README.md); nothing is registered in `core/`. Keep plugin-specific algorithms, dataset preparation, and experiment instructions in that package's documentation, not in this run guide.

## 12. Building native executables

`./setup_all.sh` is the standard installation path. To build individual releases:

```bash
python tools/build_shield.py                               # 202411, repro build
python tools/build_shield.py --release 202604 --comp prod
python tools/build_shield.py --release 202604 --no-compile  # fetch/check only
```

The builder reads the modules from the **repository defaults** (`config/run_config.yaml`), not case-local YAML. It requires gfortran ≥10 and builds in `/tmp/$USER/FV3-<release>-public`, which is deleted and rebuilt on every run. The executable is installed in `bin/` as `<release>.fv3.public_SHiELD_nh.<comp>.<bit>.gnu.x`, for example `202604.fv3.public_SHiELD_nh.repro.32bit.gnu.x`, so builds of different releases, compiler modes and precisions do not overwrite each other. A `<exe>.manifest.yaml` file next to it records source commits, SHA-256, and toolchain, and the script prints the `shield_exe` value to use. To rebuild, rerun the same command. If a build fails, read the log printed at the failure, fix the cause, and rerun; nothing from the failed tree is reused. Set `shield_exe` in your case YAML to the printed path and confirm that it was built for the same `shield_release`. Build options include `--comp repro|prod|debug`, `--clean cleanall|clean|noclean`, `--bit 32bit|64bit` (default `32bit`: 32-bit dynamical core, 64-bit physics), `--pic/--no-pic`, `--avx-level`, and `--fflags`.

`tools/build_ufs.py` installs native UFS_UTILS release 1.12.0 to `bin/preprocessing`. It is the newest release whose `chgres_cube` writes the surface file format SHiELD reads; 1.13.0 and later write a different format and cannot be used. Setting `ufs_bin_dir` selects the matching workflow. The build writes `bin/preprocessing/manifest.yaml` with the UFS_UTILS release, module and library versions, and source tags and commits; each run logs the release. `setup_all.sh` supports `--skip-env`, `--skip-shield`, `--skip-ufs`, `--skip-fix`, `--release`, and `-j`.

## 13. Examples and troubleshooting

| Example under `config/examples/` | Use |
| --- | --- |
| `run_config.uniform.yaml` | Global uniform grid. |
| `run_config.stretch.yaml`, `run_config.nest.yaml` | Stretched or telescoping nested grids. |
| `run_config.regional.yaml` | Regional ESG grid. |
| `run_config.external_ic.yaml` | Pre-staged initial conditions. |
| `run_config.monthly.yaml` | Monthly, multi-year integration. |

If submission fails, check the validation message, executable and fix paths, and whether the submission directory lies inside `case_root`. If preprocessing fails, inspect `LOGS/preprocess/`, dataset availability, and `chgres_cube` overrides. If a model segment fails, inspect the driver and model logs, correct the cause, and use `case_continue.sh` rather than starting over. `tests/submit.sh` submits the packaged HPC test cases.

## References

- [SHiELD](https://www.gfdl.noaa.gov/shield/)
- [FV3 documentation](https://www.gfdl.noaa.gov/fv3/fv3-documentation-and-references/) and [FV3 namelist guide](https://www.gfdl.noaa.gov/wp-content/uploads/2017/09/fv3_namelist_Feb2017.pdf)
- [UFS_UTILS](https://noaa-emcufs-utils.readthedocs.io/en/latest/pyufs.html)
- [FMS](https://noaa-gfdl.github.io/FMS/md_docs_doxygenGuide.html)
