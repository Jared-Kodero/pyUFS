# FV3State audit

Reviewed at `ufs_py` 6574559a13a6 plus this change set. `FV3State` is a `dict`;
its class annotations document keys but initialize nothing, and attribute access
to an absent key returns `None` (`FV3State.__getattr__`). A misspelled or
not-yet-assigned key therefore reads as `None` rather than raising. The table
below lists every key read by `py_scripts/*.py` (static scan of `state.<key>`,
`state["<key>"]`, `state.get("<key>")`) with its source.

## Sources, in the order they are applied

| Stage | Where | What it sets |
|---|---|---|
| 1. Configuration | `fv3_init_driver._load_initial_state` | Case `run_config.yaml`; a missing or null case key is filled from `configs/run_config.yaml`. |
| 2. Environment | `fv3_utils.runtime_env_vars` | `case_name` (when `CASE_NAME` is set), `n_cpus`, `n_nodes`, `n_cpus_per_node`, `multi_node`, `ensemble_id`, `n_ensembles`, `resubmit` (`CASE_RESUBMIT_MAX`), `resubmit_idx`. These override stage 1. |
| 3. Derived (init) | `_load_initial_state` | `case_description`, `description`, `init_datetime` (parsed), `run_config`, `c_res` (parsed), `continue_run`, `warm_start`, `restart_no = 0`, `resubmit_idx = 0`, `total_restarts`, `total_run_hours`, `forecast_length`, `refine_ratio` (list, or `1` for non-nested grids), `res_km`, `n_nests`; regional bounds coerced to lists; `tgrad_perturbations` validated (not modified). |
| 4. Paths | `fv3_paths.paths` via `configure_directories` and `load_fv3_state` | `work_dir`, `fix_src`, `run_dir`, `case_dir`, `archive_dir` from the environment; `ufs_exe = /UFS_UTILS/exec`; `ufs_utils` and `configs` from the location of the `ufs_py` checkout; `tmp`, `hist`, `grid`, `logs`, `fix`, `input`, `output`, `restarts`, `ic_data`, `bc_data` under `work_dir`. `ufs_utils` and `fix_src` override the configuration values of the same name. |
| 5. Preprocessing | grid, nesting, PE and IC modules | `nest_type` and target adjustments (`validate_nests`); `parent_tile`, `istart_nest`, `iend_nest`, `jstart_nest`, `jend_nest`, `nest_ioffsets`, `nest_joffsets` (dynamic keys in `fv3_nesting`); `c_res` replaced by the equivalent resolution for regional grids; `ngrid_cells`, `ntiles`, `npx`, `npy`, `layout`, `io_layout`, `blocksize`, `grid_pes`, `total_pes`; `<domain>_ic_source` (dynamic, `chgres_cube.run_chgres`); `generate_ic_data = False` after conversion; preprocess flags reset. |
| 6. Namelists | `fv3_namelists.update_nml_configs` | `model_start_date`, `dt_atmos`, `dt_ocean`, `k_split`, `n_split`; `checksum` (init driver). |
| 7. Serialization | `save_fv3_state` | All keys except the stage-4 `case_paths` group; `Path` values become strings; `init_datetime` as `YYYYMMDDHHZ`. |
| 8. Restart load | `load_fv3_state`, `fv3_restart_driver._load_restart_state` | Persisted keys, then stage-4 paths (as `Path`), then stage-2 environment; `restart_no`, `resubmit_idx`, `total_restarts`, `continue_run = True`, `warm_start = True`; checksum compared. `run_config.yaml` is not reread. |
| 9. External IC | `fv3_external_ic.init_external_ic` | Bundle `state.yaml` supplies grid and IC keys (`BUNDLE_KEYS`, `<domain>_ic_source`); since this change the case keeps all other keys, and `init_datetime`, `gtype`, `levels` must match the bundle. |

Keys that were `Path` objects at initialization and are not in stage 4
(`run_config`, `external_ic_dir`, image paths) are strings after a restart. No
restart-path read applies a `Path` operation to them.

## Required and conditional keys by path

| Path | Keys read | Source |
|---|---|---|
| All | `init_datetime`, `c_res`, `gtype`, `levels`, `run_length`, `run_length_units`, `resubmit`, `description`, `do_deep`, `fv3_debug`, `use_modern_diag`, `merge_freq`, `pack_output`, `tgrad_perturbations`, `sm_perturbations`, `modules`, `shield_exe`, `jobtmp`, `case_root`, `archive_root`, `containers_root`, image paths | 1 (defaults exist for all except `init_datetime`, `run_length`, which are required) |
| All | `n_cpus`, `n_nodes`, `n_cpus_per_node`, `multi_node`, `resubmit`, `resubmit_idx` | 2 |
| All | `total_restarts`, `total_run_hours`, `restart_no`, `res_km`, `n_nests`, `refine_ratio`, `checksum` | 3, 6, 8 |
| All (model run) | `npx`, `npy`, `ntiles`, `layout`, `io_layout`, `blocksize`, `grid_pes`, `total_pes`, `dt_atmos`, `dt_ocean`, `k_split`, `n_split` | 5, 6 (`dt_*`, `*_split` may be null in the configuration; then `fv3_timings` computes them) |
| `uniform` / `stretch` | `stretch_factor`, `target_lon`, `target_lat` | 1 |
| `nest` | `refine_ratio` (list), `lon_min`, `lon_max`, `lat_min`, `lat_max`, `parent_tile`, `halo`; derived `nest_type`, nest indices and offsets | 1, 5 |
| `regional_gfdl` / `regional_esg` | bounds (validated non-null), `halo`, `idim`, `jdim`, `delx`, `dely`; `bc_data`, `total_run_hours`, `regional_ic_source` (read with a GFS fallback) | 1, 3, 4, 5 |
| Generated IC | `fix_src`, `forecast_hour`, `add_lake`, `lake_cutoff`, `make_gsl_orog`, `ufs_exe`; nest `external_model` is read from `chgres_cube_nestNN.yaml`, not from state | 1, 4 |
| External IC | `external_ic_dir`, bundle `state.yaml` | 1, 9 |
| Ensemble | `ensemble_run`, `ensemble_id`, `n_ensembles`; perturbations generated only on the generated-IC path | 1, 2 |
| `preprocess_only` | preprocess flags; exits before tgrad files, namelists and run scripts | 1 |
| Regridding (fregrid container) | `c_res`, `gtype`, `n_nests`, `nest_type`, `refine_ratio`, bounds, `restart_no`, `resubmit`, `resubmit_idx`, `total_restarts`, `merge_freq`, `pack_output`, `description`, `case_description` | persisted state only (no environment) |

Physical namelist settings (`fdiag`, `fhzero`, `ftsfs`, `ico2`, `ictm`, `isol`,
...) live in the namelist dictionaries and the case `input*.yaml|nml`
overrides; they are not top-level state keys.

## Findings

| Finding | Type | Status |
|---|---|---|
| `model_start_date` assigned, not annotated | Annotation omission | Annotated `list[int]` |
| `regional_ic_source` read and dynamically assigned, not annotated | Annotation omission | Annotated; IC-source mappings typed `dict[str, str \| None]` |
| `jobtmp`, `case_root`, `archive_root`, `containers_root` read by `parse_dirs`, not annotated | Annotation omission | Annotated |
| `refine_ratio` is `1` for non-nested grids; image paths are strings | Annotation imprecision | Documented here |
| External-IC route replaced the whole state with the bundle's `state.yaml`, discarding the case `description`, segment settings, output settings and `tgrad_perturbations` (reproduced: an SST4 case reverted to the bundle's `None`) | Runtime defect on the external-IC path | Fixed in `fv3_external_ic.py`; tested |
| `istart_nest` and related keys are read by the init driver before preprocessing assigns them | By design (`None` passed to `run_driver`) | No change |
| Restart checksum covers grid keys, `init_datetime` and, when set, `tgrad_perturbations`; it does not cover `run_length`, `resubmit` or namelist overrides | Design note | Documented |
| No required key was found unassigned on the traced global, nested, regional, generated-IC, external-IC or restart paths | Audit result | Static trace and synthetic tests; not exercised on HPC |
