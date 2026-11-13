# plugins: optional modules

`src/` is the core. Everything that is not needed by every run lives here:
experiments (`pamip` with the external files, `sm_perturbations`) and derived variables
(`derived`). The core never imports a plugin by name, except `plugins.derived` for the derived variables. `src/fv3_plugins.py` is the only
module in `src/` that imports `plugins`.

## How the core and a plugin communicate

A plugin never imports from `src/`. The core passes what a hook needs as keyword
arguments.

| Direction | What | How |
|---|---|---|
| core to plugin | run state | `state=state`, the live state: assigning to it changes the run |
| core to plugin | core functions | passed as keyword arguments by the call in `src/fv3_plugins.py`, for example `cp=fv3_utils.cp, open_grib=grib_io.open_grib` |
| plugin to core | namelist settings | `patch_nml({group: {key: value}})`, passed to the hook that needs it |
| plugin to core | everything else | files (staged inputs, boundary files, perturbed soil state) |
| core to core | which plugins exist | `plugins.__all__`, `ACTIVE_KEYS` and `HASH_KEYS` of each plugin |

`src/fv3_plugins.py` has one wrapper per plugin (`stage_derived_module`, `apply_pamip`, `apply_sm_perturbations`), and each lists the core functions that plugin receives and imports only its own dependencies. A hook that needs another core function gets one more keyword there. Pass what
an internal function needs on to it as explicit keyword arguments. Log with
`logging.getLogger("PLUGINS")` and say what is happening, not the options the user
already set in `run_config.yaml`.

Assign to `state` only to normalise a value in `validate` (for example an empty
string to `None`).

## Registration

`plugins/__init__.py` imports each plugin package and lists it in `__all__`, which
is the order the plugins run. Each plugin package exports `ACTIVE_KEYS` (the core
calls its hooks only when one of these `run_config.yaml` keys is set), `HASH_KEYS`
(keys added to the case checksum when set) and its hooks, so the core reads
`plugins.<name>.ACTIVE_KEYS`. A key that has a release default, such as
`pamip_forcing_year`, must not be an active key. Importing a plugin package pulls
in no heavy dependency (the hooks import their numerical code when called), so a
broken or unavailable dependency of one plugin cannot affect other runs or the
regridding container.

## Hooks

Each hook is a module-level function exported by `plugins/<name>/__init__.py` with
keyword-only arguments. Both are optional.

| Hook | Called | Returns |
|---|---|---|
| `validate(*, state)` | `fv3_plugins.validate()`, before grid and IC generation, so a bad configuration fails early (also in `preprocess_only` runs) | nothing; raise for a bad configuration |
| `apply(*, state, restart, <functions>)` | `fv3_plugins.do_plugins(restart=...)`, the last step of the cold start (`restart=False`) and of each restart segment (`restart=True`) | nothing |

`do_plugins` runs the active plugins in the order of `plugins.__all__`, after the core has
written the namelists, tables and fixed files and the run script, and before the state is saved. The core has
already written the namelists; a plugin changes them with `patch_nml(settings)`, which
merges `{group: {key: value}}` into the global and nest namelists and then applies the
user's namelist override files again, so the user's files always win. A plugin that
changes a namelist value that `update_fixed_files` reads (pamip sets `ictm`) is passed
`update_fixed_files` and calls it after patching.

`apply` is called on the cold start and on every restart segment. A plugin that acts on
the cold start only returns when `restart` is true (pamip does). A cold start with
`CASE_RESTART_DIR` calls `do_plugins` twice in one process, first with `restart=False`
and then with `restart=True`.

Later steps read the written namelist instead of asking a plugin. For example,
`update_fixed_files` reads `ictm` from `input.nml` to choose the CO2 table.

## Call order

```
cold start (init_driver)
  fv3_plugins.validate()           before grid and IC generation
  ... grid and IC generation, compute_checksum
  update_nml_configs()
  update_table_files()
  update_fixed_files()
  gen_shield_run_sh()
  fv3_plugins.do_plugins(restart=False)
  save_fv3_state()

restart segment (restart_driver)
  restart_config -> update_fixed_files -> update_table_files -> link_bc_to_input
  gen_shield_run_sh()
  fv3_plugins.do_plugins(restart=True)
  save_fv3_state()
```

`validate` stays early so that a bad configuration fails before the slow grid and IC
step, also in `preprocess_only` runs, and because it normalises hashed keys before the
checksum is computed.

## Derived variables

`plugins/derived/derived_vars.py` holds the built-in functions and lists them in `derive`; `use_derived.py` appends the functions of `derived_vars_module`.
A function has the signature

```python
def calc_x(ds: xr.Dataset, *, state, variables: dict[str, str], **_) -> xr.Dataset:
```

`variables` maps each native diag_table field name to the output name chosen in the
table. Name the arguments you use and let `**_` take the rest, so the core can pass
more later without breaking your functions.

To add your own, set `derived_vars_module` in `run_config.yaml` to a Python file
that defines `derive = [func, ...]`. The built-ins run first, then your list in
order. Functions are called once per output stream, and some streams lack the
inputs (for example a surface-only stream has no pressure levels), so return `ds`
unchanged when an input is missing, as the built-ins do. The file's directory is appended to `sys.path`, so helper files next to it
import normally. The file is copied to `derived_vars_user.py`
in the case directory at the cold start. Derived variables are post-processing: they do not
affect the model or the restart checksum.

## Adding an experiment

1. Create a package `plugins/<name>/` whose `__init__.py` re-exports the hooks from its modules (code stays out of `__init__.py`).
2. Export `ACTIVE_KEYS` and `HASH_KEYS` (from a light `config.py`) and the hooks in its `__init__.py`, import the package in `plugins/__init__.py` and add its name to `__all__`.
   Add a wrapper `apply_<name>(*, restart)` to `src/fv3_plugins.py` that passes the core functions and register it in `APPLY`.
3. Declare its `run_config.yaml` keys as fields of `FV3State` in `src/fv3_state.py`
   and give their defaults in `config/<release>/run_config.yaml`.
