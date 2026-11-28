# plugins: optional modules

`core/` is the core. Everything that is not needed by every run lives here:
the experiments `pamip` (with the external files), `sst_perturbation` and
`sm_perturbations`. `core/fv3_plugins.py` is the only module in
`core/` that loads plugins, and it loads them by name, so adding a plugin edits nothing in
`core/`.

## What a plugin is

A folder `<name>/` with an `__init__.py`, in `pyUFS/plugins/` or in a folder listed under
`plugin_dirs` in the plugins file. The `__init__.py` holds the settings and the hooks;
larger code goes in modules next to it, imported relatively (`from .boundary import ...`)
inside the hook that needs them, so importing a plugin pulls in no heavy dependency and a
broken dependency of one plugin cannot affect other runs or the regridding stage. A
plugin defines only the hooks it needs, at least one; a hook it does not define is
skipped, and one it defines may return None when it has nothing to do. `DEFAULTS` and
`HASHED` are optional.

```python
# keys of the section and their defaults (None: no default)
DEFAULTS = {"delta": 1.0}
# the settings are part of the case checksum
HASHED = True


def validate(*, config: dict) -> None: ...


def cold_start(*, config: dict, state) -> dict | None: ...


def restart(*, config: dict, state) -> dict | None: ...


def regrid(ds, *, config: dict, state, variables) -> xr.Dataset | None: ...
```

A plugin imports from `core/` only the state-free library modules `fv3_utils`, `grib_io`
and `latlon_to_fv3` (for example `from fv3_utils import copy_path, ic_filename`). It
receives the run state as `state` and never imports `fv3_state`, `fv3_paths` or
`fv3_runtime`.

| Direction | What | How |
|---|---|---|
| user to plugin | settings | `config`, the plugin's section of the plugins file with the defaults filled in; the same dict is in `state.plugins[name]`, so changing it in `validate` changes what is saved and hashed |
| core to plugin | run state | `state`, the live state: assigning to it changes the run |
| plugin to core | namelist settings | the return value of `cold_start` or `restart`, `{group: {key: value}}` |
| plugin to core | everything else | files (staged inputs, boundary files, perturbed soil state) |

Log with `logging.getLogger("PLUGINS")` and say what is happening, not the options the
user already set in the plugins file. Assign to `config` only to normalise a value in
`validate` (for example an empty string to `None`). A plugin's packages must be in the
preprocess image (container mode) or in `conda_env/pyufs` (native mode, `tools/env.yaml`).

## Plugins file

`plugins_config` in `run_config.yaml` names a YAML file that maps plugin names to their
settings, and may list extra plugin folders:

```yaml
plugin_dirs: [~/pyufs_exp]   # searched first, then pyUFS/plugins; relative to this file
pamip: {experiment: "1.6"}
t2m_celsius: true             # a plugin without settings
```

The cold start reads it with `fv3_plugins.load_plugins` into `state.plugins`, a plain
dict of dicts, and `state.plugin_dirs`, both stored in `state.yaml`, so restarts and the
regridding stage never read the file again. A plugin is active when its name is a key of
`state.plugins`; null, false and empty sections are dropped, and `true` enables a plugin
with its defaults. Active plugins run in the order the
file lists them, so a plugin that uses what another sets is listed after it. A plugin with `DEFAULTS` gets its section
merged over them by `fv3_plugins.resolve`, which rejects an unknown key with the closest
valid key suggested; one without `DEFAULTS` (`sm_perturbations`) gets the mapping as
written and checks its own keys. An unknown plugin name stops the run with the closest
available name suggested.

## Hooks

| Hook | Called | Returns |
|---|---|---|
| `validate(*, config)` | `fv3_plugins.validate()`, before grid and IC generation, so a bad configuration fails early (also in `preprocess_only` runs) | nothing; raise for a bad configuration |
| `cold_start(*, config, state)` | `fv3_plugins.apply("cold_start")`, the last step of the cold start | namelist changes or `None` |
| `restart(*, config, state)` | `fv3_plugins.apply("restart")`, the last step of each restart segment | namelist changes or `None` |
| `regrid(ds, *, config, state, variables)` | `fv3_plugins.apply("regrid", ds, variables)`, once per output stream, after the built-in derived variables | a new dataset or `None` (ds unchanged) |

Each hook a plugin defines is checked against these signatures when the plugin loads, so
a misnamed argument stops the run at the start of preprocessing rather than at its end.

`apply("cold_start")` and `apply("restart")` run after the core has written the namelists, tables, fixed files and run
script, and before the state is saved. The namelist changes a hook returns are merged
into the global and nest namelists by `patch_nml` before the next plugin runs, and the
user's namelist override files are applied again, so the user's files always win. When
any plugin returned changes, the core then reruns `update_fixed_files`, which reads
`ictm` from `input.nml` to choose the CO2 table (pamip sets `ictm`). A cold start with
`CASE_RESTART_DIR` calls `cold_start` and then `restart` in one process.

## Call order

```
cold start (init_driver)
  fv3_plugins.validate()           before grid and IC generation
  ... grid and IC generation, compute_checksum
  update_nml_configs()
  update_table_files()
  update_fixed_files()
  gen_shield_run_sh()
  fv3_plugins.apply("cold_start")
  save_fv3_state()

restart segment (restart_driver)
  restart_config -> update_fixed_files -> update_table_files -> link_bc_to_input
  gen_shield_run_sh()
  fv3_plugins.apply("restart")
  save_fv3_state()
```

`validate` stays early so that a bad configuration fails before the slow grid and IC
step, also in `preprocess_only` runs, and because it normalises hashed keys before the
checksum is computed.

## Derived variables

`core/derived_variables.py` holds the built-in derived variables, which `fv3_regrid`
computes for every case and every output stream before the plugins' `regrid` hooks. Each
is a function

```python
def calc_x(ds: xr.Dataset, *, state, variables: dict[str, str], **_) -> xr.Dataset:
```

listed in `derive`. `variables` maps each native diag_table field name to the output
name chosen in the table. Functions are called once per output stream, and some streams
lack the inputs (for example a surface-only stream has no pressure levels), so they
return `ds` unchanged when an input is missing.

Your own derived variables are a plugin with a `regrid` hook in one of your
`plugin_dirs`; see `config/examples/plugins/t2m_celsius`. Derived variables are
post-processing: they do not affect the model or the restart checksum.

## Adding an experiment

1. Create a folder `<name>/` with an `__init__.py` in `pyUFS/plugins/` or in one of your `plugin_dirs`, defining `DEFAULTS` (omit it for a free-form mapping), `HASHED`, and the hooks it needs among `validate`, `cold_start`, `restart` and `regrid`.
2. Add a `<name>:` section to the case plugins file, after any plugin whose output it uses.
3. Document its section in this file. No change to `core/` or the `run_config.yaml` defaults is needed.
