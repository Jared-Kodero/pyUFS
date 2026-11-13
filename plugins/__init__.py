"""Optional pyUFS modules: experiments and derived variables.

Each plugin is a subpackage that exports ACTIVE_KEYS (run_config keys; the core
calls the plugin's hooks only when one of them is set, not None), HASH_KEYS (keys
added to the case checksum when set) and its hooks. Importing a plugin package
imports no heavy dependency; the hooks import their numerical code when called.
``src/fv3_plugins.py`` is the only module in ``src/`` that imports this package.
See ``plugins/README.md`` for the hook contract.

``__all__`` lists the plugins in the order they run.
"""

from . import derived, pamip, sm_perturbations

__all__ = ["derived", "pamip", "sm_perturbations"]
