"""Optional pyUFS modules: experiments and derived variables.

Each plugin is a subpackage that exports its hooks, optionally DEFAULTS (the keys of
its section in the plugins file with their defaults) and HASHED (the settings are part
of the case checksum). The core calls the hooks of a plugin only when its section is in
the plugins file. Importing a plugin package imports no heavy dependency; the hooks
import their numerical code when called.
``src/fv3_plugins.py`` is the only module in ``src/`` that imports this package.
See ``plugins/README.md`` for the hook contract.

``__all__`` lists the plugins in the order they run.
"""

from . import derived, pamip, sm_perturbations, sst_perturbation

__all__ = ["derived", "pamip", "sm_perturbations", "sst_perturbation"]
