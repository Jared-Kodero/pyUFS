"""PAMIP experiments (Smith et al. 2019) and user SST, sea-ice and land-mask files that
replace the SHiELD surface climatologies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fv3_utils import open_yaml

from . import external_files

# Keys of the `pamip` section of the plugins file. experiment is quoted text such as
# "1.6"; baseline is pamip, shield or external; fields is both, sst or sic. The external_*
# files are GRIB files in the format of the SHiELD climatologies they replace.
DEFAULTS = {
    "experiment": None,
    "baseline": None,
    "fields": None,
    "forcing_year": 2000,
    "external_sst": None,
    "external_sic": None,
    "external_mask": None,
}
HASHED = True

REGISTRY_FILE = Path(__file__).resolve().parent / "registry.yaml"
CONTROL = "1.1"
BASELINES = ("pamip", "shield", "external")
FIELDS = ("both", "sst", "sic")

SOM_RESTORE_DAYS = 90.0

FORCING_VARYING = ("5.1", "5.2")
PAMIP_EXPERIMENTS: dict[str, str] = {
    "1.1": "pdSST-pdSIC",
    "1.2": "piSST-piSIC",
    "1.3": "piSST-pdSIC",
    "1.4": "futSST-pdSIC",
    "1.5": "pdSST-piArcSIC",
    "1.6": "pdSST-futArcSIC",
    "1.7": "pdSST-piAntSIC",
    "1.8": "pdSST-futAntSIC",
    "1.9": "pdSST-pdSICSIT",
    "1.10": "pdSST-futArcSICSIT",
    "2.1": "pa-pdSIC",
    "2.2": "pa-piArcSIC",
    "2.3": "pa-futArcSIC",
    "2.4": "pa-piAntSIC",
    "2.5": "pa-futAntSIC",
    "3.1": "pdSST-futOkhotskSIC",
    "3.2": "pdSST-futBKSeasSIC",
    "4.1": "modelSST-pdSIC",
    "4.2": "modelSST-futArcSIC",
    "5.1": "amip-climSST",
    "5.2": "amip-climSIC",
    "6.1": "pa-pdSIC-ext",
    "6.2": "pa-futArcSIC-ext",
    "6.3": "pa-futAntSIC-ext",
}


def forcing_year(config: dict) -> int | None:
    """Year of the fixed CO2 forcing, or None when CO2 follows the run years."""
    if config["experiment"] is None or config["experiment"] in FORCING_VARYING:
        return None
    return config["forcing_year"]


def validate(*, config: dict) -> None:
    """Check the external files, experiment, baseline and fields of the pamip section."""
    external_files.validate(config)
    experiment = config["experiment"]
    baseline = config["baseline"]
    fields = config["fields"]

    if experiment is None or experiment == "":
        config["experiment"] = None
        if baseline is not None or fields is not None:
            raise ValueError(
                "`baseline` and `fields` only apply to a PAMIP experiment: "
                + 'set `experiment` (for example "1.6") or set both options to null'
            )
        return

    if not isinstance(experiment, str):
        raise TypeError(
            f'`experiment` must be quoted text such as "1.6". Got {experiment!r}'
        )
    experiment = experiment.strip()
    if experiment not in PAMIP_EXPERIMENTS:
        raise ValueError(
            f"`experiment` must be one of {tuple(PAMIP_EXPERIMENTS)}. Got {experiment!r}"
        )
    registry = open_yaml(REGISTRY_FILE)["experiments"]
    if experiment not in registry:
        raise ValueError(
            f"PAMIP experiment {experiment} has no SST and sea-ice files in {REGISTRY_FILE.name}."
        )
    if baseline is not None and baseline not in BASELINES:
        raise ValueError(
            f"`baseline` must be one of {BASELINES} or null. Got {baseline!r}"
        )
    if fields is not None and fields not in FIELDS:
        raise ValueError(f"`fields` must be one of {FIELDS} or null. Got {fields!r}")
    config["experiment"] = experiment
    if baseline == "external":
        missing = [
            key
            for key in ("external_sst", "external_sic", "external_mask")
            if not config[key]
        ]
        if missing:
            raise ValueError(
                f"`baseline: external` requires all external files. Not set: {missing}"
            )


def _experiment_settings(config: dict, state: Any) -> dict[str, dict]:
    """Write the boundary files and return the namelist settings of the experiment."""
    from .boundary import set_cold_start_surface, write_boundary_files

    fields, files = write_boundary_files(config=config, state=state)
    if state.restart_no == 0:
        set_cold_start_surface(fields, state=state)

    # Sea-ice mask and concentration follow the PAMIP FIXED climatologies at every
    # surface cycle; fsicl covers points already classified as ice (non-open-ocean
    # branch of the sfcsub merge). The SST is set by the slab ocean (som_mlm.F90) at
    # open-water points, which restores it toward the SST climatology (restore_method 1)
    # with time scale SOM_RESTORE_DAYS; with do_ocean the surface cycle no longer
    # overwrites it, and ftsfs only acts on ts_clim_iano, which method 1 does not use.
    # Beyond 60 degrees the slab needs maxlat and the quasi sea-ice albedo switched
    # off (som_mlm.F90).
    physics = {
        "use_ext_sst": False,
        "nstf_name": [0, 0, 1, 0, 5],
        "do_ocean": True,
        "disable_radiation_quasi_sea_ice": True,
    }
    year = forcing_year(config)
    if year is not None:
        # Fixed CO2 forcing, e.g. ictm = 20200 for the CO2 of year 2020
        physics["ictm"] = year * 10
    return {
        "namsfc": {"faiss": 0, "fsics": 0, "fsicl": 0, **files},
        "gfs_physics_nml": physics,
        "ocean_nml": {
            "ocean_option": "SOM",
            "mld_option": "obs",
            "mld_obs_ratio": 1.0,
            "restore_method": 1,
            "sst_restore_tscale": SOM_RESTORE_DAYS,
            "start_lat": -90.0,
            "end_lat": 90.0,
            "maxlat": 90.0,
        },
    }


def cold_start(*, config: dict, state: Any) -> dict[str, dict]:
    """Stage the external files and the experiment; return the namelist settings."""
    settings = external_files.stage(config, state)
    if config["experiment"] is not None:
        for group, entries in _experiment_settings(config, state).items():
            settings.setdefault(group, {}).update(entries)
    return settings
