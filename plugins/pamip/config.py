"""Keys, constants and the experiment table of the pamip plugin (light imports only)."""

from __future__ import annotations

from pathlib import Path

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
