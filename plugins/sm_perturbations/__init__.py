"""Perturbations of the soil state in the surface files of the cold start and of the
restart segments. The section is the perturbation mapping itself; soil_moisture.py
checks its keys."""

from typing import Any

HASHED = False


def cold_start(*, config: dict, state: Any) -> None:
    from .soil_moisture import apply_sm_perturbations

    apply_sm_perturbations(config=config, state=state)


def restart(*, config: dict, state: Any) -> None:
    from .soil_moisture import apply_sm_perturbations

    apply_sm_perturbations(config=config, state=state)
