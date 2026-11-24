from collections.abc import Callable
from typing import Any


def apply(
    *,
    config: dict,
    state: Any,
    restart: bool,
    copy_path: Callable,
    segment_hours: Callable,
    load_grid: Callable,
    sfc_filename: Callable,
    to_fv3_grid: Callable,
) -> None:
    from plugins.sm_perturbations.soil_moisture import apply_sm_perturbations

    apply_sm_perturbations(
        config=config,
        state=state,
        copy_path=copy_path,
        segment_hours=segment_hours,
        load_grid=load_grid,
        sfc_filename=sfc_filename,
        to_fv3_grid=to_fv3_grid,
    )
