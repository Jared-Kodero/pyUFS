import math
from typing import Literal

import numpy as np
import pandas as pd

from fv3gfs_state import state

BASE_TIMINGS = {
    48: {"dt": 3600, "k_split": 2, "n_split": 6},
    96: {"dt": 1800, "k_split": 2, "n_split": 6},
    192: {"dt": 900, "k_split": 2, "n_split": 6},
    384: {"dt": 450, "k_split": 2, "n_split": 6},
    768: {"dt": 225, "k_split": 2, "n_split": 6},
    1152: {"dt": 150, "k_split": 2, "n_split": 6},
    3072: {"dt": 90, "k_split": 2, "n_split": 10},
}


def _get_user_timings(name: Literal["global", "nest"], nest=None) -> dict:
    timings = {}
    if name == "global":
        if state.k_split is not None:
            timings["k_split"] = state.k_split
        if state.n_split is not None:
            timings["n_split"] = state.n_split
        if state.dt_atmos is not None:
            timings["dt_atmos"] = state.dt_atmos
            timings["dt_ocean"] = state.dt_ocean

    elif name == "nest":
        if nest is None:
            raise ValueError("nest must be provided for nest timing overrides")

        nest_idx = nest - 1  # zero-based index

        if state.nest_k_split is not None:
            timings["k_split"] = state.nest_k_split[nest_idx]
        if state.nest_n_split is not None:
            timings["n_split"] = state.nest_n_split[nest_idx]

    return timings


def apply_user_timings(nml, name: Literal["global", "nest"], nest=None) -> dict:
    # check for user timing overrides suplied in cli args or config
    timings_overrides = _get_user_timings(name, nest)
    if not timings_overrides:
        return nml

    if nest is None:
        dt_atmos = timings_overrides.get("dt_atmos")
        dt_ocean = timings_overrides.get("dt_ocean")
        if dt_atmos is not None:
            nml["coupler_nml"]["dt_atmos"] = dt_atmos
        if dt_ocean is not None:
            nml["coupler_nml"]["dt_ocean"] = dt_ocean

    n_split = timings_overrides.get("n_split")
    k_split = timings_overrides.get("k_split")
    if n_split is not None:
        nml["fv_core_nml"]["n_split"] = n_split
    if k_split is not None:
        nml["fv_core_nml"]["k_split"] = k_split

    return nml


def _extrapolate_dt(C: int) -> int:
    df = pd.DataFrame(
        [(k, v["dt"]) for k, v in BASE_TIMINGS.items()],
        columns=["c", "dt"],
    )
    log_c = np.log(df["c"].values)
    log_dt = np.log(df["dt"].values)
    slope, intercept = np.polyfit(log_c, log_dt, 1)
    dt_est = np.exp(intercept) * C**slope
    valid = np.array([d for d in range(1, 3601) if 3600 % d == 0])
    return int(valid[np.argmin(np.abs(valid - dt_est))])


def _cres_timing(C: int) -> dict:
    if C in BASE_TIMINGS:
        timing = BASE_TIMINGS[C]
        return {
            "ideal_dt": timing["dt"],
            "k_split": timing["k_split"],
            "n_split": timing["n_split"],
        }
    else:
        return {
            "ideal_dt": _extrapolate_dt(C),
            "k_split": 2,
            "n_split": 10 if C >= 3072 else 6,
        }


def get_first_guess_timings() -> dict:
    c_res = state.res
    n_nests = state.n_nests
    refine_ratio = state.refine_ratio
    nest_type = state.nest_type

    # 1. Map out resolutions for all domains
    c_vals = [c_res]
    if n_nests > 0:
        current_c = c_res
        for i in range(n_nests):
            ratio = refine_ratio[i] if i < len(refine_ratio) else refine_ratio[-1]
            current_c = (
                current_c * ratio if nest_type == "telescoping" else c_res * ratio
            )
            c_vals.append(current_c)

    # 2. Extract targets
    ideal_timings = [_cres_timing(c) for c in c_vals]
    dt = ideal_timings[-1]["ideal_dt"]

    optimum_k = []
    optimum_n = []

    # 3. Calculate optimal splits for each specific domain
    for i, ideal in enumerate(ideal_timings):
        target_acoustic = ideal["ideal_dt"] / (ideal["k_split"] * ideal["n_split"])
        is_finest = i == len(ideal_timings) - 1

        k_split = 2 if is_finest else 1
        n_split = max(1, math.ceil(dt / (k_split * target_acoustic)))

        optimum_k.append(k_split)
        optimum_n.append(n_split)

    return {
        "dt_atmos": dt,
        "dt_ocean": dt,
        "global_k_split": optimum_k[0],
        "global_n_split": optimum_n[0],
        "nest_k_splits": optimum_k[1:],
        "nest_n_splits": optimum_n[1:],
    }
