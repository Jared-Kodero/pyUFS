import math

import numpy as np
import pandas as pd
from fv3_state import state

BASE_TIMINGS = {
    48: {"dt_atmos": 1800, "k_split": 2, "n_split": 6},
    96: {"dt_atmos": 900, "k_split": 2, "n_split": 6},
    192: {"dt_atmos": 900, "k_split": 2, "n_split": 6},
    384: {"dt_atmos": 450, "k_split": 2, "n_split": 6},
    768: {"dt_atmos": 225, "k_split": 2, "n_split": 6},
    1152: {"dt_atmos": 150, "k_split": 2, "n_split": 6},
    3072: {"dt_atmos": 90, "k_split": 2, "n_split": 10},
}


def _extrapolate_dt(C: int) -> int:
    df = pd.DataFrame(
        [(k, v["dt_atmos"]) for k, v in BASE_TIMINGS.items()],
        columns=["c", "dt_atmos"],
    )
    log_c = np.log(df["c"].values)
    log_dt = np.log(df["dt_atmos"].values)
    slope, intercept = np.polyfit(log_c, log_dt, 1)
    dt_est = np.exp(intercept) * C**slope
    valid = np.array([d for d in range(1, 3601) if 3600 % d == 0])
    return int(valid[np.argmin(np.abs(valid - dt_est))])


def _cres_timing(C: int) -> dict:
    if C in BASE_TIMINGS:
        timing = BASE_TIMINGS[C]
        return {
            "ideal_dt": timing["dt_atmos"],
            "k_split": timing["k_split"],
            "n_split": timing["n_split"],
        }
    else:
        return {
            "ideal_dt": _extrapolate_dt(C),
            "k_split": 2,
            "n_split": 10 if C >= 3072 else 6,
        }


def get_best_guess_timings() -> dict:
    c_res = state.c_res
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
        "k_split": optimum_k,
        "n_split": optimum_n,
    }


def get_timings() -> dict:

    best_guess_timings = get_best_guess_timings()
    dt_atmos = state.dt_atmos or best_guess_timings["dt_atmos"]
    dt_ocean = state.dt_ocean or best_guess_timings["dt_ocean"]
    k_split = state.k_split or best_guess_timings["k_split"]
    n_split = state.n_split or best_guess_timings["n_split"]

    if len(k_split) != state.n_nests + 1:
        raise ValueError(
            f"Length of k_split ({len(k_split)}) does not match number of domains ({state.n_nests + 1})"
        )
    if len(n_split) != state.n_nests + 1:
        raise ValueError(
            f"Length of n_split ({len(n_split)}) does not match number of domains ({state.n_nests + 1})"
        )

    timings = {}

    timings["dt_atmos"] = dt_atmos
    timings["dt_ocean"] = dt_ocean
    timings["k_split"] = k_split
    timings["n_split"] = n_split

    return timings
