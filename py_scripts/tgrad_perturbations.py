from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import eccodes
import numpy as np
from fv3_runtime import log
from fv3_state import state
from fv3_utils import cp
from scipy.spatial import cKDTree

SST_FILE = "RTGSST.1982.2012.monthly.clim.grb"
ICE_FILE = "CFSR.SEAICE.1982.2012.monthly.clim.grb"
LAND_FILE = "seaice_newland.grb"  # 0.5 deg land mask, 1 = land (verified on Oscar)

# Literature bounds on the polar SST change [K]. CMIP6 central-Arctic Aug-Sep
# SST warming by 2100 spans about 0.7-8 K across models under SSP2-4.5, about
# 3 K in moderately ice-free models (ERL 2023, doi:10.1088/1748-9326/ad0c8a);
# observed Arctic amplification 1979-2021 is 3.8 (Rantanen et al. 2022). +2
# and +4 K are standard sensitivity amplitudes; values up to 8 K are accepted
# as strong-forcing experiments; larger magnitudes are rejected.
DELTA_T_STANDARD = 4.0
DELTA_T_MAX = 8.0

T_FREEZE = 271.2  # K, con_tice in SHiELD physcons
ICE_EDGE = 0.15  # aislim in sfcsub: SIC >= 0.15 is treated as sea ice
OCEAN_FRACTION = (0.6, 0.8)  # plausible global ocean fraction; else mask is wrong
POLES = {"north": ("north",), "south": ("south",), "both": ("north", "south")}

Transform = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
_WRITTEN: set[str] = set()  # configurations already written in this process


def per_pole(value: float | dict, poles: str, name: str) -> dict[str, float]:
    """Amplitude per pole: a scalar applies to every pole selected by `poles`;
    a mapping {north: x, south: y} sets each pole explicitly.
    """
    if isinstance(value, dict):
        bad = [k for k in value if k not in ("north", "south")]
        if bad:
            raise ValueError(f"`{name}` keys must be north and/or south. Got {bad}")
        return {k: float(v) for k, v in value.items()}
    return {pole: float(value) for pole in POLES[poles]}


def monthly_scale(p: dict) -> list[float]:
    scale = p.get("monthly_scale") or [1.0] * 12
    if len(scale) != 12:
        raise ValueError("monthly_scale must have 12 values (Jan..Dec).")
    return [float(a) for a in scale]


def smoothstep_weight(phi: np.ndarray, p: dict) -> np.ndarray:
    """W = x^2 (3 - 2x), x = clip((phi - lat_start) / (lat_full - lat_start), 0, 1)."""
    span = p["lat_full"] - p["lat_start"]
    x = np.clip((phi - p["lat_start"]) / span, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def polar_pattern(lat: np.ndarray, p: dict, amps: dict[str, float]) -> np.ndarray:
    """Sum over poles of amplitude * W, with phi = lat (north) or -lat (south)."""
    out = np.zeros_like(lat, dtype=float)
    for pole, amp in amps.items():
        out += amp * smoothstep_weight(lat if pole == "north" else -lat, p)
    return out


def message_month(gid: int) -> int:
    return (eccodes.codes_get(gid, "dataDate") // 100) % 100


def read_grib(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {month: (values, latitudes, longitudes)}; bitmapped points are NaN."""
    fields = {}
    with open(path, "rb") as f:
        while (gid := eccodes.codes_grib_new_from_file(f)) is not None:
            vals = eccodes.codes_get_values(gid)
            if eccodes.codes_get(gid, "bitmapPresent"):
                vals = np.where(
                    vals == eccodes.codes_get(gid, "missingValue"), np.nan, vals
                )
            fields[message_month(gid)] = (
                vals,
                eccodes.codes_get_array(gid, "latitudes"),
                eccodes.codes_get_array(gid, "longitudes"),
            )
            eccodes.codes_release(gid)
    return fields


def read_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes of the first message (all messages share a grid)."""
    with open(path, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        lat = eccodes.codes_get_array(gid, "latitudes")
        lon = eccodes.codes_get_array(gid, "longitudes")
        eccodes.codes_release(gid)
    return lat, lon


def nearest_index(
    src_lat: np.ndarray, src_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Index of the nearest source point (on the sphere) for each target point."""

    def xyz(la: np.ndarray, lo: np.ndarray) -> np.ndarray:
        phi, lam = np.deg2rad(la), np.deg2rad(lo)
        return np.column_stack(
            (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))
        )

    _, idx = cKDTree(xyz(src_lat, src_lon)).query(xyz(lat, lon))
    return idx


def perturb_grib(src: Path, dst: Path, transform: Transform) -> tuple[float, float]:
    """Rewrite each GRIB message with transformed values, keeping all headers.

    Returns the minimum and maximum change over all messages.
    """
    d_min, d_max = 0.0, 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while (gid := eccodes.codes_grib_new_from_file(fin)) is not None:
            vals = eccodes.codes_get_values(gid)
            lat = eccodes.codes_get_array(gid, "latitudes")
            new = transform(vals, lat, message_month(gid))
            if eccodes.codes_get(gid, "bitmapPresent"):
                miss = vals == eccodes.codes_get(gid, "missingValue")
                new = np.where(miss, vals, new)
            d_min = min(d_min, float(np.min(new - vals)))
            d_max = max(d_max, float(np.max(new - vals)))
            out = eccodes.codes_clone(gid)
            eccodes.codes_set_values(out, new)
            eccodes.codes_write(out, fout)
            eccodes.codes_release(out)
            eccodes.codes_release(gid)
    return d_min, d_max


def ice_masks_on_sst_grid(
    ice_src: Path, ice_fn: Transform | None, idx: np.ndarray
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Perennial-ice mask and per-month opened-water masks on the SST grid.

    frozen: final SIC >= 0.15 in every month (land flags count as frozen).
    opened[m]: ice in the original field (0.15 <= SIC <= 1) but open water in
    the final field. Computed on the ice grid, mapped with the index idx.
    """
    frozen, opened = None, {}
    for month, (vals, lat, _) in read_grib(ice_src).items():
        sic0 = np.nan_to_num(vals, nan=0.0)
        sic1 = sic0 if ice_fn is None else ice_fn(sic0, lat, month)
        ice1 = sic1 >= ICE_EDGE
        frozen = ice1 if frozen is None else frozen & ice1
        opened[month] = ((sic0 >= ICE_EDGE) & (sic0 <= 1.0) & ~ice1)[idx]
    return frozen[idx], opened


def ocean_on_sst_grid(land_src: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Ocean mask on the SST grid from seaice_newland.grb (1 = land).

    Raises if the cos(lat)-weighted ocean fraction is implausible.
    """
    vals, mlat, mlon = next(iter(read_grib(land_src).values()))
    land = np.nan_to_num(vals, nan=1.0) >= 0.5
    ocean = ~land[nearest_index(mlat, mlon, lat, lon)]
    cosw = np.cos(np.deg2rad(lat))
    frac = float(np.sum(cosw * ocean) / np.sum(cosw))
    log.info(f"tgrad sst: ocean fraction from {land_src} = {frac:.3f}")
    if not OCEAN_FRACTION[0] <= frac <= OCEAN_FRACTION[1]:
        raise ValueError(f"Ocean fraction {frac:.3f} from {land_src} is implausible.")
    return ocean


def polar_shift(
    lat: np.ndarray, month: int, p: dict, active: np.ndarray | None, scale: list
) -> np.ndarray:
    """dT = A_m * sum_pole polar_delta_t_k * W(phi)   [K]."""
    return scale[month - 1] * polar_pattern(lat, p, p["dt_poles"])


def uniform_shift(
    lat: np.ndarray, month: int, p: dict, active: np.ndarray | None, scale: list
) -> np.ndarray:
    """polar_shift replaced by its monthly mean over the SST-active ocean
    (cos(lat) area weights; land and perennial ice excluded)."""
    pattern = polar_pattern(lat, p, p["dt_poles"])
    cosw = np.cos(np.deg2rad(lat)) * active
    mean = np.full_like(pattern, np.sum(pattern * cosw) / np.sum(cosw))
    return scale[month - 1] * mean


def ice_loss(
    vals: np.ndarray, lat: np.ndarray, month: int, p: dict, scale: list
) -> np.ndarray:
    """SIC' = SIC * (1 - min(1, A_m * sum_pole sic_reduction * W(phi)))."""
    loss = np.clip(scale[month - 1] * polar_pattern(lat, p, p["sic_poles"]), 0.0, 1.0)
    # Values outside [0, 1] are land flags (1.57 in the CFSR file); keep them.
    fraction = (vals >= 0.0) & (vals <= 1.0)
    return np.where(fraction, np.clip(vals * (1.0 - loss), 0.0, 1.0), vals)


def adjust_sea_ice(p: dict, methods: list, am_dir: Path) -> Transform | None:
    """Write FIXED sea ice; return the SIC transform when ice_loss is applied."""
    src = am_dir / ICE_FILE
    dst = Path(state.fix) / "MODS" / ICE_FILE

    if not src.exists():  # already reported by update_namsfc
        return None

    dst.unlink(missing_ok=True)  # never write through a symlink into fix_src

    if "ice_loss" not in methods:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cp(src, dst)
        log.info(f"tgrad sea_ice: unperturbed copy {src} -> {dst}")
        return None

    scale = monthly_scale(p)

    def ice_fn(vals: np.ndarray, lat: np.ndarray, month: int) -> np.ndarray:
        return ice_loss(vals, lat, month, p, scale)

    d_min, d_max = perturb_grib(src, dst, ice_fn)
    log.info(
        f"tgrad sea_ice: perturbed {src} -> {dst}; "
        f"change range [{d_min:.3f}, {d_max:.3f}]"
    )
    return ice_fn


def adjust_sst(p: dict, methods: list, am_dir: Path, ice_fn: Transform | None) -> None:
    """Write FIXED SST: dT = uniform_warming_k + shift method, plus
    opened_water_delta_t_k where ice_loss opens water. dT is zero at
    perennial-ice points; cooling never takes SST below T_FREEZE."""
    src = am_dir / SST_FILE
    dst = Path(state.fix) / "MODS" / SST_FILE

    if not src.exists():  # already reported by update_namsfc
        return

    dst.unlink(missing_ok=True)  # never write through a symlink into fix_src

    shift_methods = {"polar_shift": polar_shift, "uniform_shift": uniform_shift}
    shifts = [m for m in methods if m in shift_methods]
    opened_dt = 0.0
    if "ice_loss" in methods:
        opened_dt = float(p.get("opened_water_delta_t_k", 0.0))

    if not shifts and opened_dt == 0.0:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cp(src, dst)
        log.info(f"tgrad sst: unperturbed copy {src} -> {dst}")
        return

    lat, lon = read_grid(src)
    ice_src = am_dir / ICE_FILE
    if ice_src.exists():
        idx = nearest_index(*read_grid(ice_src), lat, lon)
        frozen, opened = ice_masks_on_sst_grid(ice_src, ice_fn, idx)
        del idx
    else:
        log.warning(f"tgrad sst: {ice_src} missing; no ice masks")
        frozen, opened = np.zeros(lat.shape, dtype=bool), {}

    active = None
    if "uniform_shift" in shifts:
        active = ocean_on_sst_grid(am_dir / LAND_FILE, lat, lon) & ~frozen

    scale = monthly_scale(p)
    uniform = float(p.get("uniform_warming_k", 0.0)) if shifts else 0.0

    def sst_fn(vals: np.ndarray, lat: np.ndarray, month: int) -> np.ndarray:
        d_t = np.zeros_like(vals)
        if shifts:
            d_t = uniform + shift_methods[shifts[0]](lat, month, p, active, scale)
        d_t = np.where(frozen, 0.0, d_t) + opened_dt * opened.get(month, False)
        return np.maximum(vals + d_t, np.minimum(vals, T_FREEZE))

    d_min, d_max = perturb_grib(src, dst, sst_fn)
    log.info(
        f"tgrad sst: perturbed {src} -> {dst}; change range "
        f"[{d_min:.3f}, {d_max:.3f}] K; {int(frozen.sum())} perennial-ice or "
        "land-flagged points unchanged"
    )


def apply_tgrad_perturbations() -> bool:
    """Write the FIXED SST and sea-ice climatologies from pristine fix_src/am
    according to the `tgrad_perturbations` config in the state.

    Both files are rewritten under their original names, perturbed or not, so a
    case switched back to a control arm cannot keep a stale perturbed copy.
    fix_src is only read. Returns True when the files were written.
    """

    perturbations = state.tgrad_perturbations
    if not perturbations:
        return False

    if not isinstance(perturbations, dict):
        raise TypeError("`tgrad_perturbations` config must be a mapping")

    p = dict(perturbations)  # state copy stays as configured (checksum)

    allowed = ("none", "polar_shift", "uniform_shift", "ice_loss")
    required = ("method",)
    missing = [k for k in required if k not in p]
    if missing:
        raise KeyError(f"Missing tgrad_perturbations keys: {missing}")

    soft_keys = (
        "polar_delta_t_k",
        "sic_reduction",
        "poles",
        "lat_start",
        "lat_full",
        "ocean_mode",
        "uniform_warming_k",
        "monthly_scale",
        "opened_water_delta_t_k",
    )

    for k in p:
        if k not in required and k not in soft_keys:
            raise ValueError(f"Unknown key in tgrad_perturbations: {k}")

    methods = p.get("method")

    if isinstance(methods, str):
        check_methods = [methods]
    else:
        check_methods = list(methods)

    for m in check_methods:
        if m not in allowed:
            raise ValueError(f"`method` must be one of {allowed}. Got `{m}`")

    # Conditional parameter checks

    if "polar_shift" in check_methods and "polar_delta_t_k" not in p:
        raise KeyError(
            "If method includes 'polar_shift', you must provide key 'polar_delta_t_k'"
        )

    if "uniform_shift" in check_methods and "polar_delta_t_k" not in p:
        raise KeyError(
            "If method includes 'uniform_shift', you must provide key 'polar_delta_t_k'"
        )

    if "ice_loss" in check_methods and "sic_reduction" not in p:
        raise KeyError(
            "If method includes 'ice_loss', you must provide key 'sic_reduction'"
        )

    if "none" in check_methods and len(check_methods) > 1:
        raise ValueError("Method 'none' cannot be combined with other methods")

    if "polar_shift" in check_methods and "uniform_shift" in check_methods:
        raise ValueError("Use only one of 'polar_shift' and 'uniform_shift'")

    p.setdefault("poles", "north")
    p.setdefault("lat_start", 60.0)
    p.setdefault("lat_full", 80.0)
    p.setdefault("ocean_mode", "prescribed")

    if p["poles"] not in POLES:
        raise ValueError(f"`poles` must be one of {tuple(POLES)}")

    if p["lat_full"] <= p["lat_start"]:
        raise ValueError("`lat_full` must exceed `lat_start`")

    if p["ocean_mode"] not in ("prescribed", "mlo"):
        raise ValueError("`ocean_mode` must be one of ('prescribed', 'mlo')")

    monthly_scale(p)
    p["dt_poles"] = per_pole(
        p.get("polar_delta_t_k", 0.0), p["poles"], "polar_delta_t_k"
    )
    p["sic_poles"] = per_pole(p.get("sic_reduction", 0.0), p["poles"], "sic_reduction")

    for pole, dt in p["dt_poles"].items():
        if abs(dt) > DELTA_T_MAX:
            raise ValueError(
                f"polar_delta_t_k ({pole}) = {dt} K exceeds the CMIP6 range of "
                f"central-Arctic SST warming (|dT| <= {DELTA_T_MAX} K)."
            )

    for pole, r in p["sic_poles"].items():
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"sic_reduction ({pole}) = {r} must be within 0..1.")

    # update_namsfc runs once per domain (global and nests); write once.
    key = f"{state.fix}|{p!r}"
    if key in _WRITTEN:
        return True

    log.info(f"tgrad_perturbations detected; applying {check_methods}")

    am_dir = Path(state.fix_src) / "am"
    ice_fn = adjust_sea_ice(p, check_methods, am_dir)
    adjust_sst(p, check_methods, am_dir, ice_fn)

    _WRITTEN.add(key)

    log.info("Finished applying tgrad perturbations")

    return True
