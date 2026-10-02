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

# method -> key it requires. SST and sea-ice files are always both rewritten.
METHODS = {
    "none": None,
    "polar_shift": "polar_delta_t_k",
    "uniform_shift": "polar_delta_t_k",
    "ice_loss": "sic_reduction",
}
REQUIRED = ("method",)
SOFT_KEYS = (
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
DEFAULTS = {
    "poles": "north",
    "lat_start": 60.0,
    "lat_full": 80.0,
    "ocean_mode": "prescribed",
    "uniform_warming_k": 0.0,
    "opened_water_delta_t_k": 0.0,
}
POLES = {"north": ("north",), "south": ("south",), "both": ("north", "south")}
OCEAN_MODES = ("prescribed", "mlo")

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

Transform = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
_WRITTEN: set[tuple[str, str]] = set()  # configurations already written this process


def as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


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


def read_config() -> dict | None:
    """Validated tgrad_perturbations with defaults filled, or None if unset.

    Adds the per-pole amplitudes dt_poles and sic_poles and the flags sst_on,
    ice_on and matched (uniform_shift).
    """
    p = state.tgrad_perturbations
    if p is None:
        return None
    if not isinstance(p, dict):
        raise TypeError("`tgrad_perturbations` config must be a mapping")

    missing = [k for k in REQUIRED if k not in p]
    if missing:
        raise KeyError(f"Missing tgrad_perturbations keys: {missing}")
    for k in p:
        if k not in REQUIRED and k not in SOFT_KEYS:
            raise ValueError(f"Unknown key in tgrad_perturbations: {k}")

    methods = as_list(p["method"])
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"`method` must be one of {tuple(METHODS)}. Got `{m}`")
        if METHODS[m] is not None and METHODS[m] not in p:
            raise KeyError(
                f"If method includes '{m}', you must provide key '{METHODS[m]}'"
            )
    if "none" in methods and len(methods) > 1:
        raise ValueError("Method 'none' cannot be combined with other methods")
    if {"polar_shift", "uniform_shift"} <= set(methods):
        raise ValueError("Use only one of 'polar_shift' and 'uniform_shift'")

    cfg = {**DEFAULTS, **p}
    if cfg["poles"] not in POLES:
        raise ValueError(f"`poles` must be one of {tuple(POLES)}")
    if cfg["lat_full"] <= cfg["lat_start"]:
        raise ValueError("`lat_full` must exceed `lat_start`")
    if cfg["ocean_mode"] not in OCEAN_MODES:
        raise ValueError(f"`ocean_mode` must be one of {OCEAN_MODES}")

    cfg["sst_on"] = bool({"polar_shift", "uniform_shift"} & set(methods))
    cfg["ice_on"] = "ice_loss" in methods
    cfg["matched"] = "uniform_shift" in methods
    cfg["dt_poles"] = per_pole(
        cfg.get("polar_delta_t_k", 0.0), cfg["poles"], "polar_delta_t_k"
    )
    cfg["sic_poles"] = per_pole(
        cfg.get("sic_reduction", 0.0), cfg["poles"], "sic_reduction"
    )

    for pole, dt in cfg["dt_poles"].items():
        if abs(dt) > DELTA_T_MAX:
            raise ValueError(
                f"polar_delta_t_k ({pole}) = {dt} K exceeds the CMIP6 range of "
                f"central-Arctic SST warming (|dT| <= {DELTA_T_MAX} K)."
            )
    for pole, r in cfg["sic_poles"].items():
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"sic_reduction ({pole}) = {r} must be within 0..1.")
    return cfg


def magnitude_notes(cfg: dict) -> None:
    """Log where the chosen amplitudes sit relative to the literature."""
    if cfg["sst_on"]:
        for pole, dt in cfg["dt_poles"].items():
            if abs(dt) > DELTA_T_STANDARD:
                log.warning(
                    "tgrad: polar_delta_t_k (%s) = %s K is a strong-forcing amplitude "
                    "(standard sensitivity amplitudes are 2 and 4 K).",
                    pole,
                    dt,
                )
            if dt < 0.0:
                log.warning(
                    "tgrad: polar_delta_t_k (%s) = %s K cools the pole and "
                    "strengthens the gradient.",
                    pole,
                    dt,
                )
            if pole == "south" and dt > 0.0:
                log.warning(
                    "tgrad: projected Southern Ocean surface warming is delayed "
                    "and weaker than Arctic warming; treat a southern polar "
                    "warming as an idealized sensitivity test."
                )
    if cfg["ice_on"] and cfg.get("monthly_scale") is None:
        for pole, r in cfg["sic_poles"].items():
            if r > 0.5:
                log.warning(
                    "tgrad: sic_reduction (%s) = %s applied in every month; projected "
                    "ice loss is largest in late summer, so winter loss this large is "
                    "idealized. Consider monthly_scale.",
                    pole,
                    r,
                )


def smoothstep_weight(phi: np.ndarray, cfg: dict) -> np.ndarray:
    """W = x^2 (3 - 2x), x = clip((phi - lat_start) / (lat_full - lat_start), 0, 1)."""
    span = cfg["lat_full"] - cfg["lat_start"]
    x = np.clip((phi - cfg["lat_start"]) / span, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def polar_pattern(lat: np.ndarray, cfg: dict, amps: dict[str, float]) -> np.ndarray:
    """Sum over poles of amplitude * W, with phi = lat (north) or -lat (south)."""
    out = np.zeros_like(lat, dtype=float)
    for pole, amp in amps.items():
        out += amp * smoothstep_weight(lat if pole == "north" else -lat, cfg)
    return out


def monthly_scale(cfg: dict) -> list[float]:
    scale = cfg.get("monthly_scale") or [1.0] * 12
    if len(scale) != 12:
        raise ValueError("monthly_scale must have 12 values (Jan..Dec).")
    return [float(a) for a in scale]


def message_month(gid: int) -> int:
    return (eccodes.codes_get(gid, "dataDate") // 100) % 100


def read_grib(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {month: (values, latitudes, longitudes)} for a monthly GRIB file.

    Missing (bitmapped) points are returned as NaN.
    """
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


def unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack(
        (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))
    )


def nearest_index(
    src_lat: np.ndarray, src_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Index of the nearest source point (on the sphere) for each target point."""
    _, idx = cKDTree(unit_vectors(src_lat, src_lon)).query(unit_vectors(lat, lon))
    return idx


def perturb_grib(src: Path, dst: Path, transform: Transform) -> tuple[float, float]:
    """Rewrite each GRIB message with transformed values, keeping all headers.

    Returns the minimum and maximum change over all messages.
    """
    d_min, d_max = 0.0, 0.0
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


def ice_transform(cfg: dict) -> Transform:
    """SIC' = SIC * (1 - min(1, A_m * sum_pole sic_reduction * W(phi)))."""
    scale = monthly_scale(cfg)

    def _apply(vals: np.ndarray, lat: np.ndarray, month: int) -> np.ndarray:
        pattern = polar_pattern(lat, cfg, cfg["sic_poles"])
        loss = np.clip(scale[month - 1] * pattern, 0.0, 1.0)
        # Values outside [0, 1] are land flags (1.57 in the CFSR file); keep them.
        fraction = (vals >= 0.0) & (vals <= 1.0)
        return np.where(fraction, np.clip(vals * (1.0 - loss), 0.0, 1.0), vals)

    return _apply


def ice_masks_on_sst_grid(
    ice_src: Path, ice_fn: Transform | None, idx: np.ndarray
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Perennial-ice mask and per-month opened-water masks on the SST grid.

    Computed on the ice grid and mapped with the nearest-neighbour index idx.
    frozen: final SIC >= 0.15 in every month (land flags count as frozen).
    opened[m]: ice in the original field (0.15 <= SIC <= 1) but open water in
    the final field (SIC' < 0.15).
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
    log.info("tgrad sst: ocean fraction from %s = %.3f", land_src, frac)
    if not OCEAN_FRACTION[0] <= frac <= OCEAN_FRACTION[1]:
        raise ValueError(f"Ocean fraction {frac:.3f} from {land_src} is implausible.")
    return ocean


def sst_transform(
    cfg: dict,
    sst_on: bool,
    frozen: np.ndarray,
    opened: dict[int, np.ndarray],
    active: np.ndarray | None,
) -> Transform:
    """dT = uniform_warming_k + A_m * sum_pole polar_delta_t_k * W(phi)   [K]
    plus opened_water_delta_t_k where the ice reduction opens water.

    With active (method uniform_shift), the polar pattern is replaced by its monthly
    mean over the SST-active ocean (cos(lat) area weights; land and perennial
    ice excluded). dT is zero at perennial-ice points. Cooling never takes SST
    below T_FREEZE (points already colder are left unchanged).
    """
    scale = monthly_scale(cfg)
    uniform = float(cfg["uniform_warming_k"]) if sst_on else 0.0
    opened_dt = float(cfg["opened_water_delta_t_k"])

    def _apply(vals: np.ndarray, lat: np.ndarray, month: int) -> np.ndarray:
        d_t = np.zeros_like(vals)
        if sst_on:
            pattern = polar_pattern(lat, cfg, cfg["dt_poles"])
            if active is not None:
                cosw = np.cos(np.deg2rad(lat)) * active
                pattern = np.full_like(pattern, np.sum(pattern * cosw) / np.sum(cosw))
            d_t = uniform + scale[month - 1] * pattern
        d_t = np.where(frozen, 0.0, d_t) + opened_dt * opened.get(month, False)
        return np.maximum(vals + d_t, np.minimum(vals, T_FREEZE))

    return _apply


def tgrad_namelist(nml: dict) -> dict:
    """Surface settings shared by all arms (CTRL, method none, included).

    Applied whenever tgrad_perturbations is set, before case overrides.
    namsfc ftsfs = faiss = fsics = fsicl = 0: SST, ice mask and ice
    concentration follow the climatologies at every surface cycle. fsicl
    matters because concentration at points already classified as ice passes
    through the non-open-ocean branch of sfcsub merge.
    ocean_mode prescribed (default): do_ocean = use_ext_sst = false and NSST
    off, so SST is the climatology itself. ocean_mode mlo: restored
    mixed-layer ocean (do_ocean = true, restore_method = 1); the restoring
    timescale and latitude limits stay as set in ocean_nml.
    """
    cfg = read_config()
    if cfg is None:
        return nml
    nml["namsfc"].update({"ftsfs": 0, "faiss": 0, "fsics": 0, "fsicl": 0})
    phys = nml.setdefault("gfs_physics_nml", {})
    phys["use_ext_sst"] = False
    if cfg["ocean_mode"] == "prescribed":
        phys["do_ocean"] = False
        phys["nstf_name"] = [
            0,
            0,
            1,
            0,
            5,
        ]  # SHiELD default; nstf_name(1) = 0: NSST off
    else:
        phys["do_ocean"] = True
        nml.setdefault("ocean_nml", {})["restore_method"] = 1
    return nml


def apply_tgrad_perturbations() -> None:
    """Write the FIXED SST/ice climatologies from pristine fix_src.

    tgrad_perturbations: null leaves the original staging untouched. Otherwise
    both files are rewritten under their original names in FIXED/, perturbed
    or pristine, so a case switched from a perturbed arm back to CTRL cannot
    retain a stale perturbed copy. fix_src is only read.
    """
    cfg = read_config()
    if cfg is None:
        return
    key = (str(state.fix), repr(cfg))
    if key in _WRITTEN:  # update_namsfc runs once per domain (global and nests)
        return
    magnitude_notes(cfg)

    sst_on, ice_on, matched = cfg["sst_on"], cfg["ice_on"], cfg["matched"]
    am_dir = Path(state.fix_src) / "am"
    ice_fn = ice_transform(cfg) if ice_on else None
    opened_on = ice_on and float(cfg["opened_water_delta_t_k"]) != 0.0
    log.info("tgrad_perturbations detected; applying %s", state.tgrad_perturbations)

    src, dst = stage(am_dir / ICE_FILE, Path(state.fix) / ICE_FILE)
    if src is not None:
        if ice_fn is None:
            copy_logged("sea_ice", src, dst)
        else:
            d_min, d_max = perturb_grib(src, dst, ice_fn)
            log.info(
                "tgrad sea_ice: perturbed %s -> %s; change range [%.3f, %.3f]",
                src,
                dst,
                d_min,
                d_max,
            )

    src, dst = stage(am_dir / SST_FILE, Path(state.fix) / SST_FILE)
    if src is not None and not (sst_on or opened_on):
        copy_logged("sst", src, dst)
    elif src is not None:
        lat, lon = read_grid(src)
        ice_src = am_dir / ICE_FILE
        if ice_src.exists():
            idx = nearest_index(*read_grid(ice_src), lat, lon)
            frozen, opened = ice_masks_on_sst_grid(ice_src, ice_fn, idx)
            del idx
        else:
            log.warning("tgrad sst: %s missing; no ice masks", ice_src)
            frozen, opened = np.zeros(lat.shape, dtype=bool), {}

        active = None
        if matched:
            active = ocean_on_sst_grid(am_dir / LAND_FILE, lat, lon) & ~frozen

        transform = sst_transform(cfg, sst_on, frozen, opened, active)
        d_min, d_max = perturb_grib(src, dst, transform)
        log.info(
            "tgrad sst: perturbed %s -> %s; change range [%.3f, %.3f] K; "
            "%d perennial-ice or land-flagged points unchanged",
            src,
            dst,
            d_min,
            d_max,
            int(frozen.sum()),
        )
    _WRITTEN.add(key)


def stage(src: Path, dst: Path) -> tuple[Path | None, Path]:
    """Prepare dst for rewriting; return (None, dst) if src is missing."""
    if not src.exists():  # already reported by update_namsfc
        return None, dst
    dst.unlink(missing_ok=True)  # never write through a symlink into fix_src
    return src, dst


def copy_logged(name: str, src: Path, dst: Path) -> None:
    cp(src, dst)
    log.info("tgrad %s: unperturbed copy %s -> %s", name, src, dst)
