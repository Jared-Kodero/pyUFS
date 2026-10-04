from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr
from fv3_runtime import log
from fv3_state import state
from fv3_utils import cp
from grib_io import open_grib, save_grib
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


def polar_weight(lat: np.ndarray, p: dict, amps: dict[str, float]) -> np.ndarray:
    """Sum over poles of amplitude times a smoothstep ramp in latitude.

    Parameters
    ----------
    lat : numpy.ndarray
        Latitude [degrees north].
    p : dict
        Validated parameters with ``lat_start`` and ``lat_full`` [degrees].
    amps : dict of str to float
        Amplitude per pole.

    Returns
    -------
    numpy.ndarray
        :math:`\\sum_{pole} a_{pole} W(\\phi)`, with :math:`\\phi` = `lat`
        for the north and -`lat` for the south.

    Notes
    -----
    .. math:: W = x^2 (3 - 2x), \\quad
              x = \\mathrm{clip}\\left(\\frac{\\phi - \\phi_s}{\\phi_f - \\phi_s}, 0, 1\\right)
    """
    span = p["lat_full"] - p["lat_start"]
    out = np.zeros_like(lat, dtype=float)
    for pole, amp in amps.items():
        phi = lat if pole == "north" else -lat
        x = np.clip((phi - p["lat_start"]) / span, 0.0, 1.0)
        out += amp * (x * x * (3.0 - 2.0 * x))
    return out


def nearest_index(
    src_lat: np.ndarray, src_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Nearest source point on the sphere for each target point.

    Parameters
    ----------
    src_lat, src_lon : numpy.ndarray
        Source grid coordinates [degrees].
    lat, lon : numpy.ndarray
        Target grid coordinates [degrees].

    Returns
    -------
    numpy.ndarray
        Index into the source points for every target point.
    """

    def xyz(la: np.ndarray, lo: np.ndarray) -> np.ndarray:
        phi, lam = np.deg2rad(la), np.deg2rad(lo)
        return np.column_stack(
            (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))
        )

    _, idx = cKDTree(xyz(src_lat, src_lon)).query(xyz(lat, lon))
    return idx


def perturb_grib(src: Path, dst: Path, transform: Transform) -> tuple[float, float]:
    """Rewrite every monthly field with transformed values, keeping all headers.

    Parameters
    ----------
    src : Path
        Source GRIB climatology with one variable on one level.
    dst : Path
        Output GRIB file.
    transform : callable
        ``transform(values, lat, month) -> values`` on flattened fields.

    Returns
    -------
    d_min, d_max : float
        Smallest and largest change over all months; 0 when nothing changes.

    Notes
    -----
    Missing (NaN) points stay missing. Headers and packing come from `src`
    through :func:`grib_io.save_grib`.
    """
    tree = open_grib(src)
    node = tree.leaves[0]
    name = next(iter(node.data_vars))
    field = node[name]
    lat = xr.broadcast(field.latitude, field.longitude)[0].values.ravel()
    data = np.empty(field.shape, dtype=field.dtype)
    d_min, d_max = 0.0, 0.0
    for i, month in enumerate(field.time.dt.month.values):
        old = field[i, 0].values.ravel().astype(float)
        new = np.where(np.isnan(old), np.nan, transform(old, lat, int(month)))
        change = new - old
        if np.isfinite(change).any():
            d_min = min(d_min, float(np.nanmin(change)))
            d_max = max(d_max, float(np.nanmax(change)))
        data[i, 0] = new.reshape(field.shape[2:])
    node[name] = field.copy(data=data)
    save_grib(tree, dst)
    return d_min, d_max


def adjust_sea_ice(p: dict, methods: list, am_dir: Path) -> Transform | None:
    """Write the FIXED sea-ice climatology.

    Parameters
    ----------
    p : dict
        Validated perturbation parameters.
    methods : list of str
        Selected methods.
    am_dir : Path
        Pristine ``fix_src/am`` directory.

    Returns
    -------
    callable or None
        The sea-ice transform when ``ice_loss`` is applied, else None.

    Raises
    ------
    FileNotFoundError
        If the source climatology is missing.

    Notes
    -----
    .. math:: SIC' = SIC \\left(1 - \\min\\left(1, A_m \\sum_{pole} r_{pole} W(\\phi)\\right)\\right)

    Values outside [0, 1] are land flags (1.57 in the CFSR file) and are kept.
    """
    src = am_dir / ICE_FILE
    dst = Path(state.fix) / "MODS" / ICE_FILE

    if not src.exists():
        raise FileNotFoundError(f"Sea ice climatology not found: {src}")

    dst.unlink(missing_ok=True)  # never write through a symlink into fix_src

    if "ice_loss" not in methods:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cp(src, dst)
        log.info(f"Copied unperturbed sea ice to: {dst}")
        return None

    scale = p["monthly_scale"]

    def ice_fn(vals: np.ndarray, lat: np.ndarray, month: int) -> np.ndarray:
        weight = scale[month - 1] * polar_weight(lat, p, p["sic_poles"])
        loss = np.clip(weight, 0.0, 1.0)
        fraction = (vals >= 0.0) & (vals <= 1.0)
        return np.where(fraction, np.clip(vals * (1.0 - loss), 0.0, 1.0), vals)

    d_min, d_max = perturb_grib(src, dst, ice_fn)
    log.info(
        f"Applied ice_loss with sic_reduction={p['sic_reduction']}: "
        + f"dSIC [{d_min:.3f}, {d_max:.3f}]"
    )
    return ice_fn


def adjust_sst(p: dict, methods: list, am_dir: Path, ice_fn: Transform | None) -> None:
    """Write the FIXED SST climatology.

    Parameters
    ----------
    p : dict
        Validated perturbation parameters.
    methods : list of str
        Selected methods.
    am_dir : Path
        Pristine ``fix_src/am`` directory.
    ice_fn : callable or None
        Sea-ice transform from :func:`adjust_sea_ice`.

    Raises
    ------
    FileNotFoundError
        If the source climatology is missing.
    ValueError
        If the land mask gives an implausible ocean fraction.

    Notes
    -----
    dT = ``uniform_warming_k`` + the shift method, plus
    ``opened_water_delta_t_k`` where ``ice_loss`` opens water. dT is zero
    where the final SIC >= ``ICE_EDGE`` in every month (perennial ice and
    land flags), and cooling never takes SST below ``T_FREEZE``.

    polar_shift:

    .. math:: \\Delta T = A_m \\sum_{pole} \\Delta T_{pole} W(\\phi)

    uniform_shift replaces that pattern by its cos(lat)-weighted mean over
    ocean points that are not perennial ice.
    """
    src = am_dir / SST_FILE
    dst = Path(state.fix) / "MODS" / SST_FILE

    if not src.exists():
        raise FileNotFoundError(f"SST climatology not found: {src}")

    dst.unlink(missing_ok=True)  # never write through a symlink into fix_src

    shifts = [m for m in methods if m in ("polar_shift", "uniform_shift")]
    opened_dt = 0.0
    if "ice_loss" in methods:
        opened_dt = float(p.get("opened_water_delta_t_k", 0.0))

    if not shifts and opened_dt == 0.0:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cp(src, dst)
        log.info(f"Copied unperturbed SST to: {dst}")
        return

    sst = open_grib(src).leaves[0]
    sst = sst[next(iter(sst.data_vars))]
    lat, lon = (c.values.ravel() for c in xr.broadcast(sst.latitude, sst.longitude))

    ice_src = am_dir / ICE_FILE
    opened = {}
    if ice_src.exists():
        ice = open_grib(ice_src).leaves[0]
        ice = ice[next(iter(ice.data_vars))]
        ice_lat, ice_lon = (
            c.values.ravel() for c in xr.broadcast(ice.latitude, ice.longitude)
        )
        idx = nearest_index(ice_lat, ice_lon, lat, lon)
        frozen = None
        for i, month in enumerate(ice.time.dt.month.values):
            sic0 = np.nan_to_num(ice[i, 0].values.ravel().astype(float), nan=0.0)
            sic1 = sic0 if ice_fn is None else ice_fn(sic0, ice_lat, int(month))
            ice1 = sic1 >= ICE_EDGE
            frozen = ice1 if frozen is None else frozen & ice1
            opened[int(month)] = ((sic0 >= ICE_EDGE) & (sic0 <= 1.0) & ~ice1)[idx]
        frozen = frozen[idx]
        del idx
    else:
        log.warning(f"Sea ice climatology not found: {ice_src}; no perennial-ice mask")
        frozen = np.zeros(lat.shape, dtype=bool)

    scale = p["monthly_scale"]
    uniform = float(p.get("uniform_warming_k", 0.0)) if shifts else 0.0
    pattern = polar_weight(lat, p, p["dt_poles"])

    if "uniform_shift" in shifts:
        land = open_grib(am_dir / LAND_FILE).leaves[0]
        land = land[next(iter(land.data_vars))]
        land_lat, land_lon = (
            c.values.ravel() for c in xr.broadcast(land.latitude, land.longitude)
        )
        is_land = np.nan_to_num(land[0, 0].values.ravel().astype(float), nan=1.0) >= 0.5
        ocean = ~is_land[nearest_index(land_lat, land_lon, lat, lon)]
        cosw = np.cos(np.deg2rad(lat))
        frac = float(np.sum(cosw * ocean) / np.sum(cosw))
        log.info(f"Ocean fraction: {frac:.3f}")
        if not OCEAN_FRACTION[0] <= frac <= OCEAN_FRACTION[1]:
            raise ValueError(
                f"Ocean fraction {frac:.3f} outside {OCEAN_FRACTION}; "
                + f"check {am_dir / LAND_FILE}"
            )
        cosw = cosw * (ocean & ~frozen)
        pattern = np.full_like(pattern, np.sum(pattern * cosw) / np.sum(cosw))

    def sst_fn(vals: np.ndarray, lat: np.ndarray, month: int) -> np.ndarray:
        d_t = np.zeros_like(vals)
        if shifts:
            d_t = uniform + scale[month - 1] * pattern
        d_t = np.where(frozen, 0.0, d_t) + opened_dt * opened.get(month, False)
        return np.maximum(vals + d_t, np.minimum(vals, T_FREEZE))

    d_min, d_max = perturb_grib(src, dst, sst_fn)
    applied = shifts + (["opened_water"] if opened_dt else [])
    log.info(
        f"Applied {', '.join(applied)} to SST: dT [{d_min:.2f}, {d_max:.2f}] K, "
        + f"{int(frozen.sum())} ice/land points held"
    )


def validate_tgrad_perturbations(perturbations: object) -> tuple[dict, list] | None:
    """Validate ``tgrad_perturbations`` before any file is written.

    Parameters
    ----------
    perturbations : object
        The ``tgrad_perturbations`` block from the state; empty or None
        disables the perturbation.

    Returns
    -------
    tuple of (dict, list) or None
        Parameters with defaults, ``monthly_scale`` as twelve factors and
        per-pole amplitudes ``dt_poles`` and ``sic_poles``, and the selected
        methods; None when no block is configured.

    Raises
    ------
    KeyError
        If ``method`` or a key required by a method is missing.
    TypeError
        If the block is not a mapping.
    ValueError
        If a key, method or value is invalid.

    Notes
    -----
    A scalar amplitude applies to every pole selected by ``poles``; a mapping
    ``{north: x, south: y}`` sets each pole explicitly. The ``DELTA_T_MAX``
    limit applies to the base ``polar_delta_t_k``, before ``monthly_scale``.
    """
    if not perturbations:
        return None

    if not isinstance(perturbations, dict):
        raise TypeError("`tgrad_perturbations` config must be a mapping")

    p = dict(perturbations)  # state copy stays as configured (checksum)

    allowed = ("none", "polar_shift", "uniform_shift", "ice_loss")
    keys = (
        "method",
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

    if "method" not in p:
        raise KeyError("Missing perturbation key: method")

    unknown = [k for k in p if k not in keys]
    if unknown:
        raise ValueError(f"Unknown keys in perturbation config: {unknown}")

    methods = p["method"]
    if isinstance(methods, str):
        methods = [methods]
    if not isinstance(methods, (list, tuple)) or not methods:
        raise ValueError("`method` must be a name or a non-empty list; use `none`")
    methods = list(methods)

    for m in methods:
        if m not in allowed:
            raise ValueError(f"`method` must be one of {allowed}. Got `{m}`")

    needs = {
        "polar_shift": "polar_delta_t_k",
        "uniform_shift": "polar_delta_t_k",
        "ice_loss": "sic_reduction",
    }
    for m in methods:
        if m in needs and needs[m] not in p:
            raise KeyError(f"If method includes '{m}', you must provide '{needs[m]}'")

    if "none" in methods and len(methods) > 1:
        raise ValueError("`none` cannot be combined with other methods")

    if "polar_shift" in methods and "uniform_shift" in methods:
        raise ValueError("only one of `polar_shift` and `uniform_shift` can be used")

    p.setdefault("poles", "north")
    p["lat_start"] = float(p.get("lat_start", 60.0))
    p["lat_full"] = float(p.get("lat_full", 80.0))
    p.setdefault("ocean_mode", "prescribed")

    if p["poles"] not in POLES:
        raise ValueError(f"`poles` must be one of {tuple(POLES)}. Got `{p['poles']}`")

    if not 0.0 <= p["lat_start"] < p["lat_full"] <= 90.0:
        raise ValueError("`lat_start` and `lat_full` need 0 <= start < full <= 90")

    if p["ocean_mode"] not in ("prescribed", "mlo"):
        raise ValueError(
            f"`ocean_mode` must be prescribed or mlo. Got `{p['ocean_mode']}`"
        )

    for k in ("uniform_warming_k", "opened_water_delta_t_k"):
        if k in p:
            p[k] = float(p[k])

    scale = p.get("monthly_scale")
    if scale is None:
        scale = [1.0] * 12
    if not isinstance(scale, (list, tuple)) or len(scale) != 12:
        raise ValueError("`monthly_scale` needs 12 values (Jan..Dec)")
    p["monthly_scale"] = [float(a) for a in scale]

    for key, out in (("polar_delta_t_k", "dt_poles"), ("sic_reduction", "sic_poles")):
        value = p.get(key, 0.0)
        if isinstance(value, dict):
            bad = [k for k in value if k not in ("north", "south")]
            if bad:
                raise ValueError(f"`{key}` keys must be north or south. Got {bad}")
            p[out] = {k: float(v) for k, v in value.items()}
        else:
            p[out] = {pole: float(value) for pole in POLES[p["poles"]]}

    for pole, dt in p["dt_poles"].items():
        if abs(dt) > DELTA_T_MAX:
            raise ValueError(
                f"`polar_delta_t_k` must be within +/-{DELTA_T_MAX} K. "
                + f"Got {dt} K ({pole})"
            )

    for pole, r in p["sic_poles"].items():
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"`sic_reduction` must be within 0..1. Got {r} ({pole})")

    return p, methods


def apply_tgrad_perturbations() -> bool:
    """Write the FIXED SST and sea-ice climatologies from pristine ``fix_src/am``.

    Uses the ``tgrad_perturbations`` config in the state. Both files are
    rewritten under their original names, perturbed or not, so a case switched
    back to a control arm cannot keep a stale perturbed copy. ``fix_src`` is
    only read.

    Returns
    -------
    bool
        True when the files are written (or were already written in this
        process for the same config), False when no perturbation is configured.
    """

    validated = validate_tgrad_perturbations(state.tgrad_perturbations)
    if validated is None:
        return False

    p, methods = validated

    # update_namsfc runs once per domain (global and nests); write once.
    key = f"{state.fix}|{p!r}"
    if key in _WRITTEN:
        return True

    am_dir = Path(state.fix_src) / "am"
    scale = p["monthly_scale"]

    log.info(f"tgrad_perturbations detected; applying {', '.join(methods)}")
    log.info(f"Using climatologies from: {am_dir}")
    if any(a != 1.0 for a in scale):
        log.info(f"Using monthly_scale={scale}")

    if {"polar_shift", "uniform_shift"} & set(methods):
        for pole, dt in p["dt_poles"].items():
            if abs(dt) > DELTA_T_STANDARD:
                log.info(f"Strong forcing: polar_delta_t_k={dt} K ({pole})")
            if pole == "south" and dt > 0.0:
                log.info(f"Idealized: southern warming of {dt} K")
            peak = max(abs(a) for a in scale) * abs(dt)
            if peak > DELTA_T_MAX:
                log.warning(f"Scaled anomaly {peak} K exceeds {DELTA_T_MAX} K ({pole})")
    if "ice_loss" in methods:
        for pole, r in p["sic_poles"].items():
            if min(min(max(a * r, 0.0), 1.0) for a in scale) > 0.5:
                log.info(f"Idealized: ice loss above 50% in every month ({pole})")

    ice_fn = adjust_sea_ice(p, methods, am_dir)
    adjust_sst(p, methods, am_dir, ice_fn)

    _WRITTEN.add(key)

    log.info("Finished applying tgrad perturbations")

    return True
