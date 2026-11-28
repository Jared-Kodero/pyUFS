"""
ECMWF (IFS/ERA5) initial conditions written into a GFS GRIB2 template.

``chgres_cube`` reads GFS ``pgrb2.0p25`` files through fixed NCEP GRIB2 codes.
For a GFS cycle and lead, this tool downloads the GFS file (or takes it with
``--gfs``) and the ERA5 analysis valid at the same time (137 model levels plus
surface fields), converts the ECMWF state to the variables, units, levels and
masks of the GFS file, and writes a new GFS-format file with
``grib_io.save_grib`` using the GFS file as template. List the result under
``GFS`` in ``ic_grib_files`` of run_config.yaml; ``chgres_cube`` reads it as a
GFS file (``external_model: GFS``), so no ``chgres_cube`` setting changes.

Usage::

    python tools/ecmwf_ic.py --cycle 2000060100 --fh 0 --out /data/era5_ic

writes ``/data/era5_ic/GFS.2000060100Z.f000.0p25deg.ECMWF.grib2``; the GFS
template and the ERA5 downloads are kept in ``/data/era5_ic/download``. The
ERA5 model levels come from the MARS tape archive, so the request can queue
for hours; run the tool before the case, not inside it. Requires a CDS key in
``~/.cdsapirc``.

Atmosphere (GFS isobaric levels 1 Pa to 1000 hPa)
    ECMWF model-level fields are interpolated linearly in ln p from full-level
    pressures ``p_k = (p_{k-1/2} + p_{k+1/2}) / 2`` with
    ``p_{k+1/2} = a_{k+1/2} + b_{k+1/2} p_s``; the ``a, b`` coefficients come
    from the ``pv`` array of the GRIB messages and ``p_s = exp(lnsp)``. Above
    the top full level the top value is kept. Below the lowest full level
    temperature and geopotential use the ECMWF extrapolation of Trenberth,
    Berry and Buja (1993, NCAR/TN-396+STR) as coded in NCL
    ``vinth2p_ecmwf``; every other variable keeps its lowest-level value.
    Geopotential is integrated hydrostatically from the surface geopotential
    to the model half levels (Simmons and Burridge 1981 scheme) with virtual
    temperature ``T (1 + (Rv/Rd - 1) q)``, then inside the model layer that
    holds each pressure level, ``Phi(p) = Phi(k+1/2) + Rd Tv_k ln(p(k+1/2)/p)``;
    below the surface it uses the Trenberth et al. formula. (Linear ln p
    interpolation of full-level geopotential is biased by up to 11 m at 1 hPa
    against ERA5 pressure-level ``z``.)

    Derived fields: ``gh = Phi / g``; ``wz`` (DZDT, read by ``chgres_cube``)
    ``= -omega Rd Tv / (g p)``; ``tcc = 100 cc``; graupel is set to zero
    (ERA5 has no graupel). Hydrometeors are ECMWF specific contents, which is
    what the GFS mixing-ratio records hold. ``r`` and ``absv`` are dropped:
    ``chgres_cube`` does not read them and they would describe the GFS state.

Surface
    The GFS land mask, soil type, vegetation and sea-ice thickness are kept,
    because ``chgres_cube`` uses the input soil type to rescale soil moisture.
    ECMWF fields are placed on that mask: land values come from the nearest
    ECMWF land point (``lsm >= 0.5``), sea values from the nearest ECMWF sea
    point. Sea ice is ECMWF ``ci`` on GFS water; ``chgres_cube`` flags ice
    where ``ci > 0.15``.

    - ``sp = exp(lnsp)``, ``orog = z_s / g`` (consistent with the atmosphere).
    - Skin temperature ``t``: land ``skt``; open water ``sst`` (lakes, where
      ``sst`` is undefined and ``cl >= 0.5``, use ``skt``); ice points ``skt``.
    - Soil temperature and volumetric moisture are mapped from the HTESSEL
      layers (0-7, 7-28, 28-100, 100-289 cm) to the Noah layers (0-10,
      10-40, 40-100, 100-200 cm) by depth-weighted layer overlap. Moisture is
      rescaled by soil wetness index from the HTESSEL class (ERA5 ``slt``,
      Balsamo et al. 2009, table 2) to the Noah STATSGO class of the GFS
      file, with the reference and wilting points computed as in
      ``chgres_cube`` ``calc_soil_params``. Liquid moisture uses the IFS
      soil-water freezing function; ``chgres_cube`` recomputes it on the
      target grid in any case.
    - Ice points take ECMWF ice temperatures ``istl1-4`` (0-7, 7-28, 28-100,
      100-150 cm) in the soil temperature records, as GFS does.
    - Snow: ``sdwe = 1000 sd`` [kg m-2], ``sde = 1000 sd / rsn`` [m]. GFS
      land-ice points (``soilw = 1``), ECMWF glacier points and snow on sea
      ice, which ECMWF does not analyse, keep the GFS values; GFS soil
      moisture is also kept on GFS land ice. HTESSEL holds glaciers at 10 m
      w.e. of snow and interpolation dilutes that value on small glaciers,
      so ECMWF snow above ``GLACIER_SWE`` (1 m w.e.) counts as glacier.
    - Canopy water ``cnwat = 1000 src`` [kg m-2], capped at the GFS maximum.
    - ``2t, 2d, 10u, 10v`` directly; ``2sh`` from ``2d`` and ``p_s`` with the
      IFS saturation vapour pressure over water; ``prmsl = msl``. ``2r``,
      ``aptmp`` and ``mslet`` are dropped.

Every other GFS record is copied unchanged. The ECMWF data must be on the GFS
grid (regular 0.25 degree, 0..359.75 E); ``retrieve_era5`` requests it so.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path

import cdsapi
import eccodes
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from grib_io import open_grib, save_grib

log = logging.getLogger("ecmwf_ic")

# GFS 0.25 degree files: NOAA AWS S3, then NCAR GDEX (the sources of
# core/fv3_ic_data.get_ic_data).
GFS_URLS = (
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/{hour}/atmos/"
    "gfs.t{hour}z.pgrb2.0p25.f{fh:03d}",
    "https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d084001/{year}/{date}/"
    "gfs.0p25.{date}{hour}.f{fh:03d}.grib2",
)

# IFS physical constants (IFS documentation, Part III; also used by ERA5).
RD = 287.0597  # J kg-1 K-1
RV = 461.5250  # J kg-1 K-1
G = 9.80665  # m s-2
EPS = RD / RV
RETV = RV / RD - 1.0
TMELT = 273.16  # K, IFS reference for saturation vapour pressure
TFRZ = 273.15  # K
GAMMA = 0.0065  # K m-1, standard lapse rate (Trenberth et al. 1993)
GLACIER_SWE = 1.0  # m w.e.; ECMWF snow above this is treated as glacier

# Saturation vapour pressure over water, IFS Tetens coefficients.
ES_A1, ES_A3, ES_A4 = 611.21, 17.502, 32.19

# Layer boundaries in metres.
HTESSEL_SOIL = np.array([0.0, 0.07, 0.28, 1.00, 2.89])
ERA5_ICE = np.array([0.0, 0.07, 0.28, 1.00, 1.50])
NOAH_SOIL = np.array([0.0, 0.1, 0.4, 1.0, 2.0])

# HTESSEL soil classes 1-6 (ERA5 slt): saturation, field capacity and
# permanent wilting point [m3 m-3]; Balsamo et al. (2009), table 2.
# Class 7 (tropical organic) is not in that table and is not rescaled.
HTESSEL_CAP = {1: 0.244, 2: 0.347, 3: 0.383, 4: 0.448, 5: 0.541, 6: 0.663}
HTESSEL_PWP = {1: 0.059, 2: 0.151, 3: 0.133, 4: 0.279, 5: 0.335, 6: 0.267}

# Noah STATSGO tables from UFS_UTILS chgres_cube program_setup.F90
# (calc_soil_params_driver, input grid, non-sigio input).
STATSGO_BB = np.array(
    [4.05, 4.26, 4.74, 5.33, 5.33, 5.25, 6.77, 8.72, 8.17, 10.73, 10.39, 11.55,
     5.25, -9.99, 4.05, 4.26]
)  # fmt: skip
STATSGO_MAXSMC = np.array(
    [0.395, 0.421, 0.434, 0.476, 0.476, 0.439, 0.404, 0.464, 0.465, 0.406,
     0.468, 0.457, 0.464, -9.99, 0.200, 0.421]
)  # fmt: skip
STATSGO_SATDK = np.array(
    [1.7600e-4, 1.4078e-5, 5.2304e-6, 2.8089e-6, 2.8089e-6, 3.3770e-6,
     4.4518e-6, 2.0348e-6, 2.4464e-6, 7.2199e-6, 1.3444e-6, 9.7384e-7,
     3.3770e-6, -9.99, 1.4078e-5, 1.4078e-5]
)  # fmt: skip
STATSGO_SATPSI = np.array(
    [0.0350, 0.0363, 0.1413, 0.7586, 0.7586, 0.3548, 0.1349, 0.6166, 0.2630,
     0.0977, 0.3236, 0.4677, 0.3548, -9.99, 0.0350, 0.0363]
)  # fmt: skip

# ERA5 parameters (paramId) for the model-level retrieval.
ML_PARAMS = {
    "t": 130,
    "q": 133,
    "u": 131,
    "v": 132,
    "w": 135,
    "o3": 203,
    "clwc": 246,
    "ciwc": 247,
    "crwc": 75,
    "cswc": 76,
    "cc": 248,
}
SFC_VARIABLES = [
    "skin_temperature",
    "sea_surface_temperature",
    "sea_ice_cover",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "ice_temperature_layer_1",
    "ice_temperature_layer_2",
    "ice_temperature_layer_3",
    "ice_temperature_layer_4",
    "snow_depth",
    "snow_density",
    "land_sea_mask",
    "soil_type",
    "lake_cover",
    "skin_reservoir_content",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "geopotential",
]

# GFS records that would be left describing the GFS state and that
# chgres_cube does not read.
DROPPED = {
    "isobaricInhPa": ("r", "absv"),
    "isobaricInPa": ("r", "absv"),
    "heightAboveGround": ("2r", "aptmp"),
    "meanSea": ("mslet",),
}


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #


def retrieve_era5(valid: pd.Timestamp, dest: Path, expver: str = "1") -> dict:
    """Download the ERA5 analysis at ``valid`` from the CDS on the GFS grid.

    Returns paths for ``ml`` (137 model levels), ``ml_surface`` (``z`` and
    ``lnsp`` on model level 1) and ``sfc`` (single-level fields). Requires a
    CDS key in ``~/.cdsapirc``. Model levels come from ``reanalysis-era5-
    complete`` (MARS tape archive, which can queue for a long time). If
    ``expver`` 1 is not yet available, the preliminary ERA5T stream
    (``expver`` 5) is tried.
    """
    if valid.minute or valid.second:
        raise ValueError(f"ERA5 analyses are hourly; got {valid}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = valid.strftime("%Y%m%d%H")
    paths = {
        "ml": dest / f"ERA5.{stamp}.ml.grib",
        "ml_surface": dest / f"ERA5.{stamp}.ml_surface.grib",
        "sfc": dest / f"ERA5.{stamp}.sfc.grib",
    }
    common = {
        "class": "ea",
        "stream": "oper",
        "type": "an",
        "date": valid.strftime("%Y-%m-%d"),
        "time": valid.strftime("%H:00:00"),
        "levtype": "ml",
        "grid": "0.25/0.25",
        "format": "grib",
    }
    requests = {
        "ml": (
            "reanalysis-era5-complete",
            {
                **common,
                "levelist": "1/to/137",
                "param": "/".join(str(p) for p in ML_PARAMS.values()),
            },
        ),
        "ml_surface": (
            "reanalysis-era5-complete",
            {
                **common,
                "levelist": "1",
                "param": "129/152",
            },
        ),
        "sfc": (
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": SFC_VARIABLES,
                "date": [valid.strftime("%Y-%m-%d")],
                "time": [valid.strftime("%H:00")],
                "data_format": "grib",
                "download_format": "unarchived",
            },
        ),
    }
    client = cdsapi.Client()
    for key, (dataset, request) in requests.items():
        path = paths[key]
        if path.exists() and path.stat().st_size > 0:
            continue
        partial = path.with_name(f".{path.name}.part")
        versions = [expver, "5"] if dataset.endswith("complete") else [None]
        for version in dict.fromkeys(versions):
            if version is not None:
                request = {**request, "expver": version}
            try:
                log.info(f"Requesting {key} from CDS {dataset} for {valid}")
                client.retrieve(dataset, request, str(partial))
                os.replace(partial, path)
                break
            except Exception as error:  # cdsapi raises generic exceptions
                partial.unlink(missing_ok=True)
                if version == versions[-1] or version is None:
                    raise RuntimeError(f"CDS request {key} failed: {error}") from error
                log.warning(f"CDS {key} with expver {version} failed: {error}")
    return paths


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def read_field(tree: xr.DataTree, node: str, name: str) -> np.ndarray:
    """Return a variable at its first time as float64; squeeze 1-level axes."""
    data = tree[node].ds[name].isel(time=0).values.astype(np.float64)
    return data[0] if data.shape[0] == 1 else data


def read_soil_layer(tree: xr.DataTree, node: str, name: str) -> np.ndarray:
    """Return a per-layer variable (e.g. ``stl1``) from the one layer it fills."""
    data = tree[node].ds[name].isel(time=0).values.astype(np.float64)
    filled = ~np.all(np.isnan(data), axis=(1, 2))
    if filled.sum() != 1:
        raise ValueError(f"{node}/{name}: expected exactly one filled layer")
    return data[np.argmax(filled)]


def fill_from_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace cells outside ``valid`` (and NaNs) by their nearest valid value."""
    valid = valid & np.isfinite(values if values.ndim == 2 else values.min(axis=0))
    if valid.all():
        return values
    if not valid.any():
        raise ValueError("No valid points to fill from")
    # Pad in longitude so the nearest point can lie across 0/360 E.
    wrap, nx = 32, valid.shape[1]
    padded = np.concatenate([valid[:, -wrap:], valid, valid[:, :wrap]], axis=1)
    iy, ix = distance_transform_edt(
        ~padded, return_distances=False, return_indices=True
    )
    iy, ix = iy[:, wrap:-wrap], (ix[:, wrap:-wrap] - wrap) % nx
    return values[..., iy, ix]


def map_layers(
    layers: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Average ``layers`` (source layer bounds) onto target layer bounds [m].

    Each target layer is the depth-weighted mean of the source layers it
    overlaps; a target layer below the source column takes the deepest one.
    """
    weights = np.zeros((target.size - 1, source.size - 1))
    for i in range(target.size - 1):
        for j in range(source.size - 1):
            top, bottom = max(target[i], source[j]), min(target[i + 1], source[j + 1])
            weights[i, j] = max(0.0, bottom - top)
        if weights[i].sum() == 0:
            weights[i, -1] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ts,syx->tyx", weights, layers)


def replace_variable(tree: xr.DataTree, node: str, name: str, values) -> None:
    """Overwrite ``tree[node][name]`` with ``values`` (float32, same shape)."""
    data = tree[node].ds[name]
    new = np.asarray(values, np.float32).reshape(data.shape[1:])[None]
    tree[node] = tree[node].to_dataset().assign({name: data.copy(data=new)})


def rescale_soil_moisture(
    theta: np.ndarray, era_type: np.ndarray, noah_type: np.ndarray
) -> np.ndarray:
    """Map HTESSEL volumetric moisture to Noah by soil wetness index.

    ``SWI = (theta - pwp) / (cap - pwp)`` is preserved; the Noah reference
    and wilting points stand for field capacity and wilting point, as in
    ``chgres_cube`` ``rescale_soil_moisture``. Points whose HTESSEL class is
    outside 1-6, or whose Noah class has no hydraulic data (water, class 14),
    keep the HTESSEL value. The result is limited to the Noah saturation.
    """
    # Noah reference and wilting points, as chgres_cube calc_soil_params.
    maxsmc, bb = STATSGO_MAXSMC, STATSGO_BB
    has_soil = (maxsmc > 0) & (maxsmc < 1)
    refsmc, wltsmc = np.zeros_like(maxsmc), np.zeros_like(maxsmc)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref1 = maxsmc * (5.79e-9 / STATSGO_SATDK) ** (1.0 / (2.0 * bb + 3.0))
        wlt1 = maxsmc * (200.0 / STATSGO_SATPSI) ** (-1.0 / bb)
    refsmc[has_soil] = (ref1 + (maxsmc - ref1) / 6.0)[has_soil]
    wltsmc[has_soil] = (wlt1 - 0.5 * wlt1)[has_soil]

    out = theta.copy()
    era = np.rint(era_type).astype(int)
    noah = np.rint(noah_type).astype(int)
    for code in HTESSEL_CAP:
        cap, pwp = HTESSEL_CAP[code], HTESSEL_PWP[code]
        for n in range(1, maxsmc.size + 1):
            if not has_soil[n - 1]:
                continue
            mask = (era == code) & (noah == n)
            if not mask.any():
                continue
            swi = (theta[:, mask] - pwp) / (cap - pwp)
            out[:, mask] = wltsmc[n - 1] + swi * (refsmc[n - 1] - wltsmc[n - 1])

    # Cap at the Noah saturation of each point's soil class.
    limit = np.where(
        (noah >= 1) & (noah <= maxsmc.size),
        maxsmc[np.clip(noah, 1, maxsmc.size) - 1],
        np.nan,
    )
    limit = np.where(limit > 0, limit, np.nan)
    out = np.where(np.isfinite(limit), np.minimum(out, limit), out)
    return np.maximum(out, 0.02)  # GFS lower bound for SOILW


# --------------------------------------------------------------------------- #
# atmosphere
# --------------------------------------------------------------------------- #


def ecmwf_to_pressure_levels(ml: xr.DataTree, ps, phis, a, b, levels_pa) -> dict:
    """Interpolate ECMWF model-level fields to ``levels_pa`` [Pa].

    ``ps`` [Pa] and ``phis`` [m2 s-2] are the surface pressure and
    geopotential of the model levels; ``a`` [Pa] and ``b`` are the half-level
    coefficients. Returns float32 arrays (level, lat, lon) under GFS short
    names and units.
    """
    node = "hybrid"
    nlev = a.size - 1
    levels = ml[node].ds[node].values
    if levels.size != nlev or not np.array_equal(levels, np.arange(1, nlev + 1)):
        raise ValueError(f"Model levels must be 1..{nlev}; found {levels[[0, -1]]}")
    af = 0.5 * (a[:-1] + a[1:])  # full-level coefficients
    bf = 0.5 * (b[:-1] + b[1:])

    def load(name):
        log.info(f"Loading ECMWF model-level {name}")
        return ml[node].ds[name].isel(time=0).values.astype(np.float32)

    t, q = load("t"), load("q")

    # Half-level geopotential, integrated upward from the surface (IFS scheme).
    # Level 0 (p = 0) is never used: the top layer is integrated in-layer.
    log.info("Integrating half-level geopotential")
    phi_half = np.empty((nlev + 1, *ps.shape), np.float32)
    phi_half[nlev] = phis
    phi = phis.copy()
    for k in range(nlev - 1, 0, -1):
        dlogp = np.log((a[k + 1] + b[k + 1] * ps) / (a[k] + b[k] * ps))
        phi = phi + dlogp * RD * t[k] * (1.0 + RETV * q[k])
        phi_half[k] = phi
    phi_half[0] = np.inf

    # Bracketing full levels and ln p weights for every target level.
    plev = np.asarray(levels_pa, np.float64)
    count = np.zeros((plev.size, *ps.shape), np.int16)
    for k in range(nlev):
        count += (af[k] + bf[k] * ps)[None] < plev[:, None, None]
    kup = np.clip(count.astype(np.int32) - 1, 0, nlev - 2)
    kdn = kup + 1
    lnp = np.log(plev)[:, None, None]
    ln_up = np.log(af[kup] + bf[kup] * ps)
    ln_dn = np.log(af[kdn] + bf[kdn] * ps)
    weight = np.clip((lnp - ln_up) / (ln_dn - ln_up), 0.0, 1.0).astype(np.float32)
    below = count == nlev  # target level below the lowest full level
    del count, ln_up, ln_dn

    def interp(stack):
        upper = np.take_along_axis(stack, kup, axis=0)
        lower = np.take_along_axis(stack, kdn, axis=0)
        return upper + weight * (lower - upper)

    # Terms of the Trenberth et al. (1993) extrapolation (NCL vinth2p_ecmwf).
    alpha = GAMMA * RD / G
    t_bot = t[-1].astype(np.float64)
    p_bot = af[-1] + bf[-1] * ps
    hgt = phis / G
    tstar = t_bot * (1.0 + alpha * (ps / p_bot - 1.0))
    t0 = tstar + GAMMA * hgt

    out = {}

    # Temperature; below the lowest level, NCL varflg 1.
    out["t"] = interp(t)
    tplat = np.minimum(t0, 298.0)
    tprime0 = np.where(
        hgt <= 2500.0, 0.002 * ((2500.0 - hgt) * t0 + (hgt - 2000.0) * tplat), tplat
    )
    for j, p in enumerate(plev):
        if not below[j].any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            high = np.where(
                tprime0 < tstar, 0.0, RD * (tprime0 - tstar) / phis * np.log(p / ps)
            )
        alnp = np.where(hgt < 2000.0, alpha * np.log(p / ps), high)
        t_ext = tstar * (1.0 + alnp + 0.5 * alnp**2 + alnp**3 / 6.0)
        out["t"][j] = np.where(below[j], t_ext, out["t"][j])

    # Geopotential height: hydrostatic integration inside the model layer
    # holding p, Phi(p) = Phi(k+1/2) + Rd Tv_k ln(p(k+1/2) / p); below the
    # surface NCL varflg -1. Both equal Phi_s at p = p_s.
    with np.errstate(divide="ignore", invalid="ignore"):
        case1 = (tstar <= 290.5) & (t0 > 290.5)
        case2 = (tstar > 290.5) & (t0 > 290.5)
        alph = np.where(case1, RD / phis * (290.5 - tstar), np.where(case2, 0.0, alpha))
    tstar_z = np.where(case2, 0.5 * (290.5 + tstar), tstar)
    tstar_z = np.where(tstar_z < 255.0, 0.5 * (tstar_z + 255.0), tstar_z)
    gh = np.empty(weight.shape, np.float32)
    for j, p in enumerate(plev):
        layer = np.zeros(ps.shape, np.int32)
        for k in range(1, nlev):
            layer += (a[k] + b[k] * ps) < p
        layer = layer[None]
        p_lower = a[layer[0] + 1] + b[layer[0] + 1] * ps
        tv = np.take_along_axis(t, layer, 0)[0] * (
            1.0 + RETV * np.take_along_axis(q, layer, 0)[0]
        )
        inside = np.take_along_axis(phi_half, layer + 1, 0)[0] + RD * tv * np.log(
            p_lower / p
        )
        alnp = alph * np.log(p / ps)
        under = hgt - RD * tstar_z / G * np.log(p / ps) * (
            1.0 + 0.5 * alnp + alnp**2 / 6.0
        )
        gh[j] = np.where(p > ps, under * G, inside) / G
    out["gh"] = gh
    del phi_half

    spfh = np.maximum(interp(q), 0.0)
    out["q"] = spfh
    del t, q

    # Remaining fields keep their lowest-level value below ground.
    for gfs, era in (
        ("u", "u"),
        ("v", "v"),
        ("w", "w"),
        ("o3mr", "o3"),
        ("clwmr", "clwc"),
        ("icmr", "ciwc"),
        ("rwmr", "crwc"),
        ("snmr", "cswc"),
        ("tcc", "cc"),
    ):
        out[gfs] = interp(load(era))
    for name in ("o3mr", "clwmr", "icmr", "rwmr", "snmr"):
        out[name] = np.maximum(out[name], 0.0)
    out["tcc"] = np.clip(100.0 * out["tcc"], 0.0, 100.0)  # fraction to %
    out["grle"] = np.zeros_like(out["t"])  # ERA5 has no graupel

    # Vertical velocity dz/dt = -omega / (rho g), with rho = p / (Rd Tv).
    tv = out["t"] * (np.float32(1.0) + np.float32(RETV) * spfh)
    rho_g = (
        np.float32(G) * plev[:, None, None].astype(np.float32) / (np.float32(RD) * tv)
    )
    out["wz"] = -out["w"] / rho_g
    return out


# --------------------------------------------------------------------------- #
# surface
# --------------------------------------------------------------------------- #


def ecmwf_surface(gfs: xr.DataTree, sfc: xr.DataTree, ps: np.ndarray, phis: np.ndarray):
    """Map ECMWF surface fields onto the GFS land/sea/ice layout."""
    s = "surface"
    gfs_land = read_field(gfs, s, "lsm") >= 0.5
    gfs_water = ~gfs_land
    e_land = read_field(sfc, s, "lsm") >= 0.5
    e_water = ~e_land
    out = {"sp": ps, "orog": phis / G}

    # Sea ice on GFS water only.
    ci = fill_from_nearest(read_field(sfc, s, "ci"), e_water)
    ci = np.where(gfs_land, 0.0, np.clip(ci, 0.0, 1.0))
    ice = gfs_water & (ci > 0.0)
    out["ci"] = ci
    # ECMWF has no analysed ice thickness. Keep the GFS value where GFS has ice
    # and take the nearest GFS ice thickness where only ECMWF has ice.
    hice = read_field(gfs, s, "sithick")
    out["sithick"] = np.where(
        ice, fill_from_nearest(hice, gfs_water & (hice > 0.0)), 0.0
    )

    # Skin temperature: land skt, water sst (skt on lakes), ice skt.
    skt = read_field(sfc, s, "skt")
    sst = read_field(sfc, s, "sst")
    lake = read_field(sfc, s, "cl") >= 0.5
    water_t = np.where(np.isfinite(sst), sst, np.where(lake, skt, np.nan))
    water_t = fill_from_nearest(water_t, np.isfinite(water_t))
    land_skt = fill_from_nearest(skt, e_land)
    out["t"] = np.where(gfs_land, land_skt, np.where(ci > 0.15, skt, water_t))

    # Snow (ECMWF analyses snow on land only).
    sd = read_field(sfc, s, "sd")
    rsn = read_field(sfc, s, "rsn")
    land_sd = fill_from_nearest(sd, e_land)
    land_rsn = fill_from_nearest(rsn, e_land)
    layer = "depthBelowLandLayer"
    gfs_soilw = gfs[layer].ds["soilw"].isel(time=0).values.astype(np.float64)
    gfs_soill = gfs[layer].ds["soill"].isel(time=0).values.astype(np.float64)
    landice = gfs_land & (gfs_soilw[0] >= 0.999)  # GFS land-ice flag (SOILW = 1)
    # HTESSEL represents glaciers as snow fixed at 10 m w.e.; interpolation to
    # 0.25 degree dilutes the flag on small glaciers, so any ECMWF snow above
    # GLACIER_SWE is treated as glacier and keeps the GFS analysis.
    glacier = landice | (land_sd > GLACIER_SWE)
    gfs_sdwe, gfs_sde = read_field(gfs, s, "sdwe"), read_field(gfs, s, "sde")
    sdwe = np.where(gfs_land, 1000.0 * land_sd, np.nan)  # m w.e. to kg m-2
    sde = np.where(gfs_land, 1000.0 * land_sd / land_rsn, np.nan)  # depth [m]
    keep = (gfs_land & glacier) | ice
    sdwe = np.where(keep, np.nan_to_num(gfs_sdwe), sdwe)
    sde = np.where(keep, np.nan_to_num(gfs_sde), sde)
    out["sdwe"], out["sde"] = sdwe, sde

    # Canopy water, m to kg m-2, capped at the GFS maximum.
    src = fill_from_nearest(read_field(sfc, s, "src"), e_land)
    cap = np.nanmax(read_field(gfs, s, "cnwat"))
    out["cnwat"] = np.where(gfs_land, np.minimum(1000.0 * src, cap), np.nan)

    # Soil (land) and ice (sea-ice points) layers, mapped to the Noah layers.
    stl = np.stack([read_soil_layer(sfc, layer, f"stl{i}") for i in range(1, 5)])
    swvl = np.stack([read_soil_layer(sfc, layer, f"swvl{i}") for i in range(1, 5)])
    istl = np.stack([read_soil_layer(sfc, layer, f"istl{i}") for i in range(1, 5)])
    stl = map_layers(fill_from_nearest(stl, e_land), HTESSEL_SOIL, NOAH_SOIL)
    swvl = map_layers(fill_from_nearest(swvl, e_land), HTESSEL_SOIL, NOAH_SOIL)
    istl = map_layers(
        fill_from_nearest(istl, e_water & np.isfinite(istl[0])), ERA5_ICE, NOAH_SOIL
    )

    era_type = read_field(sfc, s, "slt")
    era_type = fill_from_nearest(era_type, e_land & (era_type > 0))
    swvl = rescale_soil_moisture(swvl, era_type, read_field(gfs, s, "slt"))

    # Liquid water from the IFS frozen fraction (Viterbo et al. 1999):
    # all liquid above Tf + 1 K, all frozen below Tf - 3 K, sine in between.
    t1, t2 = TFRZ + 1.0, TFRZ - 3.0
    frozen = 0.5 * (1.0 - np.sin(np.pi * (stl - 0.5 * (t1 + t2)) / (t1 - t2)))
    frozen = np.where(stl >= t1, 0.0, np.where(stl <= t2, 1.0, frozen))
    soill = swvl * (1.0 - frozen)

    out["st"] = np.where(gfs_land, stl, np.where(ice, istl, np.nan))
    out["soilw"] = np.where(landice, gfs_soilw, np.where(gfs_land, swvl, np.nan))
    out["soill"] = np.where(landice, gfs_soill, np.where(gfs_land, soill, np.nan))

    # Screen level; 2 m specific humidity from the dewpoint with the IFS
    # saturation vapour pressure over water (Tetens).
    t2m, td2m = read_field(sfc, s, "2t"), read_field(sfc, s, "2d")
    e = ES_A1 * np.exp(ES_A3 * (td2m - TMELT) / (td2m - ES_A4))
    out["2t"], out["2d"] = t2m, td2m
    out["2sh"] = EPS * e / (ps - (1.0 - EPS) * e)
    out["10u"], out["10v"] = read_field(sfc, s, "10u"), read_field(sfc, s, "10v")
    out["prmsl"] = read_field(sfc, s, "msl")
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def build_gfs_from_ecmwf(
    gfs_file: Path,
    ml_file: Path,
    ml_surface_file: Path | None,
    sfc_file: Path,
    output: Path,
) -> Path:
    """Write ``output``: the GFS file with its fields replaced by ECMWF data.

    ``ml_surface_file`` holds ``lnsp`` and ``z`` on model level 1, the
    surface of the model levels. Without it the single-level ``sp`` and ``z``
    of ``sfc_file`` are used.
    """
    gfs = open_grib(gfs_file)
    ml, sfc = open_grib(ml_file), open_grib(sfc_file)
    trees = [("ml", ml, "hybrid"), ("sfc", sfc, "surface")]
    if ml_surface_file is not None:
        ml_surface = open_grib(ml_surface_file)
        trees.append(("ml_surface", ml_surface, "hybrid"))
        ps = np.exp(read_field(ml_surface, "hybrid", "lnsp"))
        phis = read_field(ml_surface, "hybrid", "z")
    else:
        log.warning("No model-level lnsp/z file; using single-level sp and z")
        ps, phis = read_field(sfc, "surface", "sp"), read_field(sfc, "surface", "z")

    # ECMWF data must share the GFS grid and valid time.
    reference = gfs["isobaricInhPa"].ds
    for label, tree, node in trees:
        for coord in ("latitude", "longitude"):
            if not np.array_equal(reference[coord].values, tree[node].ds[coord].values):
                raise ValueError(f"{label}: {coord} differs from the GFS grid")
        if tree[node].ds.time.values[0] != reference.time.values[0]:
            raise ValueError(
                f"{label} valid time {tree[node].ds.time.values[0]} differs from GFS "
                f"{reference.time.values[0]}"
            )

    # Half-level a [Pa] and b from the GRIB pv array of the first message.
    with Path(ml_file).open("rb") as file:
        gid = eccodes.codes_grib_new_from_file(file)
        try:
            pv = eccodes.codes_get_array(gid, "pv").astype(np.float64)
        finally:
            eccodes.codes_release(gid)
    a, b = pv[: pv.size // 2], pv[pv.size // 2 :]

    # Interpolate to the GFS levels of both isobaric nodes (Pa, then hPa).
    nodes = [n for n in ("isobaricInPa", "isobaricInhPa") if n in gfs.children]
    scale = {"isobaricInPa": 1.0, "isobaricInhPa": 100.0}
    levels = np.concatenate([gfs[n].ds[n].values * scale[n] for n in nodes])
    atm = ecmwf_to_pressure_levels(ml, ps, phis, a, b, levels)

    bounds, start = {}, 0
    for n in nodes:
        bounds[n] = slice(start, start + gfs[n].ds.sizes[n])
        start = bounds[n].stop
    for name in list(atm):
        values = atm.pop(name)  # pop to free memory as fields are placed
        for n in nodes:
            if name in gfs[n].ds:
                replace_variable(gfs, n, name, values[bounds[n]])
        del values

    surface = ecmwf_surface(gfs, sfc, ps, phis)
    placement = {
        "surface": ("sp", "orog", "ci", "sithick", "t", "sdwe", "sde", "cnwat"),
        "depthBelowLandLayer": ("st", "soilw", "soill"),
        "heightAboveGround": ("2t", "2d", "2sh", "10u", "10v"),
        "meanSea": ("prmsl",),
    }
    screen_height = {"2t": 2.0, "2d": 2.0, "2sh": 2.0, "10u": 10.0, "10v": 10.0}
    for node, names in placement.items():
        for name in names:
            if node == "heightAboveGround":
                # Only the 2 m or 10 m level changes; other heights keep GFS.
                data = gfs[node].ds[name]
                full = data.isel(time=0).values.astype(np.float64)
                heights = gfs[node].ds[node].values
                index = int(np.flatnonzero(heights == screen_height[name])[0])
                full[index] = surface[name]
                replace_variable(gfs, node, name, full)
            else:
                replace_variable(gfs, node, name, surface[name])

    for node, names in DROPPED.items():
        if node in gfs.children:
            present = [n for n in names if n in gfs[node].ds]
            gfs[node] = gfs[node].to_dataset().drop_vars(present)

    log.info(f"Writing {output}")
    save_grib(gfs, output, template=gfs_file)
    return Path(output)


def retrieve_gfs(cycle: pd.Timestamp, forecast_hour: int, dest: Path) -> Path:
    """Download the GFS 0.25 degree file of ``cycle`` and lead ``forecast_hour``."""
    path = Path(dest) / f"GFS.{cycle:%Y%m%d%H}Z.f{forecast_hour:03d}.0p25deg.grib2"
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    fields = {"date": f"{cycle:%Y%m%d}", "hour": f"{cycle:%H}", "year": f"{cycle:%Y}"}
    for template in GFS_URLS:
        url = template.format(fh=forecast_hour, **fields)
        try:
            log.info(f"Downloading {url}")
            urllib.request.urlretrieve(url, partial)
            os.replace(partial, path)
            return path
        except OSError as error:
            partial.unlink(missing_ok=True)
            log.warning(f"{url} failed: {error}")
    raise RuntimeError(
        f"GFS {cycle:%Y%m%d%H} f{forecast_hour:03d} not found; use --gfs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write ERA5 initial conditions into a GFS GRIB2 file for chgres_cube."
    )
    parser.add_argument("--cycle", required=True, help="GFS cycle, YYYYMMDDHH")
    parser.add_argument("--fh", type=int, default=0, help="GFS lead [h] (default: 0)")
    parser.add_argument("--out", type=Path, default=Path.cwd(), help="output directory")
    parser.add_argument(
        "--gfs", type=Path, help="GFS template file (default: download)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cycle = pd.Timestamp(pd.to_datetime(args.cycle, format="%Y%m%d%H"))
    out = args.out.expanduser().resolve()
    download = out / "download"
    gfs = (
        args.gfs.expanduser().resolve()
        if args.gfs
        else retrieve_gfs(cycle, args.fh, download)
    )
    era = retrieve_era5(cycle + pd.Timedelta(hours=args.fh), download / "ERA5")
    output = out / f"GFS.{cycle:%Y%m%d%H}Z.f{args.fh:03d}.0p25deg.ECMWF.grib2"
    build_gfs_from_ecmwf(gfs, era["ml"], era["ml_surface"], era["sfc"], output)
    print(output)


if __name__ == "__main__":
    main()
