"""Locate static (fix) files in fix_src and fetch missing ones from NOAA.

A file is looked up in this order, and the first hit is used:

1. ``fix_src/<rel>`` itself.
2. A file already present in ``fix_src`` under the name NOAA uses for it
   (for example ``am/fix_co2_update/global_co2historicaldata_2020.txt`` for
   ``am/co2historicaldata_2020.txt``). ``<rel>`` is linked to it.
3. The NOAA global fix bucket
   (https://noaa-nws-global-pds.s3.amazonaws.com/index.html#fix/), searched
   from the newest version directory to the oldest, so files dropped from the
   latest release (the binary orography inputs read by the ``orog`` program of
   the preprocessing image) are still found. The file is stored in
   ``fix_src`` under its NOAA name and ``<rel>`` is linked to it.

An error is raised only when a required file is found neither locally nor
remotely. Downloads are written to a temporary name and renamed into place,
so concurrent cases sharing a fix tree do not see partial files. Set
``UFS_PY_FIX_REMOTE=0`` to disable remote lookups.

Used at run time only. ``configs/update_fix.py`` is a separate, self-contained
tool that mirrors the whole fix tree in advance.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger("FIX")

# UFS_PY_FIX_BUCKET_URL and UFS_PY_FIX_VARMAP_URL select a mirror.
BUCKET_URL = os.environ.get(
    "UFS_PY_FIX_BUCKET_URL", "https://noaa-nws-global-pds.s3.amazonaws.com"
)
BUCKET_PREFIX = "fix"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# chgres_cube variable maps are not in the NOAA bucket. They are taken from
# the UFS_UTILS fork that the preprocessing image is built from, pinned to
# the commit of that build (gaea branch, 2024-10-18).
VARMAP_URL = os.environ.get(
    "UFS_PY_FIX_VARMAP_URL",
    "https://raw.githubusercontent.com/kaiyuan-cheng/UFS_UTILS/"
    + "0f7c355f2cf2c8f3b27d0c059e6d9f4a1612b578/parm/varmap_tables",
)

TIMEOUT_S = 120
ATTEMPTS = 3

# --- Files read by the workflow ---------------------------------------------

# namsfc climatologies (fv3_namelists.update_namsfc), keyed by namelist entry.
NAMSFC_FILES = {
    "fnabsc": "global_mxsnoalb.uariz.t1534.3072.1536.rg.grb",
    "fnaisc": "CFSR.SEAICE.1982.2012.monthly.clim.grb",
    "fnalbc": "global_snowfree_albedo.bosu.t1534.3072.1536.rg.grb",
    "fnalbc2": "global_albedo4.1x1.grb",
    "fnglac": "global_glacier.2x2.grb",
    "fnmldc": "mld_DR003_c1m_reg2.0.grb",
    "fnmskh": "seaice_newland.grb",
    "fnmxic": "global_maxice.2x2.grb",
    "fnslpc": "global_slope.1x1.grb",
    "fnsmcc": "global_soilmgldas.t1534.3072.1536.grb",
    "fnsnoc": "global_snoclim.1.875.grb",
    "fnsotc": "global_soiltype.statsgo.t1534.3072.1536.rg.grb",
    "fntg3c": "global_tg3clim.2.6x1.5.grb",
    "fntsfc": "RTGSST.1982.2012.monthly.clim.grb",
    "fnvegc": "global_vegfrac.0.144.decpercent.grb",
    "fnvetc": "global_vegtype.igbp.t1534.3072.1536.rg.grb",
    "fnvmnc": "global_shdmin.0.144x0.144.grb",
    "fnvmxc": "global_shdmax.0.144x0.144.grb",
}

# Radiation and ozone/water-vapour physics inputs (fv3_fixed_files). The
# year-dependent CO2 and volcanic files are added per run.
PHYSICS_FILES = (
    "aerosol.dat",
    "co2historicaldata_glob.txt",
    "co2monthlycyc.txt",
    "sfc_emissivity_idx.txt",
    "solarconstant_noaa_an.txt",
    "global_h2oprdlos.f77",
    "global_o3prdlos.f77",
)

# Inputs of the orog program in the preprocessing image (fv3_make_orog).
OROG_FILES = (
    "thirty.second.antarctic.new.bin",
    "landcover30.fixed",
    "gmted2010.30sec.int",
)

GSL_OROG_FILES = (
    "HGT.Beljaars_filtered.lat-lon.30s_res.nc",
    "geo_em.d01.lat-lon.2.5m.HGT_M.nc",
)

# lakefrac of the preprocessing image reads GLDB v2 data under these names.
LAKE_FILES = ("GlobalLakeStatus.dat", "GlobalLakeDepth.dat")

SFC_CLIMO_FILES = (
    "facsf.1.0.nc",
    "substrate_temperature.gfs.0.5.nc",
    "maximum_snow_albedo.0.05.nc",
    "snowfree_albedo.4comp.0.05.nc",
    "slope_type.1.0.nc",
    "soil_color.clm.0.05.nc",
    "vegetation_greenness.0.144.nc",
)

VARMAP_FILES = ("GFSphys_var_map.txt", "GSDphys_var_map.txt")

HRRR_GEOGRID = "geo_em.d01.nc_HRRRX"

# Files without a remote source; they must be staged in fix_src by hand.
LOCAL_ONLY_HINTS = {
    "am/mld_DR003_c1m_reg2.0.grb": "GFDL fvGFS_INPUT_DATA (not in the NOAA bucket)",
    "era5/": "fix_src/era5 (built locally from ERA5)",
    "carto/": "configs/update_fix.py (Cartopy Natural Earth download)",
}


def run_manifest(
    years: list[int] | range,
    levels: int | list[int] = 64,
    veg_type_src: str = "modis.igbp.0.05",
    soil_type_src: str = "statsgo.0.05",
    gsl: bool = False,
    lake: bool = False,
    hrrr: bool = True,
) -> list[str]:
    """Fix-relative paths read by a run spanning `years`."""
    rel = [f"am/{name}" for name in NAMSFC_FILES.values()]
    rel += [f"am/{name}" for name in PHYSICS_FILES]
    rel += [f"am/co2historicaldata_{y}.txt" for y in years]
    rel += [f"am/{v}" for v in volcanic_files(years)]
    for nlev in [levels] if isinstance(levels, int) else levels:
        rel.append(f"am/global_hyblev.l{nlev}.txt")
    rel += [f"orog/{name}" for name in OROG_FILES]
    if gsl:
        rel += [f"orog/{name}" for name in GSL_OROG_FILES]
    if lake:
        rel += [f"orog/{name}" for name in LAKE_FILES]
    rel += [f"sfc_climo/{name}" for name in SFC_CLIMO_FILES]
    rel += [
        f"sfc_climo/soil_type.{soil_type_src}.nc",
        f"sfc_climo/vegetation_type.{veg_type_src}.nc",
    ]
    rel += [f"varmap_tables/{name}" for name in VARMAP_FILES]
    if hrrr:
        rel.append(f"am/{HRRR_GEOGRID}")
    return list(dict.fromkeys(rel))


def volcanic_files(years: list[int] | range) -> list[str]:
    """Decadal volcanic aerosol tables for `years` (available 1850-1999)."""
    decades = sorted({y - y % 10 for y in years if 1850 <= y <= 1999})
    return [f"volcanic_aerosols_{d}-{d + 9}.txt" for d in decades]


# --- Name mapping ----------------------------------------------------------------


def remote_names(rel: str) -> list[str]:
    """Candidate names, inside a NOAA version directory, for `rel`.

    The first element is the name of `rel` itself. For CO2 the observed
    record (fix_co2_update) takes precedence over the 4a set (co2dat_4a), and
    both over the projection (fix_co2_proj).
    """
    top, _, name = rel.partition("/")
    names = [name]

    if top == "am":
        if m := re.fullmatch(r"co2historicaldata_(\d{4})\.txt", name):
            g = f"global_co2historicaldata_{m.group(1)}.txt"
            names += [f"fix_co2_update/{g}", f"co2dat_4a/{g}", f"fix_co2_proj/{g}", g]
        elif name == "co2historicaldata_glob.txt":
            g = "global_co2historicaldata_glob.txt"
            names += [f"co2dat_4a/{g}", g]
        elif name == "aerosol.dat":
            names.append("global_climaeropac_global.txt")
        elif name == "global_h2oprdlos.f77":
            names.append("global_h2o_pltc.f77")
        elif not name.startswith("global_") and "/" not in name:
            names.append(f"global_{name}")
    elif top == "orog" and name in LAKE_FILES:
        names.append(name.replace(".dat", "_GLDBv2release.dat"))

    return list(dict.fromkeys(names))


# --- Remote access ---------------------------------------------------------------


def _open(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "pyUFS"})
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)


_VERSIONS: dict[str, tuple[str, ...]] = {}


def bucket_versions(top: str) -> tuple[str, ...]:
    """Version directories of fix/<top>/ in the bucket, newest first.

    A successful listing is cached for the process; a failed one is not, so a
    transient network error does not disable later lookups.
    """
    if top in _VERSIONS:
        return _VERSIONS[top]
    url = f"{BUCKET_URL}/?list-type=2&prefix={BUCKET_PREFIX}/{top}/&delimiter=/"
    try:
        with _open(url) as response:
            root = ET.fromstring(response.read())
    except (urllib.error.URLError, OSError, ET.ParseError) as exc:
        log.warning(f"Cannot list {url}: {exc}")
        return ()

    versions = []
    for prefix in root.findall("s3:CommonPrefixes/s3:Prefix", S3_NS):
        version = prefix.text.rstrip("/").rsplit("/", 1)[-1]
        if version.isdigit():
            versions.append(version)
    _VERSIONS[top] = tuple(sorted(versions, reverse=True))
    return _VERSIONS[top]


def remote_urls(rel: str) -> list[tuple[str, str]]:
    """(url, local name) pairs to try for `rel`, in order."""
    top, _, _ = rel.partition("/")
    if top == "varmap_tables":
        name = rel.partition("/")[2]
        return [(f"{VARMAP_URL}/{name}", name)]
    if top not in ("am", "orog", "sfc_climo", "mom6"):
        return []

    urls = []
    for version in bucket_versions(top):
        for name in remote_names(rel):
            urls.append((f"{BUCKET_URL}/{BUCKET_PREFIX}/{top}/{version}/{name}", name))
    return urls


def download(url: str, dest: Path) -> bool:
    """Download `url` to `dest` atomically. False when the object is absent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")

    for attempt in range(ATTEMPTS):
        try:
            with _open(url) as response, open(tmp, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            os.chmod(tmp, 0o644)
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if exc.code in (403, 404):
                return False
            log.warning(f"Download failed ({exc}); retrying: {url}")
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            log.warning(f"Download failed ({exc}); retrying: {url}")
        time.sleep(5 * (attempt + 1))
    return False


# --- Lookup ----------------------------------------------------------------------


def _link(link: Path, target: Path) -> None:
    """Create `link` -> `target` (relative); tolerate a concurrent creator."""
    rel_target = os.path.relpath(target, start=link.parent)
    try:
        link.symlink_to(rel_target)
    except FileExistsError:
        pass


def find_local(fix_src: Path, rel: str, link: bool = True) -> Path | None:
    """`rel` or its NOAA-named equivalent in fix_src; links the latter to
    `rel` unless link is false (then the equivalent itself is returned)."""
    path = fix_src / rel
    if path.exists():
        return path

    top = rel.partition("/")[0]
    for name in remote_names(rel)[1:]:
        candidate = fix_src / top / name
        if candidate.exists():
            if not link:
                return candidate
            if path.is_symlink():  # broken link
                path.unlink()
            _link(path, candidate)
            log.info(f"Linked {path} -> {candidate}")
            return path
    return None


def fetch_remote(fix_src: Path, rel: str) -> Path | None:
    """Download `rel` from its remote source into fix_src."""

    path = fix_src / rel
    top = rel.partition("/")[0]

    for url, name in remote_urls(rel):
        dest = fix_src / top / name
        try:
            found = download(url, dest)
        except PermissionError as exc:
            raise PermissionError(
                f"fix_src is not writable ({exc}); cannot store {rel}. Download "
                + f"{url} into {dest} or point fix_src at a writable copy."
            ) from exc
        if not found:
            continue
        log.info(f"Downloaded {url} -> {dest}")
        if dest != path:
            if path.is_symlink():
                path.unlink()
            _link(path, dest)
        return path
    return None


def ensure_fix_file(
    fix_src: Path | str, rel: str, required: bool = True, link: bool = True
) -> Path | None:
    """Return the local path of `rel`, fetching it when missing.

    Raises FileNotFoundError for a required file found neither locally nor
    remotely; returns None for an optional one. With link false nothing in
    fix_src is created (no compatibility link, no download).
    """
    fix_src = Path(fix_src)
    path = find_local(fix_src, rel, link=link)
    if path is None and link:
        path = fetch_remote(fix_src, rel)
    if path is not None:
        return path
    if required:
        raise FileNotFoundError(missing_message(fix_src, [rel]))
    return None


def ensure_fix_files(fix_src: Path | str, rels: list[str]) -> list[Path]:
    """Resolve every file in `rels`; raise once, naming all that are missing."""
    fix_src = Path(fix_src)
    resolved, missing = [], []
    for rel in rels:
        path = find_local(fix_src, rel) or fetch_remote(fix_src, rel)
        if path is None:
            missing.append(rel)
        else:
            resolved.append(path)
    if missing:
        raise FileNotFoundError(missing_message(fix_src, missing))
    return resolved


def missing_message(fix_src: Path, rels: list[str]) -> str:
    lines = [f"Required fix file(s) not found in {fix_src} or remotely:"]
    for rel in rels:
        hint = next((h for k, h in LOCAL_ONLY_HINTS.items() if rel.startswith(k)), None)
        source = hint or f"{BUCKET_URL}/index.html#{BUCKET_PREFIX}/{rel.split('/')[0]}/"
        lines.append(f"  - {fix_src / rel}  (source: {source})")

    return "\n".join(lines)


# --- Command line -----------------------------------------------------------------
