import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import xarray as xr

BASE = "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/Climatology/L4/v3.0.1/"
ROOT = Path("/tmp/esa_sst_cci")
ROOT.mkdir(parents=True, exist_ok=True)

files = [
    ROOT
    / f"D{day:03d}-ESACCI-L4_GHRSST-SSTdepth-Climatology-GLOB_CDR3.0-v02.0-fv01.0.nc"
    for day in range(1, 366)
]


def download(path: Path) -> Path:
    if path.exists():
        path.unlink()

    token = os.getenv("CEDA_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    with requests.get(
        BASE + path.name,
        headers=headers,
        stream=True,
        timeout=(30, 180),
    ) as response:
        response.raise_for_status()

        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError(f"Unexpected HTML response: {path.name}")

        temp = path.with_suffix(".part")
        with temp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        temp.replace(path)

    return path


with ThreadPoolExecutor(max_workers=8) as pool:
    for i, path in enumerate(pool.map(download, files), 1):
        print(f"{i}/365: {path.name}")
ds = xr.open_mfdataset(files)

ds = ds.groupby("time.month").mean("time", keep_attrs=True)
ds.attrs["history"] = (
    ds.attrs.get("history", "") + "; Grouped by month and averaged over time"
)
