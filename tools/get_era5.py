"""Download ERA5-Land skin temperature and ERA5 SST 1991-2020 monthly climatologies."""

import getpass
import logging
import tempfile
from pathlib import Path

import cdsapi
import xarray as xr

OUT = Path(__file__).resolve().parent.parent / "fix" / "external" / "ERA5"
TMP = Path(tempfile.gettempdir()) / getpass.getuser()
YEARS = range(1991, 2021)
DATASETS = {
    "skin": ("reanalysis-era5-land-monthly-means", "skin_temperature", "skt"),
    "sst": (
        "reanalysis-era5-single-levels-monthly-means",
        "sea_surface_temperature",
        "sst",
    ),
}

log = logging.getLogger(__name__)


def get_era5() -> dict[str, Path]:
    """Return climatology paths, downloading and averaging only missing outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key, (dataset, variable, name) in DATASETS.items():
        target = OUT / f"monthly_{key}_{YEARS[0]}-{YEARS[-1]}.nc"
        paths[key] = target

        if target.exists():
            log.info("[%s] Existing climatology: %s", key, target)
            continue

        log.info("[%s] Starting %d-%d climatology", key, YEARS[0], YEARS[-1])
        files = [TMP / f"{key}_{year}.nc" for year in YEARS]

        for i, (year, file) in enumerate(zip(YEARS, files, strict=True), 1):
            if file.exists():
                log.info("[%s] %d/%d: %d cached", key, i, len(YEARS), year)
                continue

            log.info("[%s] %d/%d: Downloading %d", key, i, len(YEARS), year)
            request = {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [variable],
                "year": [str(year)],
                "month": [f"{month:02d}" for month in range(1, 13)],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            cdsapi.Client().retrieve(dataset, request, f"{file}.part")
            Path(f"{file}.part").rename(file)

        log.info("[%s] Computing monthly climatology", key)
        with xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="year",
            preprocess=lambda ds: (
                ds.assign_coords(month=ds.valid_time.dt.month)
                .swap_dims(valid_time="month")
                .drop_vars(["valid_time", "number", "expver"], errors="ignore")
            ),
            chunks="auto",
        ) as ds:
            mean = ds[name].mean("year").compute()

        mean.attrs = {"units": "K", "source": f"{dataset}, {YEARS[0]}-{YEARS[-1]}"}
        mean.to_dataset(name=name).to_netcdf(target)
        log.info("[%s] Saved: %s", key, target)

        for file in files:
            file.unlink()
        log.info("[%s] Temporary files removed", key)

    return paths


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    get_era5()
