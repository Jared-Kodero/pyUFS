"""Download ERA5-Land skin temperature and ERA5 SST and save their 1991-2020 monthly climatologies.

    skin  ERA5-Land monthly means, skin_temperature, 0.1 degree, NaN over the ocean (doi:10.24381/cds.68d2bb30).
          Fills the land of the ESA SST set made by tools/esa_sst.py.
    sst   ERA5 monthly means on single levels, sea_surface_temperature, 0.25 degree, NaN over land
          (doi:10.24381/cds.f17050d7). Saved for comparison.

Needs ``pip install cdsapi``, a CDS account with the dataset licences accepted and ~/.cdsapirc.
Output, in ../fix/external/ERA5: monthly_skin_1991-2020.nc and monthly_sst_1991-2020.nc
(variables skt and sst, dimensions month, latitude, longitude, K). Existing files are not downloaded again.
"""

from pathlib import Path

import cdsapi
import xarray as xr

OUT = Path(__file__).resolve().parent.parent / "fix" / "external" / "ERA5"
YEARS = range(1991, 2021)
DATASETS = {  # key: (CDS dataset, CDS variable, variable in the file)
    "skin": ("reanalysis-era5-land-monthly-means", "skin_temperature", "skt"),
    "sst": (
        "reanalysis-era5-single-levels-monthly-means",
        "sea_surface_temperature",
        "sst",
    ),
}


def get_era5() -> dict[str, Path]:
    """Paths of the two climatologies; the missing ones are downloaded year by year and averaged."""
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (dataset, variable, name) in DATASETS.items():
        paths[key] = target = OUT / f"monthly_{key}_{YEARS[0]}-{YEARS[-1]}.nc"
        if target.exists():
            continue
        files = [OUT / f"{key}_{year}.nc" for year in YEARS]
        for year, file in zip(YEARS, files, strict=True):
            if not file.exists():
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
        with xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="year",
            preprocess=lambda ds: (
                ds.assign_coords(month=ds.valid_time.dt.month)
                .swap_dims(valid_time="month")
                .drop_vars(["valid_time", "number", "expver"], errors="ignore")
            ),
        ) as ds:
            mean = ds[name].mean("year").compute()
        mean.attrs = {"units": "K", "source": f"{dataset}, {YEARS[0]}-{YEARS[-1]}"}
        mean.to_dataset(name=name).to_netcdf(target)
        for file in files:
            file.unlink()
    return paths


if __name__ == "__main__":
    get_era5()
