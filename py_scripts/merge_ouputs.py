import logging
from pathlib import Path

import xarray as xr

log = logging.getLogger("merge_outputs")


def stream_family(stream: str) -> tuple:
    stream_file = str(Path(stream).name)
    restart_id = stream_file.split("_r")[-1]
    fam = stream_file.replace(f"_r{restart_id}", "")
    return fam, int(restart_id)


def get_group_name(p: Path) -> str | None:
    name = p.name
    if ".global.nc" in name:
        return "global"
    for part in name.split("."):
        if part.startswith("tile"):
            return part  # tile7, tile8, ...
    return None


def merge_outputs(
    output_dir: Path, streams: list, n_nests: int, run_nhours: int, total_restarts: int
) -> None:

    for stream in streams:
        handle, _ = stream_family(stream)

        # include ALL files (not just tile*)
        candidates = list(Path(output_dir).glob(f"{handle}*.nc"))
        if not candidates:
            continue

        # discover groups automatically: global, tile7, tile8, ...
        groups = sorted({g for p in candidates if (g := get_group_name(p))})
        # name mape where global -> global, tile7 -> nest02, tile8 -> nest03, ...
        name_map = {"global": "global"}
        for i in range(1, n_nests + 1):
            name_map[f"tile{6 + i}"] = f"nest{i + 1:02d}"

        for group in groups:
            baseline = output_dir / f"{handle}.{name_map[group]}.nc"

            restart_files = sorted(
                [p for p in candidates if get_group_name(p) == group]
            )

            if not restart_files:
                continue

            inputs = []
            tmp = None

            # stage existing merged file if present
            if baseline.exists():
                tmp = baseline.with_suffix(".tmp.nc")
                baseline.rename(tmp)
                inputs.append(tmp)

            inputs.extend(restart_files)

            if len(inputs) > 1:
                ds = xr.open_mfdataset(
                    [str(p) for p in inputs],
                    combine="nested",
                    concat_dim="time",
                    coords="minimal",
                    data_vars="minimal",
                    compat="override",
                    join="exact",
                    decode_times=True,
                )

                if "time" in ds.coords:
                    ds = ds.sortby("time")
            else:
                ds = xr.open_dataset(inputs[0], decode_times=True)

            encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
            ds.to_netcdf(baseline, format="NETCDF4", encoding=encoding)
            ds.close()

            # cleanup
            if tmp and tmp.exists():
                tmp.unlink()

            for p in restart_files:
                p.unlink(missing_ok=True)
