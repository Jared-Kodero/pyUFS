#!/usr/bin/env python3
"""Download all discoverable PAMIP and PCMDI-AMIP SST/sea-ice forcing files."""

import csv
import os
import shutil
from pathlib import Path

import intake_esgf
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults

PAMIP_VARIABLES = ("tos", "siconc", "sithick")
AMIP_VARIABLES = ("tos", "tosbcs", "siconc")
AMIP_ERAS = ("CMIP6", "CMIP6Plus", "CMIP7", "CMIP7Plus")
FIELDS = ("mip_era", "source_id", "variable_id", "status", "file", "detail")


def search(era: str, target: str, variables: tuple[str, ...], amip: bool):
    """Discover archived SST/sea-ice datasets without assuming source IDs."""
    filters = {
        "project": "input4MIPs",
        "mip_era": era,
        "target_mip": target,
        "variable_id": list(variables),
    }
    if amip:
        filters["institution_id"] = "PCMDI"

    catalog = ESGFCatalog().search(**filters, quiet=True)
    pairs = catalog.df[["source_id", "variable_id"]].drop_duplicates()
    return sorted(
        (str(source), str(variable))
        for source, variable in pairs.itertuples(index=False, name=None)
        if not amip or str(source).startswith("PCMDI-AMIP-")
    )


def download(era: str, target: str, source: str, variable: str) -> list[Path]:
    """Return verified local paths, downloading the dataset if necessary."""
    catalog = ESGFCatalog().search(
        project="input4MIPs",
        mip_era=era,
        target_mip=target,
        source_id=source,
        variable_id=variable,
        quiet=True,
    )
    paths = {
        Path(path) for entries in catalog.to_path_dict().values() for path in entries
    }
    if not paths:
        raise FileNotFoundError(f"No downloadable files for {source}/{variable}")
    return sorted(paths)


def store(source: Path, destination: Path) -> str:
    """Keep the original NetCDF filename and avoid duplicate cache storage."""
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return "existing"
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return "downloaded"


def process(group: str, root: Path, list_only: bool) -> int:
    """Search all relevant eras and write one manifest per data family."""
    directory = root / group
    directory.mkdir(parents=True, exist_ok=True)
    intake_esgf.conf.set(local_cache=str(directory / "_cache"))

    eras = ("CMIP6",) if group == "PAMIP" else AMIP_ERAS
    target = "PAMIP" if group == "PAMIP" else "CMIP"
    variables = PAMIP_VARIABLES if group == "PAMIP" else AMIP_VARIABLES
    failures = 0
    found = 0

    with (directory / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for era in eras:
            try:
                pairs = search(era, target, variables, amip=group == "AMIP")
            except NoSearchResults:
                pairs = []
            except Exception as exc:
                failures += 1
                writer.writerow(
                    {"mip_era": era, "status": "search_error", "detail": str(exc)}
                )
                print(f"ERROR {group}/{era} search: {exc}", flush=True)
                continue

            if not pairs:
                writer.writerow({"mip_era": era, "status": "not_indexed"})
                print(f"NONE  {group}/{era}: no matching datasets indexed", flush=True)
                continue

            for source, variable in pairs:
                found += 1
                entry = {"mip_era": era, "source_id": source, "variable_id": variable}
                if list_only:
                    writer.writerow({**entry, "status": "indexed"})
                    print(f"FOUND {group}/{era}/{source}/{variable}", flush=True)
                    continue

                destination = (
                    directory / (era if group == "AMIP" else "") / source / variable
                )
                try:
                    for path in download(era, target, source, variable):
                        output = destination / path.name
                        status = store(path, output)
                        writer.writerow(
                            {
                                **entry,
                                "status": status,
                                "file": str(output.relative_to(root)),
                            }
                        )
                        print(f"{status.upper():10} {output}", flush=True)
                except Exception as exc:
                    failures += 1
                    writer.writerow(
                        {**entry, "status": "download_error", "detail": str(exc)}
                    )
                    print(
                        f"ERROR {group}/{era}/{source}/{variable}: {exc}",
                        flush=True,
                    )

    if not found:
        failures += 1
    print(f"{group}: {found} source-variable datasets; {failures} errors", flush=True)
    return failures


def main() -> None:

    root = Path("../fix").resolve()
    intake_esgf.conf.set(all_indices=True, confirm_download=False)
    groups = ("PAMIP", "AMIP")
    errors = sum(process(group, root, False) for group in groups)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
