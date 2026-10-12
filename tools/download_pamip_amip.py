#!/usr/bin/env python3
"""Download all discoverable PAMIP and PCMDI-AMIP SST/sea-ice forcing files."""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import yaml

import intake_esgf
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults

PAMIP_VARIABLES = ("tos", "siconc", "sithick")
AMIP_VARIABLES = ("tos", "tosbcs", "siconc")
AMIP_ERAS = ("CMIP6", "CMIP6Plus", "CMIP7", "CMIP7Plus")
PROJECTS = (
    "input4MIPs",
    "input4mips",
)  # the ESGF index spells some datasets in lower case
REGISTRY = (
    Path(__file__).resolve().parent.parent / "templates" / "pamip_experiments.yaml"
)
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

    found = set()
    for project in PROJECTS:  # one project name per search
        try:
            catalog = ESGFCatalog().search(
                **{**filters, "project": project}, quiet=True
            )
        except NoSearchResults:
            continue
        pairs = catalog.df[["source_id", "variable_id"]].drop_duplicates()
        found.update(
            (str(source), str(variable))
            for source, variable in pairs.itertuples(index=False, name=None)
            if not amip or str(source).startswith("PCMDI-AMIP-")
        )
    if not found:
        raise NoSearchResults()
    return sorted(found)


def download(era: str, target: str, source: str, variable: str) -> list[Path]:
    """Return verified local paths, downloading the dataset if necessary."""
    catalog = None
    for project in PROJECTS:
        try:
            catalog = ESGFCatalog().search(
                project=project,
                mip_era=era,
                target_mip=target,
                source_id=source,
                variable_id=variable,
                quiet=True,
            )
            break
        except NoSearchResults:
            continue
    if catalog is None:
        raise FileNotFoundError(f"No dataset {source}/{variable} in the index")
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


def needed(experiments: list[str] | None) -> list[tuple[str, str, str]]:
    """Source, variable and file name of the PAMIP files that the registry experiments use."""
    registry = yaml.safe_load(REGISTRY.read_text())["experiments"]
    keys = [str(e) for e in experiments] if experiments else [str(k) for k in registry]
    unknown = [k for k in keys if k not in {str(r) for r in registry}]
    if unknown:
        raise SystemExit(
            f"Unknown experiments {unknown}; choose from {[str(r) for r in registry]}"
        )
    pairs = set()
    for key in keys:
        entry = {str(k): v for k, v in registry.items()}[key]
        for part in ("sst", "sic"):
            spec = entry[part]
            name = f"{spec['variable_id']}_input4MIPs_SSTsAndSeaIce_PAMIP_{spec['source_id']}_gn_{spec['period']}.nc"
            pairs.add((spec["source_id"], spec["variable_id"], name))
    return sorted(pairs)


def download_needed(root: Path, experiments: list[str] | None, list_only: bool) -> int:
    """Download (or list) only the datasets of the selected PAMIP experiments."""
    directory = root / "PAMIP"
    directory.mkdir(parents=True, exist_ok=True)
    intake_esgf.conf.set(local_cache=str(directory / "_cache"))
    failures = 0
    for source, variable, name in needed(experiments):
        if list_only:
            print(f"{source}/{variable}/{name}")
            continue
        try:
            paths = [
                p
                for p in download("CMIP6", "PAMIP", source, variable)
                if p.name == name
            ]
            if not paths:  # e.g. UCI-present-1-0 also holds 1979-2014 siconc files
                raise FileNotFoundError(f"{name} is not in the index")
            for path in paths:
                print(
                    f"{store(path, directory / source / variable / path.name).upper():10} {directory / source / variable / path.name}",
                    flush=True,
                )
        except Exception as exc:
            failures += 1
            print(f"ERROR {source}/{variable}: {exc}", file=sys.stderr, flush=True)
    return failures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="fix_src directory of the case (fix_src in run_config.yaml) that receives PAMIP/ and AMIP/",
    )
    parser.add_argument(
        "--needed",
        action="store_true",
        help="download only the files of the PAMIP experiments in templates/pamip_experiments.yaml",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="with --needed: experiment number (repeatable)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="with --needed: print the identifiers and exit",
    )
    args = parser.parse_args(argv)
    if (args.list or args.experiment) and not args.needed:
        parser.error("--list and --experiment require --needed")
    root = args.root.resolve()
    intake_esgf.conf.set(all_indices=True, confirm_download=False)
    if args.needed:
        errors = download_needed(root, args.experiment, args.list)
    else:
        errors = sum(process(group, root, False) for group in ("PAMIP", "AMIP"))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
