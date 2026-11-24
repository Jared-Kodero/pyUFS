"""Download all discoverable PCMDI-AMIP SST and sea-ice forcing files (tos, tosbcs, siconc).

Files are stored, with their original names, in <root>/external/AMIP/<mip_era>/<source_id>/<variable_id>/
where <root> is the fix_src directory of the case, and listed in <root>/external/AMIP/manifest.csv.
"""

import argparse
import csv
from pathlib import Path

import intake_esgf
from get_pamip import PROJECTS, download, store
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults

VARIABLES = ("tos", "tosbcs", "siconc")
ERAS = ("CMIP6", "CMIP6Plus", "CMIP7", "CMIP7Plus")
FIELDS = ("mip_era", "source_id", "variable_id", "status", "file", "detail")


def search(era: str) -> list[tuple[str, str]]:
    """Source and variable of the PCMDI-AMIP datasets indexed for a MIP era."""
    found = set()
    for project in PROJECTS:  # one project name per search
        try:
            catalog = ESGFCatalog().search(
                project=project,
                mip_era=era,
                target_mip="CMIP",
                variable_id=list(VARIABLES),
                institution_id="PCMDI",
                quiet=True,
            )
        except NoSearchResults:
            continue
        pairs = catalog.df[["source_id", "variable_id"]].drop_duplicates()
        found.update(
            (str(source), str(variable))
            for source, variable in pairs.itertuples(index=False, name=None)
            if str(source).startswith("PCMDI-AMIP-")
        )
    return sorted(found)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="fix_src directory of the case (fix_src in run_config.yaml); files go to <root>/external/AMIP",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve() / "external"
    directory = root / "AMIP"
    directory.mkdir(parents=True, exist_ok=True)
    intake_esgf.conf.set(all_indices=True, confirm_download=False)
    intake_esgf.conf.set(local_cache=str(directory / "_cache"))
    failures = found = 0
    with (directory / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for era in ERAS:
            try:
                pairs = search(era)
            except Exception as exc:
                failures += 1
                writer.writerow(
                    {"mip_era": era, "status": "search_error", "detail": str(exc)}
                )
                print(f"ERROR AMIP/{era} search: {exc}", flush=True)
                continue
            if not pairs:
                writer.writerow({"mip_era": era, "status": "not_indexed"})
                print(f"NONE  AMIP/{era}: no matching datasets indexed", flush=True)
                continue
            for source, variable in pairs:
                found += 1
                entry = {"mip_era": era, "source_id": source, "variable_id": variable}
                try:
                    for path in download(era, "CMIP", source, variable):
                        output = directory / era / source / variable / path.name
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
                    print(f"ERROR AMIP/{era}/{source}/{variable}: {exc}", flush=True)
    failures += not found
    print(f"AMIP: {found} source-variable datasets; {failures} errors", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
