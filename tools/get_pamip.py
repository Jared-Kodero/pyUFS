"""Download the PAMIP SST and sea-ice files that the experiments of templates/pamip_experiments.yaml use.

Files are stored, with their original names, in <root>/external/PAMIP/<source_id>/<variable_id>/ where
<root> is the fix_src directory of the case. Existing files are not downloaded again.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import intake_esgf
import yaml
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults

PROJECTS = (
    "input4MIPs",
    "input4mips",
)  # the ESGF index spells some datasets in lower case
REGISTRY = (
    Path(__file__).resolve().parent.parent / "templates" / "pamip_experiments.yaml"
)


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


def needed(experiments: list[str] | None) -> list[tuple[str, str, str]]:
    """Source, variable and file name of the PAMIP files that the registry experiments use."""
    registry = {
        str(k): v
        for k, v in yaml.safe_load(REGISTRY.read_text())["experiments"].items()
    }
    keys = [str(e) for e in experiments] if experiments else list(registry)
    unknown = [k for k in keys if k not in registry]
    if unknown:
        raise SystemExit(f"Unknown experiments {unknown}; choose from {list(registry)}")
    pairs = set()
    for key in keys:
        for part in ("sst", "sic"):
            spec = registry[key][part]
            name = f"{spec['variable_id']}_input4MIPs_SSTsAndSeaIce_PAMIP_{spec['source_id']}_gn_{spec['period']}.nc"
            pairs.add((spec["source_id"], spec["variable_id"], name))
    return sorted(pairs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="fix_src directory of the case (fix_src in run_config.yaml); files go to <root>/external/PAMIP",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="experiment number (repeatable; default: all experiments of the registry)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the identifiers and exit"
    )
    args = parser.parse_args(argv)
    directory = args.root.resolve() / "external" / "PAMIP"
    intake_esgf.conf.set(all_indices=True, confirm_download=False)
    intake_esgf.conf.set(local_cache=str(directory / "_cache"))
    failures = 0
    for source, variable, name in needed(args.experiment):
        if args.list:
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
                output = directory / source / variable / path.name
                print(f"{store(path, output).upper():10} {output}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"ERROR {source}/{variable}: {exc}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
