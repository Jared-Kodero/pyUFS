import os
import sys
from pathlib import Path
from typing import Literal

import dask
import dask.array as da
import eccodes
import numpy as np
import xarray as xr


def get_grib_codes(
    parameter: str | int,
    centre: Literal[
        "NCEP", "ECMWF", "UKMO", "MeteoFrance", "JMA", "CMC", "DWD", "CMA"
    ] = "NCEP",
) -> dict:
    """Return the GRIB2 codes of a parameter given its paramId, shortName or name.

    Parameters
    ----------
    parameter : str or int
        A paramId (e.g. ``10``), a shortName (e.g. ``"ws"``) or a long name
        (e.g. ``"Wind speed"``).
    centre : {"NCEP", "ECMWF", "UKMO", "MeteoFrance", "JMA", "CMC", "DWD", "CMA"}, optional
        Originating centre whose local parameter tables are used:

        - ``"NCEP"``: US National Weather Service, National Centers for
          Environmental Prediction (WMO centre 7)
        - ``"ECMWF"``: European Centre for Medium-Range Weather Forecasts (98)
        - ``"UKMO"``: UK Met Office, Exeter (74)
        - ``"MeteoFrance"``: Météo-France, Toulouse (85)
        - ``"JMA"``: Japan Meteorological Agency, Tokyo (34)
        - ``"CMC"``: Canadian Meteorological Centre, Montreal (54)
        - ``"DWD"``: Deutscher Wetterdienst, Offenbach (78)
        - ``"CMA"``: China Meteorological Administration, Beijing (38)

        Defaults to ``"NCEP"``.

    Returns
    -------
    dict
        ``discipline``, ``parameterCategory``, ``parameterNumber``, ``paramId``,
        ``shortName``, ``name`` and ``units``, ready to use as variable attrs.
    """

    centres = {
        "NCEP": 7,
        "ECMWF": 98,
        "UKMO": 74,
        "MeteoFrance": 85,
        "JMA": 34,
        "CMC": 54,
        "DWD": 78,
        "CMA": 38,
    }
    if centre not in centres:
        raise ValueError(
            f"Unknown centre {centre!r}; choose one of {', '.join(centres)}"
        )
    if isinstance(parameter, int) or str(parameter).isdigit():
        keys = ("paramId",)
    elif any(character.isspace() for character in str(parameter)):
        keys = ("name", "shortName")
    else:
        keys = ("shortName", "name")
    gid = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(gid, "centre", centres[centre])
        for key in keys:
            try:
                eccodes.codes_set(
                    gid, key, int(parameter) if key == "paramId" else str(parameter)
                )
                break
            except eccodes.CodesInternalError:
                continue
        else:
            raise ValueError(
                f"Unknown GRIB parameter {parameter!r} for centre {centre}"
            )
        return {
            key: eccodes.codes_get(gid, key)
            for key in (
                "discipline",
                "parameterCategory",
                "parameterNumber",
                "paramId",
                "shortName",
                "name",
                "units",
            )
        }
    finally:
        eccodes.codes_release(gid)


def open_grib(path: str | Path) -> xr.DataTree:
    """Lazily open a GRIB file as a DataTree with one node per level type and grid.

    Variables are named by ``shortName`` with dimensions
    ``(time, <typeOfLevel>, latitude, longitude)``, or
    ``(time, <typeOfLevel>, values)`` for non-rectangular grids.
    Layers use an integer coordinate with ``topLevel`` and ``bottomLevel``
    auxiliary coordinates; their identity is the pair of boundaries.
    Arrays are Dask-backed with one chunk per message; a message is decoded
    only when its chunk is computed. Slots with no message are set to NaN and are
    never written by ``save_grib``.
    GRIB2 multi-field messages are rejected: their fields cannot safely be
    reopened by independent byte offsets.

    Parameters
    ----------
    path : str or Path
        Path to the GRIB file to be opened.

    Returns
    -------
    xr.DataTree
        The lazy DataTree representation of the GRIB file.
    """

    nodes, grids, fields, spans = {}, {}, [], []
    with Path(path).open("rb") as file:
        while gid := eccodes.codes_grib_new_from_file(file, headers_only=True):
            try:
                offset = int(eccodes.codes_get(gid, "offset"))
                if spans and offset == spans[-1][0]:
                    raise ValueError("GRIB2 multi-field messages are not supported")
                if eccodes.codes_get(gid, "edition") == 2:
                    message, position, count = eccodes.codes_get_message(gid), 16, 0
                    while position < len(message) - 4:
                        count += message[position + 4] == 4
                        if count > 1:
                            raise ValueError(
                                "GRIB2 multi-field messages are not supported"
                            )
                        position += int.from_bytes(
                            message[position : position + 4], "big"
                        )
                grid = eccodes.codes_get(gid, "md5GridSection")
                if grid not in grids:
                    lat = eccodes.codes_get_array(gid, "latitudes")
                    lon = eccodes.codes_get_array(gid, "longitudes")
                    y, iy = np.unique(lat, return_inverse=True)
                    x, ix = np.unique(lon, return_inverse=True)
                    if y.size * x.size == lat.size == np.unique(iy * x.size + ix).size:
                        coords = {"latitude": y, "longitude": x}
                        grids[grid] = (coords, (y.size, x.size), (iy, ix))
                    else:
                        coords = {
                            "latitude": ("values", lat),
                            "longitude": ("values", lon),
                        }
                        grids[grid] = (coords, (lat.size,), (np.arange(lat.size),))
                level_type = eccodes.codes_get(gid, "typeOfLevel")
                name = eccodes.codes_get(gid, "shortName")
                # Climatological products (e.g. GRIB1 timeRangeIndicator 51)
                # have no computable validity time; use the reference time.
                # eccodes logs that failure, so mute it for this read only.
                with open(os.devnull, "w") as null:
                    eccodes.codes_context_set_logging(null)
                    try:
                        date = eccodes.codes_get(gid, "validityDate")
                        clock = eccodes.codes_get(gid, "validityTime")
                    except eccodes.CodesInternalError:
                        date = eccodes.codes_get(gid, "dataDate")
                        clock = eccodes.codes_get(gid, "dataTime")
                    finally:
                        eccodes.codes_context_set_logging(sys.__stderr__)
                date, clock = f"{date:08d}", f"{clock:04d}"
                time = np.datetime64(
                    f"{date[:4]}-{date[4:6]}-{date[6:]}T{clock[:2]}:{clock[2:]}", "ns"
                )
                layer = "layer" in level_type.lower() or (
                    eccodes.codes_is_defined(gid, "typeOfSecondFixedSurface")
                    and eccodes.codes_get(gid, "typeOfSecondFixedSurface", int) != 255
                )
                level = (
                    (
                        eccodes.codes_get(gid, "topLevel", float),
                        eccodes.codes_get(gid, "bottomLevel", float),
                    )
                    if layer
                    else eccodes.codes_get(gid, "level", float)
                )
                copy = 0
                while True:
                    node = level_type if copy == 0 else f"{level_type}_{copy}"
                    entry = nodes.setdefault(
                        node, {"grid": grid, "level_type": level_type, "variables": {}}
                    )
                    slots = entry["variables"].setdefault(name, {})
                    if entry["grid"] == grid and (time, level) not in slots:
                        break
                    if not slots:
                        del entry["variables"][name]
                    copy += 1
                slots[(time, level)] = len(fields)
                attrs = {
                    "units": eccodes.codes_get(gid, "units"),
                    "long_name": eccodes.codes_get(gid, "name"),
                }
                for key in (
                    "paramId",
                    "shortName",
                    "discipline",
                    "parameterCategory",
                    "parameterNumber",
                    "typeOfLevel",
                    "stepType",
                    "centre",
                    "edition",
                ):
                    if eccodes.codes_is_defined(gid, key):
                        try:
                            attrs[key] = eccodes.codes_get(gid, key)
                        except eccodes.CodesInternalError:
                            pass
                entry.setdefault("attrs", {}).setdefault(name, attrs)
                fields.append((node, name, time, level))
                spans.append(
                    (
                        int(eccodes.codes_get(gid, "offset")),
                        eccodes.codes_get(gid, "totalLength"),
                    )
                )
            finally:
                eccodes.codes_release(gid)
    if not fields:
        raise ValueError(f"No GRIB messages found in {path}")

    def decode(offset, length, shape, index):
        with Path(path).open("rb") as file:
            file.seek(offset)
            gid = eccodes.codes_new_from_message(file.read(length))
        try:
            eccodes.codes_set(gid, "missingValue", eccodes.CODES_MISSING_DOUBLE)
            values = eccodes.codes_get_values(gid)
        finally:
            eccodes.codes_release(gid)
        values[values == eccodes.CODES_MISSING_DOUBLE] = np.nan
        target = np.empty(shape, np.float32)
        target[index] = values
        return target[np.newaxis, np.newaxis]

    datasets = {}
    for node, entry in nodes.items():
        coords, shape, index = grids[entry["grid"]]
        keys = [key for slots in entry["variables"].values() for key in slots]
        times = np.unique([key[0] for key in keys])
        layer = isinstance(keys[0][1], tuple)
        levels = sorted({key[1] for key in keys})
        vertical = (
            {
                entry["level_type"]: np.arange(len(levels)),
                "topLevel": (entry["level_type"], [level[0] for level in levels]),
                "bottomLevel": (entry["level_type"], [level[1] for level in levels]),
            }
            if layer
            else {entry["level_type"]: levels}
        )
        dims = (
            "time",
            entry["level_type"],
            *(("values",) if len(shape) == 1 else ("latitude", "longitude")),
        )
        empty = da.full((1, 1, *shape), np.nan, dtype=np.float32)
        variables = {}
        for name, slots in entry["variables"].items():
            data = da.concatenate(
                [
                    da.concatenate(
                        [
                            da.from_delayed(
                                dask.delayed(decode)(
                                    *spans[slots[(t, l)]], shape, index
                                ),
                                (1, 1, *shape),
                                np.float32,
                            )
                            if (t, l) in slots
                            else empty
                            for l in levels
                        ],
                        axis=1,
                    )
                    for t in times
                ],
                axis=0,
            )
            variables[name] = (dims, data, entry["attrs"][name])
        datasets[node] = xr.Dataset(
            variables, coords={"time": times, **vertical, **coords}
        )
    tree = xr.DataTree.from_dict(datasets)
    tree.attrs["source"] = str(Path(path).resolve())
    tree.attrs["messages"] = {index: field for index, field in enumerate(fields)}
    return tree


def save_grib(
    tree: xr.DataTree, path: str | Path, template: str | Path | None = None
) -> None:
    """Write an xarray DataTree from open_grib back to GRIB using template messages.

    Only template messages whose node, variable, time, and level are still in
    the tree are written; removed ones are dropped. Fields are matched to
    template messages through ``tree.attrs["messages"]``, preserving the
    template's order. The template defaults to ``tree.attrs["source"]``.
    Layers are matched by both ``topLevel`` and ``bottomLevel`` coordinates.
    Unchanged fields are copied bit for bit, while edited fields are
    re-encoded with the template packing, where NaN values become missing.
    The spatial grid must be complete. Variables added to an existing node
    are appended: each field clones a template message of that node at the
    same time and takes its parameter from the variable's
    ``discipline/parameterCategory/parameterNumber`` or ``paramId`` attributes.
    All-NaN slots of added variables are skipped.

    Parameters
    ----------
    tree : xr.DataTree
        The DataTree structure containing GRIB data to be saved.
    path : str or Path
        The destination file path for the output GRIB file.
    template : str, Path, or None, optional
        Path to the template GRIB file. If None, defaults to
        ``tree.attrs["source"]``.

    Returns
    -------
    None
    """

    template = template or tree.attrs.get("source")
    if template is None:
        raise ValueError("No template specified and tree has no 'source' attribute")
    sources = {
        Path(template).resolve(),
        Path(tree.attrs.get("source", template)).resolve(),
    }
    if Path(path).resolve() in sources:
        raise ValueError("Output path must differ from the template and source paths")
    messages, written = tree.attrs.get("messages"), 0
    if not messages:
        raise ValueError("Tree has no 'messages' attribute from open_grib")
    parameter = ("discipline", "parameterCategory", "parameterNumber")
    known = {(node, name) for node, name, _, _ in messages.values()}
    added = [
        (node, name)
        for node in tree.children
        for name in tree[node].data_vars
        if (node, name) not in known
    ]
    for node, name in added:
        if node not in {entry[0] for entry in messages.values()}:
            raise ValueError(
                f"{node}/{name}: new variables must go in a node from the template"
            )
        attrs = tree[node][name].attrs
        if "units" not in attrs or not (
            all(k in attrs for k in parameter) or "paramId" in attrs
        ):
            raise ValueError(
                f"{node}/{name}: attrs need units and either paramId or "
                + f"{', '.join(parameter)}; use {__name__}.get_grib_codes to find them"
            )
    donors, used, positions = {}, {}, {}

    def values_for(dataset, field, gid, label):
        key = (label.split("/")[0], eccodes.codes_get(gid, "md5GridSection"))
        if key not in positions:
            lat = eccodes.codes_get_array(gid, "latitudes")
            lon = eccodes.codes_get_array(gid, "longitudes")
            if "values" in field.dims:
                if not (
                    np.array_equal(field["latitude"], lat)
                    and np.array_equal(field["longitude"], lon)
                ):
                    raise ValueError(f"{label}: grid differs from template")
                positions[key] = None
            else:
                iy = dataset.indexes["latitude"].get_indexer(lat)
                ix = dataset.indexes["longitude"].get_indexer(lon)
                if (iy < 0).any() or (ix < 0).any():
                    raise ValueError(f"{label}: spatial grid is incomplete")
                positions[key] = (iy, ix)
        if positions[key] is None:
            values = field.values
        else:
            values = field.transpose("latitude", "longitude").values[positions[key]]
        values = values.astype(float)
        if np.isinf(values).any():
            raise ValueError(f"{label}: values must be finite or NaN")
        return values

    def encode(gid, values):
        missing = np.isnan(values)
        marker = np.nextafter(np.nanmax(np.abs(values), initial=0.0), np.inf) + 1.0
        eccodes.codes_set(gid, "missingValue", marker)
        eccodes.codes_set(gid, "bitmapPresent", int(missing.any()))
        eccodes.codes_set_values(gid, np.where(missing, marker, values))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    partial = Path(path).with_name(Path(path).name + ".part")
    try:
        with Path(template).open("rb") as source, partial.open("wb") as output:
            index = -1
            while gid := eccodes.codes_grib_new_from_file(source):
                index += 1
                try:
                    if index not in messages:
                        continue
                    node, name, time, level = messages[index]
                    level_type = eccodes.codes_get(gid, "typeOfLevel")
                    template_level = (
                        (
                            eccodes.codes_get(gid, "topLevel", float),
                            eccodes.codes_get(gid, "bottomLevel", float),
                        )
                        if isinstance(level, tuple)
                        else eccodes.codes_get(gid, "level", float)
                    )
                    if (
                        eccodes.codes_get(gid, "shortName") != name
                        or template_level != level
                    ):
                        raise ValueError(
                            f"Template message {index} does not match {node}/{name}"
                        )
                    keys = (
                        parameter
                        if all(eccodes.codes_is_defined(gid, k) for k in parameter)
                        else ("paramId",)
                    )
                    used.setdefault(node, set()).add(
                        tuple((k, eccodes.codes_get(gid, k)) for k in keys)
                    )
                    if any(entry == node for entry, _ in added):
                        donors.setdefault((node, time), eccodes.codes_get_message(gid))
                    if node not in tree.children or name not in tree[node].data_vars:
                        continue
                    dataset = tree[node].to_dataset()
                    field = dataset[name]
                    for dim, key in (("time", time), (level_type, level)):
                        if isinstance(key, tuple):
                            if any(
                                k not in field.coords
                                for k in ("topLevel", "bottomLevel")
                            ):
                                raise ValueError(
                                    f"{node}/{name}: layer bounds are missing"
                                )
                            top, bottom = field["topLevel"], field["bottomLevel"]
                            if dim in field.dims:
                                if top.dims != (dim,) or bottom.dims != (dim,):
                                    raise ValueError(
                                        f"{node}/{name}: bounds must follow {dim}"
                                    )
                                matches = np.flatnonzero(
                                    (top.values == key[0]) & (bottom.values == key[1])
                                )
                                if matches.size > 1:
                                    raise ValueError(
                                        f"{node}/{name}: duplicate layer bounds {key}"
                                    )
                                field = (
                                    field.isel({dim: int(matches[0])})
                                    if matches.size
                                    else None
                                )
                            elif (
                                top.ndim
                                or bottom.ndim
                                or (top.item(), bottom.item()) != key
                            ):
                                field = None
                        elif dim in field.dims and key in dataset.indexes[dim]:
                            field = field.sel({dim: key})
                        elif (
                            dim in field.coords
                            and field[dim].ndim == 0
                            and field[dim].values != key
                        ):
                            field = None
                        if field is None or dim in field.dims:
                            field = None
                            break
                    if field is None:
                        continue
                    values = values_for(dataset, field, gid, f"{node}/{name}")
                    clone = eccodes.codes_clone(gid)
                    eccodes.codes_set(
                        clone, "missingValue", eccodes.CODES_MISSING_DOUBLE
                    )
                    original = eccodes.codes_get_values(clone)
                    eccodes.codes_release(clone)
                    original[original == eccodes.CODES_MISSING_DOUBLE] = np.nan
                    if not np.array_equal(
                        values.astype(np.float32),
                        original.astype(np.float32),
                        equal_nan=True,
                    ):
                        encode(gid, values)
                    eccodes.codes_write(gid, output)
                    written += 1
                finally:
                    eccodes.codes_release(gid)
            if max(messages) > index:
                raise ValueError(
                    "Template has fewer messages than the tree's index map"
                )

            for node, name in added:
                dataset = tree[node].to_dataset()
                array, label = dataset[name], f"{node}/{name}"
                keys = (
                    parameter
                    if all(k in array.attrs for k in parameter)
                    else ("paramId",)
                )
                codes = tuple((k, int(array.attrs[k])) for k in keys)
                if codes in used.get(node, set()):
                    raise ValueError(
                        f"{label}: attrs {dict(codes)} duplicate a template variable"
                    )
                level_type = next(
                    dim
                    for dim in dataset.indexes
                    if dim not in ("time", "latitude", "longitude")
                )
                times = np.atleast_1d(array["time"].values)
                levels = np.atleast_1d(array[level_type].values)
                for time in times:
                    for level in levels:
                        field = array
                        for dim, key in (("time", time), (level_type, level)):
                            if dim in field.dims:
                                field = field.sel({dim: key})
                        if bool(field.isnull().all()):
                            continue
                        if (node, time) not in donors:
                            raise ValueError(
                                f"{label}: template has no {node} message at {time}"
                            )
                        gid = eccodes.codes_new_from_message(donors[(node, time)])
                        try:
                            for key, value in codes:
                                eccodes.codes_set(gid, key, value)
                            layer = "layer" in level_type.lower() or (
                                eccodes.codes_is_defined(
                                    gid, "typeOfSecondFixedSurface"
                                )
                                and eccodes.codes_get(
                                    gid, "typeOfSecondFixedSurface", int
                                )
                                != 255
                            )
                            if layer:
                                if any(
                                    k not in field.coords
                                    for k in ("topLevel", "bottomLevel")
                                ):
                                    raise ValueError(
                                        f"{label}: layer bounds are missing"
                                    )
                                bounds = (
                                    field["topLevel"].item(),
                                    field["bottomLevel"].item(),
                                )
                                if not np.isfinite(bounds).all():
                                    raise ValueError(
                                        f"{label}: layer bounds must be finite"
                                    )
                                for key, bound in zip(
                                    ("topLevel", "bottomLevel"), bounds, strict=True
                                ):
                                    if eccodes.codes_get(gid, key, float) != bound:
                                        eccodes.codes_set(gid, key, float(bound))
                                encoded = [
                                    eccodes.codes_get(gid, key, float)
                                    for key in ("topLevel", "bottomLevel")
                                ]
                                if not np.allclose(encoded, bounds, rtol=1e-12, atol=0):
                                    raise ValueError(
                                        f"{label}: cannot encode layer bounds {bounds}"
                                    )
                            elif eccodes.codes_get(gid, "level", float) != level:
                                if level != int(level):
                                    raise ValueError(
                                        f"{label}: cannot encode non-integer level {level}"
                                    )
                                eccodes.codes_set(gid, "level", int(level))
                            encode(gid, values_for(dataset, field, gid, label))
                            eccodes.codes_write(gid, output)
                            written += 1
                        finally:
                            eccodes.codes_release(gid)
        if not written:
            raise ValueError("No template messages match the tree")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
