import yaml

from plugins.pamip import external_files
from plugins.pamip.config import (
    BASELINES,
    FIELDS,
    PAMIP_EXPERIMENTS,
    REGISTRY_FILE,
    SOM_RESTORE_DAYS,
    forcing_year,
)


def validate(*, state) -> None:
    """Check the external files, pamip_experiment, pamip_baseline and pamip_fields."""
    external_files.validate(state)
    experiment = state.pamip_experiment
    baseline = state.pamip_baseline
    fields = state.pamip_fields

    if experiment is None or experiment == "":
        state.pamip_experiment = None
        if baseline is not None or fields is not None:
            raise ValueError(
                "`pamip_baseline` and `pamip_fields` only apply to a PAMIP experiment: "
                + " 'set `pamip_experiment` (for example \"1.6\") or set both options to null'"
            )
        return

    if not isinstance(experiment, str):
        raise TypeError(
            f'`pamip_experiment` must be quoted text such as "1.6". Got {experiment!r}'
        )
    experiment = experiment.strip()
    if experiment not in PAMIP_EXPERIMENTS:
        raise ValueError(
            f"`pamip_experiment` must be one of {tuple(PAMIP_EXPERIMENTS)}. Got {experiment!r}"
        )
    registry = yaml.safe_load(REGISTRY_FILE.read_text())["experiments"]
    if experiment not in registry:
        raise ValueError(
            f"PAMIP experiment {experiment} has no SST and sea-ice files in {REGISTRY_FILE.name}."
        )
    if baseline is not None and baseline not in BASELINES:
        raise ValueError(
            f"`pamip_baseline` must be one of {BASELINES} or null. Got {baseline!r}"
        )
    if fields is not None and fields not in FIELDS:
        raise ValueError(
            f"`pamip_fields` must be one of {FIELDS} or null. Got {fields!r}"
        )
    state.pamip_experiment = experiment
    if baseline == "external":
        missing = [
            key
            for key in ("external_sst", "external_sic", "external_mask")
            if not state[key]
        ]
        if missing:
            raise ValueError(
                f"`pamip_baseline: external` requires all external files. Not set: {missing}"
            )


def _experiment_settings(
    state,
    open_grib,
    save_grib,
    ll_regrid,
    fill_nearest,
    load_grid,
    sfc_filename,
    to_fv3_grid,
    cp,
) -> dict[str, dict]:
    """Write the boundary files and return the namelist settings of the experiment."""
    from plugins.pamip.boundary import set_cold_start_surface, write_boundary_files

    fields, files = write_boundary_files(
        state=state,
        open_grib=open_grib,
        save_grib=save_grib,
        ll_regrid=ll_regrid,
        fill_nearest=fill_nearest,
    )
    if state.restart_no == 0:
        set_cold_start_surface(
            fields,
            state=state,
            fill_nearest=fill_nearest,
            sfc_filename=sfc_filename,
            to_fv3_grid=to_fv3_grid,
            load_grid=load_grid,
            cp=cp,
        )

    # Sea-ice mask and concentration follow the PAMIP FIXED climatologies at every
    # surface cycle; fsicl covers points already classified as ice (non-open-ocean
    # branch of the sfcsub merge). The SST is set by the slab ocean (som_mlm.F90) at
    # open-water points, which restores it toward the SST climatology (restore_method 1)
    # with time scale SOM_RESTORE_DAYS; with do_ocean the surface cycle no longer
    # overwrites it, and ftsfs only acts on ts_clim_iano, which method 1 does not use.
    # Beyond 60 degrees the slab needs maxlat and the quasi sea-ice albedo switched
    # off (som_mlm.F90).
    physics = {
        "use_ext_sst": False,
        "nstf_name": [0, 0, 1, 0, 5],
        "do_ocean": True,
        "disable_radiation_quasi_sea_ice": True,
    }
    year = forcing_year(state)
    if year is not None:
        # Fixed CO2 forcing, e.g. ictm = 20200 for the CO2 of year 2020
        physics["ictm"] = year * 10
    return {
        "namsfc": {"faiss": 0, "fsics": 0, "fsicl": 0, **files},
        "gfs_physics_nml": physics,
        "ocean_nml": {
            "ocean_option": "SOM",
            "mld_option": "obs",
            "mld_obs_ratio": 1.0,
            "restore_method": 1,
            "sst_restore_tscale": SOM_RESTORE_DAYS,
            "start_lat": -90.0,
            "end_lat": 90.0,
            "maxlat": 90.0,
        },
    }


def apply(
    *,
    state,
    restart: bool,
    patch_nml,
    update_fixed_files,
    open_grib,
    save_grib,
    ll_regrid,
    fill_nearest,
    load_grid,
    sfc_filename,
    to_fv3_grid,
    cp,
) -> None:
    """Stage the external files and the experiment, then patch the namelists and
    refresh the fixed files that depend on them."""
    if restart:
        return
    settings = external_files.stage(state)
    if state.pamip_experiment is not None:
        experiment = _experiment_settings(
            state,
            open_grib,
            save_grib,
            ll_regrid,
            fill_nearest,
            load_grid,
            sfc_filename,
            to_fv3_grid,
            cp,
        )
        for group, entries in experiment.items():
            settings.setdefault(group, {}).update(entries)
    patch_nml(settings)
    update_fixed_files()  # the CO2 table follows ictm in the patched namelist
