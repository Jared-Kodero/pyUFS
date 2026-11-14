import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
for name in ("ESMF", "esmpy", "xesmf", "eccodes", "cfgrib", "gribapi"):
    sys.modules.setdefault(name, MagicMock())

import numpy as np
import xarray as xr
import yaml

import fv3_plugins
from fv3_runtime import open_nml
from fv3_state import compute_checksum, state
from plugins import pamip, sst_perturbation
from plugins.sst_perturbation.perturb import perturb_sst


@pytest.fixture(autouse=True)
def fixed_files_calls(monkeypatch):
    import fv3_fixed_files

    calls = []
    monkeypatch.setattr(fv3_fixed_files, "update_fixed_files", lambda: calls.append(1))
    return calls


@pytest.fixture(autouse=True)
def clean_state():
    keys = ("plugins", "case_dir", "run_dir", "fix", "n_nests")
    saved = {key: state[key] for key in keys}
    state.plugins = {}
    yield
    state.update(saved)


def set_plugins(tmp_path, mapping):
    """Load a plugins file with this content into state.plugins, as the cold start does."""
    path = tmp_path / "plugins.yaml"
    path.write_text(yaml.safe_dump(mapping))
    state.plugins = fv3_plugins.load_plugins(path)


def test_load_plugins_fills_defaults_and_drops_empty(tmp_path):
    set_plugins(tmp_path, {"pamip": {"experiment": "1.6"}, "sm_perturbations": None})
    assert state.plugins == {"pamip": {**pamip.DEFAULTS, "experiment": "1.6"}}


def test_load_plugins_rejects_unknown_names_and_keys(tmp_path):
    with pytest.raises(ValueError, match="Did you mean 'pamip'"):
        set_plugins(tmp_path, {"pamp": {"experiment": "1.6"}})
    with pytest.raises(ValueError, match="Did you mean 'experiment'"):
        set_plugins(tmp_path, {"pamip": {"experiement": "1.6"}})


def test_hashed_plugins(tmp_path):
    set_plugins(
        tmp_path,
        {
            "pamip": {"experiment": "1.6"},
            "sst_perturbation": {"delta": 3.0},
            "sm_perturbations": {"target_var": "smc"},
            "derived": {"module": "x.py"},
        },
    )
    assert set(fv3_plugins.hashed(state.plugins)) == {"pamip", "sst_perturbation"}


def test_checksum_changes_only_when_hashed_plugin_set(tmp_path):
    base = compute_checksum(state)
    set_plugins(tmp_path, {"sm_perturbations": {"target_var": "smc"}})
    assert compute_checksum(state) == base
    set_plugins(tmp_path, {"pamip": {"experiment": "1.6"}})
    assert compute_checksum(state) != base


def write_case(tmp_path):
    state.case_dir = tmp_path
    state.run_dir = tmp_path / "run"
    state.fix = tmp_path / "fix"
    state.n_nests = 0
    (tmp_path / "input.nml").write_text("&namsfc\n  fntsfc = 'FIXED/default.grb'\n/\n")


def test_external_files_patch_namsfc(tmp_path):
    files = {}
    for key in ("external_sst", "external_sic", "external_mask"):
        files[key] = tmp_path / f"{key}.grb"
        files[key].write_bytes(b"x")
    set_plugins(tmp_path, {"pamip": {key: str(f) for key, f in files.items()}})
    write_case(tmp_path)
    fv3_plugins.validate()
    fv3_plugins.do_plugins(restart=False)
    assert open_nml(tmp_path / "input.nml")["namsfc"] == {
        "fntsfc": "FIXED/external/external_sst.grb",
        "fnaisc": "FIXED/external/external_sic.grb",
        "fnmskh": "FIXED/external/external_mask.grb",
    }
    assert (tmp_path / "fix/external/external_sst.grb").is_file()


def test_user_override_wins_over_plugin(tmp_path):
    sst = tmp_path / "sst.grb"
    sst.write_bytes(b"x")
    set_plugins(tmp_path, {"pamip": {"external_sst": str(sst)}})
    write_case(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run/input.yaml").write_text("namsfc:\n  fntsfc: FIXED/user.grb\n")
    fv3_plugins.do_plugins(restart=False)
    assert open_nml(tmp_path / "input.nml")["namsfc"]["fntsfc"] == "FIXED/user.grb"


def test_restart_does_not_validate_or_patch(tmp_path):
    sst = tmp_path / "sst.grb"
    sst.write_bytes(b"x")
    set_plugins(tmp_path, {"pamip": {"external_sst": str(sst)}})
    write_case(tmp_path)
    sst.unlink()
    fv3_plugins.do_plugins(restart=True)
    assert open_nml(tmp_path / "input.nml")["namsfc"]["fntsfc"] == "FIXED/default.grb"
    with pytest.raises(FileNotFoundError):
        fv3_plugins.validate()


def test_pamip_rejects_baseline_without_experiment(tmp_path):
    set_plugins(tmp_path, {"pamip": {"baseline": "shield"}})
    with pytest.raises(ValueError):
        fv3_plugins.validate()


def test_only_pamip_refreshes_fixed_files(tmp_path, fixed_files_calls):
    write_case(tmp_path)
    fv3_plugins.do_plugins(restart=False)
    assert fixed_files_calls == []
    (tmp_path / "sst.grb").write_bytes(b"x")
    set_plugins(tmp_path, {"pamip": {"external_sst": str(tmp_path / "sst.grb")}})
    fv3_plugins.do_plugins(restart=False)
    assert fixed_files_calls == [1]


def test_validate_normalises_hashed_setting(tmp_path):
    set_plugins(tmp_path, {"pamip": {"experiment": ""}})
    before = compute_checksum(state)
    pamip.validate(config=state.plugins["pamip"])
    assert compute_checksum(state) != before  # hash after validate, as init_driver does


def test_inactive_plugins_import_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "grib_io", None)
    monkeypatch.setitem(sys.modules, "latlon_to_fv3", None)
    fv3_plugins.do_plugins(restart=False)
    fv3_plugins.do_plugins(restart=True)


def test_user_derived_module(tmp_path):
    user = tmp_path / "mine.py"
    user.write_text(
        "def double(ds, *, state, variables, **_):\n"
        "    ds['x2'] = 2 * ds['x']\n"
        "    return ds\n\n\nderive = [double]\n"
    )
    set_plugins(tmp_path, {"derived": {"module": str(user)}})
    state.case_dir = tmp_path
    fv3_plugins.validate()
    fv3_plugins.do_plugins(restart=False)
    assert (tmp_path / "derived_vars_user.py").is_file()
    ds = fv3_plugins.add_derived_variables(xr.Dataset({"x": ("t", [1.0, 2.0])}))
    assert ds["x2"].values.tolist() == [2.0, 4.0]


def test_sst_perturbation_validate(tmp_path):
    bad = (
        {"taper": 1.0},
        {"delta": "3"},
        {"delta": 1, "months": [13]},
        {"delta": 1, "lat": [80, 50]},
        {"delta": 1, "lon": [0, 400]},
        {"delta": 1, "taper": -1},
    )
    for settings in bad:
        set_plugins(tmp_path, {"sst_perturbation": settings})
        with pytest.raises((TypeError, ValueError)):
            sst_perturbation.validate(config=state.plugins["sst_perturbation"])
    set_plugins(tmp_path, {"sst_perturbation": {"delta": 3}})
    sst_perturbation.validate(config=state.plugins["sst_perturbation"])


def grid():
    lat, lon = (
        np.array([-60.0, 55.0, 65.0, 85.0]),
        np.array([10.0, 100.0, 190.0, 350.0]),
    )
    sst = xr.DataArray(
        np.full((12, 4, 4), 280.0),
        coords={"month": np.arange(1, 13), "lat": lat, "lon": lon},
        dims=("month", "lat", "lon"),
    )
    land = xr.zeros_like(sst.isel(month=0, drop=True))
    return sst, xr.zeros_like(sst), land


def test_perturb_sst_box_months_and_masks():
    sst, sic, land = grid()
    sic[:, 1, 1] = 0.5  # ice at 55N, 100E
    land[2, 0] = 1.0  # land at 65N, 10E
    out = perturb_sst(
        sst, sic, land, delta=3.0, months=[6, 7], lat=[50, 80], lon=None, taper=5.0
    )
    assert out.dims == sst.dims
    assert (out.sel(month=[1, 5, 8, 12]) == 280.0).all()
    assert (out.sel(month=6, lat=[-60.0, 85.0]) == 280.0).all()
    assert out.sel(month=6, lat=65.0, lon=190.0) == 283.0
    assert out.sel(month=6, lat=55.0, lon=10.0) == 283.0
    assert out.sel(month=6, lat=55.0, lon=100.0) == 280.0
    assert out.sel(month=6, lat=65.0, lon=10.0) == 280.0


def test_perturb_sst_taper_and_wrapping_longitude():
    sst, sic, land = grid()
    out = perturb_sst(
        sst, sic, land, delta=2.0, months=[6], lat=[50, 80], lon=None, taper=10.0
    )
    assert float(out.sel(month=6, lat=55.0, lon=10.0)) == pytest.approx(281.0)
    assert float(out.sel(month=6, lat=65.0, lon=10.0)) == pytest.approx(282.0)
    wrapped = perturb_sst(
        sst, sic, land, delta=2.0, months=[6], lat=[50, 80], lon=[300, 20], taper=0.0
    )
    assert float(wrapped.sel(month=6, lat=65.0, lon=350.0)) == 282.0
    assert float(wrapped.sel(month=6, lat=65.0, lon=10.0)) == 282.0
    assert float(wrapped.sel(month=6, lat=65.0, lon=100.0)) == 280.0


def test_sst_perturbation_patches_namsfc_on_cold_start_only(tmp_path, monkeypatch):
    from plugins.sst_perturbation import perturb

    monkeypatch.setattr(
        perturb, "write_perturbed_sst", lambda *a, **k: "FIXED/x.sstpert.grb"
    )
    set_plugins(tmp_path, {"sst_perturbation": {"delta": 3.0}})
    write_case(tmp_path)
    fv3_plugins.validate()
    fv3_plugins.do_plugins(restart=True)
    assert open_nml(tmp_path / "input.nml")["namsfc"]["fntsfc"] == "FIXED/default.grb"
    fv3_plugins.do_plugins(restart=False)
    assert open_nml(tmp_path / "input.nml")["namsfc"]["fntsfc"] == "FIXED/x.sstpert.grb"
