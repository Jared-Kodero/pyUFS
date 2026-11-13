import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
for name in ("ESMF", "esmpy", "xesmf", "eccodes", "cfgrib", "gribapi"):
    sys.modules.setdefault(name, MagicMock())

import fv3_plugins
from fv3_runtime import open_nml
from fv3_state import compute_checksum, state
from plugins import pamip


@pytest.fixture(autouse=True)
def fixed_files_calls(monkeypatch):
    import fv3_fixed_files

    calls = []
    monkeypatch.setattr(fv3_fixed_files, "update_fixed_files", lambda: calls.append(1))
    return calls


@pytest.fixture(autouse=True)
def clean_state():
    keys = (
        "pamip_experiment",
        "pamip_baseline",
        "pamip_fields",
        "pamip_forcing_year",
        "external_sst",
        "external_sic",
        "external_mask",
        "sm_perturbations",
        "derived_vars_module",
        "case_dir",
        "run_dir",
        "fix",
        "n_nests",
    )
    saved = {key: state[key] for key in keys}
    yield
    state.update(saved)


def test_hash_keys_follow_manifest():
    assert fv3_plugins.hash_keys() == [
        "pamip_experiment",
        "pamip_baseline",
        "pamip_fields",
        "pamip_forcing_year",
    ]


def test_checksum_changes_only_when_plugin_key_set():
    base = compute_checksum(state)
    state.pamip_experiment = "1.6"
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
        state[key] = str(files[key])
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
    state.external_sst = str(sst)
    write_case(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run/input.yaml").write_text("namsfc:\n  fntsfc: FIXED/user.grb\n")
    fv3_plugins.do_plugins(restart=False)
    assert open_nml(tmp_path / "input.nml")["namsfc"]["fntsfc"] == "FIXED/user.grb"


def test_restart_does_not_validate_or_patch(tmp_path):
    sst = tmp_path / "sst.grb"
    sst.write_bytes(b"x")
    state.external_sst = str(sst)
    write_case(tmp_path)
    sst.unlink()
    fv3_plugins.do_plugins(restart=True)
    assert open_nml(tmp_path / "input.nml")["namsfc"]["fntsfc"] == "FIXED/default.grb"
    with pytest.raises(FileNotFoundError):
        fv3_plugins.validate()


def test_pamip_rejects_baseline_without_experiment():
    state.pamip_baseline = "shield"
    with pytest.raises(ValueError):
        fv3_plugins.validate()


def test_only_pamip_refreshes_fixed_files(tmp_path, fixed_files_calls):
    write_case(tmp_path)
    fv3_plugins.do_plugins(restart=False)
    assert fixed_files_calls == []
    state.external_sst = str(tmp_path / "sst.grb")
    (tmp_path / "sst.grb").write_bytes(b"x")
    fv3_plugins.do_plugins(restart=False)
    assert fixed_files_calls == [1]


def test_validate_normalises_hashed_key():
    state.pamip_experiment = ""
    before = compute_checksum(state)
    pamip.validate(state=state)
    assert compute_checksum(state) != before  # hash after validate, as init_driver does


def test_inactive_plugins_import_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "grib_io", None)
    monkeypatch.setitem(sys.modules, "latlon_to_fv3", None)
    fv3_plugins.do_plugins(restart=False)
    fv3_plugins.do_plugins(restart=True)


def test_user_derived_module(tmp_path):
    import xarray as xr

    user = tmp_path / "mine.py"
    user.write_text(
        "def double(ds, *, state, variables, **_):\n"
        "    ds['x2'] = 2 * ds['x']\n"
        "    return ds\n\n\nderive = [double]\n"
    )
    state.derived_vars_module = str(user)
    state.case_dir = tmp_path
    fv3_plugins.validate()
    fv3_plugins.do_plugins(restart=False)
    assert (tmp_path / "derived_vars_user.py").is_file()
    ds = fv3_plugins.add_derived_variables(xr.Dataset({"x": ("t", [1.0, 2.0])}))
    assert ds["x2"].values.tolist() == [2.0, 4.0]
