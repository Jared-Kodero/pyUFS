# Tests

`test_workflow.py` checks the Python workflow with small synthetic files; it does
not run SHiELD or the UFS_UTILS executables. Without xESMF a stub regridder
(`stubs/`) is used, so regridding weights are not tested.

```bash
python -m pytest tests
```

`cases/` holds short HPC test cases and experiment pairs, run like any case:
copy a directory to a case location and submit with `case_submit.sh`.

| Case | Purpose |
| --- | --- |
| `hourly_restart` | Hourly output across a restart (regression) |
| `monthly_validation` | Three monthly segments; `check_monthly.py` compares monthly means with the hourly samples |
| `tgrad_ctrl`, `tgrad_sst4` | SST perturbation pair over two monthly segments |
| `sm_ctrl`, `sm_dry` | Soil moisture perturbation pair |
| `nest_gfs` | Two nests initialized from GFS only |
