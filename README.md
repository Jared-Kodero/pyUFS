# ufs_py HPC Run Guide (GFDL SHiELD Model Setup)

A portable Python workflow framework for configuring, initializing, and executing the GFDL SHiELD model across non-native HPC environments

# Overview

This guide documents the end-to-end workflow for running the GFDL SHiELD model on the Oscar HPC system. The provided Python-based framework replaces the native UFS_UTILS workflow with a portable implementation designed for systems where the original infrastructure is not directly supported.

The workflow supports configuration, preprocessing, model execution, and postprocessing within a unified and reproducible environment.

## Additional Documentation
For deep dives into specific components, refer to the official documentation:
* **SHiELD Model:** [https://www.gfdl.noaa.gov/shield/](https://www.gfdl.noaa.gov/shield/)
* **NOAH MP Land Model:** [Technical Note (PDF)](https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs/LAND_SURFACE/noah_mp_tech_note.pdf)
* **FV3 Dynamical Core:** [https://www.gfdl.noaa.gov/fv3/fv3-documentation-and-references/](https://www.gfdl.noaa.gov/fv3/fv3-documentation-and-references/)
* **FV3 Namelist Guide:** [Namelist PDF](https://www.gfdl.noaa.gov/wp-content/uploads/2017/09/fv3_namelist_Feb2017.pdf)
* **UFS Utilities:** [https://noaa-emcufs-utils.readthedocs.io/en/latest/ufs_utils.html](https://noaa-emcufs-utils.readthedocs.io/en/latest/ufs_utils.html)
* **UFS Weather Model:** [https://ufs-weather-model.readthedocs.io/en/develop/Introduction.html](https://ufs-weather-model.readthedocs.io/en/develop/Introduction.html)

* **Flexible Modeling System:** [https://noaa-gfdl.github.io/FMS/md_docs_doxygenGuide.html](https://noaa-gfdl.github.io/FMS/md_docs_doxygenGuide.html)


## System & Workflow Architecture

Execution on Oscar is controlled by a set of custom Python scripts located in `<path_to_dir>/gfdl_shield/ufs_py`. These scripts reproduce the functionality of the official `UFS_UTILS` workflow system (used on NOAA and GFDL HPC Systems), bypassing the need for original bash scripts that are difficult to port due to strict software requirements. 

### The Runtime Pipeline
The pipeline proceeds through the following sequence:
1. `submit.sh`
2. `case_submit.sh`
3. `case_run.sh`
4. Preprocess *(inside Apptainer/Singularity container)*
5. SHiELD MPI integration *(inside container or using your user-built executable)*
6. Output regridding
7. Output synchronization

### Shared Environment Tree
The shared SHiELD installation is located at: `<path_to_dir>/gfdl_shield`

```text
ufs_py/
├── case_run.sh              # Executes the runtime workflow
├── case_submit.sh           # Prepares and submits jobs
├── configs/                 # YAML, namelists, and runtime configuration
│   ├── run_config.yaml
│   ├── machine_config.yaml
│   ├── env.yaml
│   ├── input.nml
│   ├── field_table.yaml
│   ├── data_table.yaml
│   ├── diag_table
│   ├── diag_field.csv
│   ├── *.vars.csv           # Variable mappings (ERA5, GFS, HRRR)
│   └── utilities (scripts, templates, notebooks)
├── docs/                    # Documentation notebooks
│   └── oscar_readme.ipynb
├── fregrid/                 # Regridding utilities (external tools interface)
├── preprocess/              # Preprocessing stage (containerized execution)
├── py_scripts/              # Core Python workflow engine
│   ├── driver.py            # Main orchestrator
│   ├── fv3gfs_*             # Grid, ICs, runtime, and physics setup modules
│   ├── chgres_cube.py       # Initial condition generation
│   ├── era5_to_fv3.py       # ERA5 conversion pipeline
│   ├── global_cycle.py      # Surface cycling
│   ├── regrid.py            # Output regridding
│   ├── merge_outputs.py     # Output consolidation
│   └── supporting utilities and compiled bytecode
├── tests/                   # Test cases and validation data
│   ├── test_case/
│   └── test_data/
└── tools/                   # Supporting CLI utilities
    ├── parse_config.py
    └── sbatch.sh    # Pipeline execution scripts

```


## Static Runtime Datasets (`fix/` directory)
The `fix` directory contains essential pre-configured data for your runs, including:
* Atmospheric climatologies
* Land surface datasets
* Terrain and orography
* Lookup tables
* Regridding support files

<small>

| Filename | Description |
|----------|-------------|
| aerosol.dat | External aerosols data file |
| CFSR.SEAICE.1982.2012.monthly.clim.grb | CFS reanalysis of monthly sea ice climatology |
| co2historicaldata_YYYY.txt | Monthly CO2 in PPMV data for year YYYY |
| global_albedo4.1x1.grb | Four albedo fields for seasonal mean climatology: 2 for strong zenith angle dependent (visible and near IR) and 2 for weak zenith angle dependent |
| global_glacier.2x2.grb | Glacier points, permanent/extreme features |
| global_h2oprdlos.f77 | Coefficients for photochemical production and loss of water (H2O) |
| global_maxice.2x2.grb | Maximum ice extent, permanent/extreme features |
| global_mxsnoalb.uariz.t126.384.190.rg.grb | Climatological maximum snow albedo |
| global_o3prdlos.f77 | Monthly mean ozone coefficients |
| global_shdmax.0.144x0.144.grb | Climatological maximum vegetation cover |
| global_shdmin.0.144x0.144.grb | Climatological minimum vegetation cover |
| global_slope.1x1.grb | Climatological slope type |
| global_snoclim.1.875.grb | Climatological snow depth |
| global_snowfree_albedo.bosu.t126.384.190.rg.grb | Climatological snow-free albedo |
| global_soilmldas.t126.384.190.grb | Climatological soil moisture |
| global_soiltype.statsgo.t126.384.190.rg.grb | Soil type from STATSGO dataset |
| global_tg3clim.2.6x1.5.grb | Climatological deep soil temperature |
| global_vegfrac.0.144.decpercent.grb | Climatological vegetation fraction |
| global_vegtype.igbp.t126.384.190.rg.grb | Climatological vegetation type |
| global_zorclim.1x1.grb | Climatological surface roughness |
| RTGSST.1982.2012.monthly.clim.grb | Monthly climatological global sea surface temperature |
| seaice_newland.grb | High resolution land mask |
| sfc_emissivity_idx.txt | External surface emissivity data table |
| solarconstant_noaa_an.txt | External solar constant data table |

</small>

**Important:** You **do not** need to manually download these files from the NOAA S3 bucket. To save time and storage space, they have already been downloaded and are available globally on the Oscar system at: 
`<path_to_dir>/gfdl_shield/fix`

If missing download from https://noaa-nws-global-pds.s3.amazonaws.com/index.html#fix/



```bash
INIT="2026031200Z"
CASE_NAME="C96.R4N2.R2N1.CNTRL"
WORK_ROOT="$HOME/scratch/shield_cases/$INIT"
CASE_DIR="$WORK_ROOT/$CASE_NAME"

echo "Creating Work Directory at: $CASE_DIR"
mkdir -p "$CASE_DIR"
```
```markdown
CASE_ROOT/
└── YYYYMMDDHHZ/
    └── CASE_NAME/
        ├── run_config.yml
        ├── submit.sh
        └── *.nml, diag_table, etc.
```

When you submit, everything is automated, you can check each step by checking the resulting dir under `CASE_NAME/run` 


### Initial Conditions & Pre-processing (`chgres_cube`)

The SHiELD model requires initial condition (IC) data to be formatted specifically for the FV3 cubed-sphere grid. This data conversion is handled by a Fortran utility called **`chgres_cube`**.

### Automatic Downloading and Execution
You do **not** need to manually download starting conditions or run the `chgres_cube` utility yourself. The Oscar Python workflow is designed to automate this entirely:

1. **Auto-Download:** As long as a valid, standard `init_datetime` (e.g., `"2026031200Z"`) is provided in your `run_config.yml`, the workflow will automatically locate and download the required operational GFS or HRRR data for that specific initialization cycle.
2. **Auto-Processing:** During the pre-processing phase of the pipeline, `driver.py` will automatically invoke `chgres_cube`. It reads your configuration, takes the downloaded GFS/HRRR data, and handles all necessary regridding and interpolation to map the data onto your specific global or nested grid setup before the main model executable runs.

### Advanced: Customizing Initial Conditions (e.g., ERA5 Integration)

Get initialization data for the specified external model

## Data Sources

### GFS (Global Forecast System) with 0.25° resolution.

- NOAA AWS S3  
  https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF  

- NCAR GDEX  
  https://tds.gdex.ucar.edu/thredds/fileServer/files/g/d084001/YYYYMMDD/gfs.0p25.YYYYMMDDHH.f000.grib2  

---

### HRRR (High-Resolution Rapid Refresh)

CONUS-specific model with 3km resolution, ideal for high-resolution runs.

- NOAA AWS S3  
  https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfnatfFF.grib2  

- Google Cloud Storage  
  https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf00.grib2  


**You can modify** the initial condition NetCDF (`.nc`) files to use alternative datasets, such as **ERA5**. 

Because `chgres_cube` handles the complex generation of the FV3 cubed-sphere geometries, you **must** generate the base tiles using GFS first. Once the structure is generated, you can swap the underlying data:

1. **Generate Base Tiles:** Run the standard pre-processing step using the default GFS data to allow `chgres_cube` to create the grid structure and output the base `.nc` tile files.
2. **Regrid and Replace:** Pause or intercept the pipeline before the main model execution. Use a Python regridding library like `xesmf` (which pairs well with `xarray` and `numpy` for data manipulation) to regrid your ERA5 data onto the newly created FV3 tile geometries.
3. **Overwrite Variables:** Replace the existing GFS data variables in the `.nc` tile files with your regridded ERA5 data. 



**Important Note on Variables:** Ensure all required SHiELD variables are present. Some specialized variables needed by the FV3 core may not exist natively in the ERA5 dataset and will need to be manually calculated from available ERA5 fields before you write them into the tile files.  Open tile sfc and atm tiles files and make sure your atm amd sfc ERA5 ds has all the required vars and levels are the same



see `<path_to_dir>/gfdl_shield/ufs_py/py_scripts/era5_to_fv3.py` for example code

### After IC generation has finished the workdir will look like this
- FIXED
- GRID
- HIST
- INIT_DATA
- INPUT
- LOGS
- OUTPUT
- RESTART
- TMP


### Model & Nest Configurations

### Example Nest Configurations Reference
If you plan to use nests, here are standard configurations to guide your `run_config.yml` setup:

* **Parent C96 with 3:1 nest:** Parent ~100 km | Child ~33 km
* **Parent C192 with 3:1 nest:** Parent ~50 km | Child ~17 km
* **Parent C384 with 3:1 nest:** Parent ~25 km | Child ~8 km
* **Convection-permitting nest:** Parent C768 (~13 km) | Refinement 4:1 | Child ~3.25 km



### The `run_config.yml` File
Each case directory must contain a `run_config.yml` file. Required parameters include: `init_datetime`, `run_nhours`, `res`, and `gtype`.

see `<path_to_dir>/gfdl_shield/ufs_py/configs/run_config.yml` for exmaple

### Example Global run_config.yml

```yaml
description: C96 control run
init_datetime: "2026031200Z"
run_nhours: 6
res: C96
gtype: uniform
levels: 64
continue_run: false
debug: false
chgres_config: null
nml: null
tileX_nml: null
sbatch:
  exclusive: false # Run with exclusive node access (true/false)
  constraint: false # Node constraints (e.g., "24-core", "32-core")
  cpus_per_task: 1 # Number of CPU cores per task
  time: 12:00:00 # Maximum wall time (HH:MM:SS)
  mem: 480 # Total memory (GB)
  nnodes: 3 # Number of nodes
  ntasks: 48 # Total number of tasks (e.g., MPI ranks)
  output: "shield.driver.log" # Path for sbatch output log
  partition: "batch" # Partition name 

```

### Example Nested run_config.yml

```yaml

init_datetime: "2026031200Z"
run_nhours: 24
res: C96
gtype: nest
levels: 64
refine_ratio: [4,2]
lon_min: [-125,-95]
lon_max: [-47,-57]
lat_min: [25,32]
lat_max: [60,55]
nml: parent.yml # or .nml
tileX_nml: nest_all.yml # or .nml
tile7_nml: nest_tile7.yml # or .nml
k_splt:
n_split:
nest_k_split:
nest_n_split:
sbatch:
  exclusive: false # Run with exclusive node access (true/false)
  constraint: false # Node constraints (e.g., "24-core", "32-core")
  cpus_per_task: 1 # Number of CPU cores per task
  time: 12:00:00 # Maximum wall time (HH:MM:SS)
  mem: 480 # Total memory (GB)
  nnodes: 3 # Number of nodes
  ntasks: 48 # Total number of tasks (e.g., MPI ranks)
  output: "shield.driver.log" # Path for sbatch output log
  partition: "batch" # Partition name 
```


### Compiling a Custom SHiELD Executable (Optional)

If you require a custom model binary, you must clone the source code 

 
```bash
# 1. Clone the repository and checkout the oscar branch
git clone -b oscar https://github.com/biosphereNclimate/SHiELD_build.git 
cd SHiELD_build

# 2. Retrieve source code and submodules
./CHECKOUT_code
git submodule update --init mkmf
```
**The `gettid` patch:** On systems with `glibc > 2.30` (like Oscar), a conflict occurs because `gettid` is already defined in glibc. 
You must modify line 51 in `SHiELD_SRC/FMS/affinity/affinity.c` to remove the duplicate `static` declaration

```bash
# 3. Patch the gettid conflict in FMS
# Replaces 'static pid_t gettid(void)' with 'pid_t gettid(void)' on line 51
sed -i '51s/static pid_t gettid(void)/pid_t gettid(void)/' ../SHiELD_SRC/FMS/affinity/affinity.c

```

Do your scientific modifications on SHiELD_SRC


**Before building** you must modify  `SHiELD_build/site/environment.gnu.sh` and update the modules, 
Replace the modules  under case oscar with

```bash
module load hpcx-mpi
module load netcdf-mpi
module load libyaml
module load cmake
```

Compile

```bash
./Build/COMPILE 64bit gnu pic

# The executable will be generated. Note its absolute path and export it, 
export SHIELD_EXE=/path/to/compiled/bin
```

###  Minimal Quick Start

The following cell will execute a full end-to-end setup for a minimal run. It will:
1. Create the necessary directories.
2. Generate the `run_config.yml`.
3. Create the `submit.sh` Slurm script.
4. Submit the job.

see `<path_to_dir>/gfdl_shield/ufs_py/configs/run_config.yml` for exmaple


### Create Configuration
```yaml
# run_config.yml
description: C96 control run
init_datetime: "2026031200Z"
run_nhours: 6
res: C96
gtype: uniform
levels: 64
continue_run: false
archive_data: true
sbatch:
  exclusive: false # Run with exclusive node access (true/false)
  constraint: false # Node constraints (e.g., "24-core", "32-core")
  cpus_per_task: 1 # Number of CPU cores per task
  time: 12:00:00 # Maximum wall time (HH:MM:SS)
  mem: 480 # Total memory (GB)
  nnodes: 3 # Number of nodes
  ntasks: 48 # Total number of tasks (e.g., MPI ranks)
  output: "shield.driver.log" # Path for sbatch output log
  partition: "batch" # Partition name 


```

### Modify Diag Table
To specify custom output frequencies and variables, you must provide a modified `diag_table` within your `$CASE_DIR`. If no file is found there, the model will use the default settings.

**Copy the default table:**
   ```bash
   cp "<path_to_dir>/gfdl_shield/ufs_py/configs/diag_table "$CASE_DIR/
   ```
**Identify variables of intrest and update table**
Refer to the available fields in: `<path_to_dir>/gfdl_shield/ufs_py/configs/diag_field.csv`



### Create Submission Script


```bash
#!/bin/bash -l
#submit.sh

"<path_to_dir>/gfdl_shield/ufs_py/case_submit.sh "


# 4. Make executable and submit
chmod +x submit.sh
echo "Submitting job from $CASE_DIR..."
# Un-comment the line below to actually submit to the Slurm queue when running:
# ./submit.sh

```