#!/bin/bash
set -e

# --- 1. Setup Directories & Environment ---
# Ensures env.yaml is found in the same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CONTAINERS_DIR}"
JOB_TMP="${HOME}/scratch/apptainer"

mkdir -p "$OUTPUT"
mkdir -p "$JOB_TMP"
# Pre-run check
if [ ! -f "$SCRIPT_DIR/env.yaml" ]; then
    echo "ERROR: env.yaml not found in $SCRIPT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT"
rm -rf "$JOB_TMP"
mkdir -p "$JOB_TMP/sandboxes"

# Apptainer environment setup
export APPTAINER_BINDPATH="$JOB_TMP:/workdir"
module purge
unset LD_LIBRARY_PATH

# --- 2. Clean Cache & Pre-download Assets ---
apptainer cache clean -f

MINICONDA_SH="$JOB_TMP/miniconda.sh"
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_SH"
chmod +x "$MINICONDA_SH"

# --- 3. Pull Base Images ---
apptainer pull "$JOB_TMP/fregrid.sif"    docker://gfdlfv3/fre-nctools > /dev/null 2>&1
apptainer pull "$JOB_TMP/preprocess.sif" docker://gfdlfv3/preprocessing > /dev/null 2>&1
apptainer pull "$JOB_TMP/shield.sif"     docker://gfdlfv3/shield > /dev/null 2>&1

# --- 4. Build Sandboxes ---
# FREGRID is built first so other images can copy its MPI files
for VAR in FREGRID PREPROCESS SHIELD; do
    VAR_LC=$(echo "$VAR" | tr '[:upper:]' '[:lower:]')
    SANDBOX="$JOB_TMP/sandboxes/$VAR"
    SOURCE_SIF="$JOB_TMP/${VAR_LC}.sif"


    apptainer build --sandbox "$SANDBOX" "$SOURCE_SIF"

    # Inject installer and env.yaml
    cp "$MINICONDA_SH" "$SANDBOX/miniconda.sh"
    cp "$SCRIPT_DIR/env.yaml" "$SANDBOX/env.yaml"

    # Sync OpenMPI share files to others from FREGRID
    if [ "$VAR" != "FREGRID" ]; then
        mkdir -p "$SANDBOX/opt/openmpi"
        cp -rf "$JOB_TMP/sandboxes/FREGRID/opt/openmpi/share" "$SANDBOX/opt/openmpi/"
    fi
done

# --- 5. Install Conda Environments ---
for VAR in FREGRID PREPROCESS; do
    VAR_LC=$(echo "$VAR" | tr '[:upper:]' '[:lower:]')
    SANDBOX="$JOB_TMP/sandboxes/$VAR"

    echo "[CONDA] Installing $VAR_LC environment..."
    apptainer exec --writable --no-home "$SANDBOX" bash -c "
        mkdir -p /workdir/tmp
        export TMPDIR=/workdir/tmp
        
        # Install Miniconda
        ./miniconda.sh -b -p /opt/conda
        
        # Accept Terms of Service (The Terms)
        /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
        /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

        # Create environment from yml
        /opt/conda/bin/conda env create -n $VAR_LC -f /env.yaml -y --quiet
        /opt/conda/bin/conda clean -a -y
        
        # Create root symlinks
        ln -sf /opt/conda/envs/$VAR_LC/bin/python /$VAR_LC
        ln -sf /opt/conda/envs/$VAR_LC/bin/wget /wget
        ln -sf /opt/conda/envs/$VAR_LC/bin/wgrib2 /wgrib2
        
        # Internal cleanup
        rm -rf /workdir/tmp /miniconda.sh /env.yaml
    "
done

# --- 6. Final SIF Creation ---
for VAR in FREGRID PREPROCESS SHIELD; do
    VAR_LC=$(echo "$VAR" | tr '[:upper:]' '[:lower:]')
    SANDBOX="$JOB_TMP/sandboxes/$VAR"
    FINAL_SIF="$OUTPUT/${VAR_LC}.sif"

    # Remove installers from SHIELD (which didn't run the conda loop)
    rm -f "$SANDBOX/miniconda.sh" "$SANDBOX/env.yaml"

    rm -f "$FINAL_SIF"
    apptainer build "$FINAL_SIF" "$SANDBOX"
done

# --- 7. Final Cleanup ---
rm -rf "$JOB_TMP"
