#!/bin/bash

# Submit the HPC test cases. Each case is submitted from its own directory
# (the directory holding its run_config.yaml, whose SBATCH OPTIONS block sets
# the SLURM request). Comment out the cases that should not run.

(cd cases/hourly_restart && ../../../case_submit.sh)
(cd cases/monthly && ../../../case_submit.sh)
(cd cases/nest_gfs && ../../../case_submit.sh)
