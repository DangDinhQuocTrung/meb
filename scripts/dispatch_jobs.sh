#!/bin/bash
set -e

declare -a seeds=(12345 28966)

# Run model
for SEED in "${seeds[@]}"; do
    sbatch ./run_slurm_job.sh ${SEED}
done
