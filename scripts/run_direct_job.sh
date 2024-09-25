#!/bin/bash

# Singularity flags
export SING_FLAGS="-B /scratch/project_2002147/ngtrung1/data/:/opt/data/ $SING_FLAGS"
export SING_FLAGS="-B /scratch/project_2002147/ngtrung1/workdir/:/opt/workdir/ $SING_FLAGS"
export SING_FLAGS="-B /scratch/project_2002147/ngtrung1/workdir/cache/:/tmp/ $SING_FLAGS"
export SING_FLAGS="-B /scratch/project_2002147/ngtrung1/inference_results/:/opt/inference_results/ $SING_FLAGS"
export SING_FLAGS="--nv --pwd /opt/workdir/ $SING_FLAGS"
# Image name
export SING_IMAGE=/scratch/project_2002147/ngtrung1/sif/meb.sif
export PP="/opt/miniconda/bin/python"
export PYTHONPATH=..
export TORCH_HOME=/opt/workdir/cache/

srun --account=project_2002147 --partition=gputest \
    --time=00:15:00 --ntasks=1 --nodes=1 \
    --cpus-per-task=8 --mem-per-cpu=8G --gres=gpu:v100:1,nvme:8 \
    --output=/users/ngtrung1/check_workdir/logs/training_job-%A-%a.txt \
    apptainer exec $SING_FLAGS $SING_IMAGE $PP \
    meb/main.py \
    seed=11111
