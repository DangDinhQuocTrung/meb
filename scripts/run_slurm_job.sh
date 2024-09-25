#!/bin/bash
#SBATCH --account=project_2002147
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1,nvme:8
#SBATCH --array=0
#SBATCH --output=/scratch/project_2002147/ngtrung1/workdir/logs/training_job-%A-%a.txt
set -e

# Singularity flags
export SING_FLAGS="-B /scratch/project_2002147/ngtrung1/data/:/opt/data/ $SING_FLAGS"
export SING_FLAGS="-B /scratch/project_2002147/ngtrung1/workdir/:/opt/workdir/ $SING_FLAGS"
export SING_FLAGS="-B /scratch/project_2002147/ngtrung1/workdir/cache/tmp/:/tmp/ $SING_FLAGS"
export SING_FLAGS="-B /scratch/project_2002147/ngtrung1/inference_results/:/opt/inference_results/ $SING_FLAGS"
export SING_FLAGS="--nv --pwd /opt/workdir/ $SING_FLAGS"
# Image name
export SING_IMAGE=/scratch/project_2002147/ngtrung1/sif/meb.sif
export PP="/opt/miniconda/bin/python"
export PYTHONPATH=..
export TORCH_HOME=/opt/workdir/cache/

srun apptainer exec $SING_FLAGS $SING_IMAGE $PP \
    meb/main.py \
    seed=$1
