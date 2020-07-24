#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=48G
#SBATCH --constraint=cpu_intel_gold_6142|cpu_intel_platinum_8260
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --job-name=create-conda-env
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err

# make sure NVCC is available
module purge
module load cuda/10.1.243

# create the conda environment
./bin/create-conda-env.sh
