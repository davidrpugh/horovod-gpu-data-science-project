#!/bin/bash --login

set -e

# set relevant build variables for horovod
export ENV_PREFIX=$PWD/env
export CUDA_HOME=$ENV_PREFIX
export NCCL_HOME=$ENV_PREFIX
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_NCCL_HOME=$NCCL_HOME
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL

# create the conda environment
conda env create --prefix $ENV_PREFIX --file environment.yml --force
conda activate $ENV_PREFIX
source postBuild

