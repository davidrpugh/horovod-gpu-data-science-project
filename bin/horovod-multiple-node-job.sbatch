#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --nodes=4
#SBATCH --gpus=4
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=90G
#SBATCH --constraint=v100
#SBATCH --constraint=cpu_intel_platinum_8260
#SBATCH --partition=batch
#SBATCH --output=../results/%x/slurm-%j.out
#SBATCH --error=../results/%x/slurm-%j.err

# Stops execution of script if any step return non-zero exit code
set -eo pipefail

# Need to define persistent storage for logging... 
PERSISTENT_LOGGING_DIR=../results/$SLURM_JOB_NAME/logs
PERSISTENT_CHECKPOINTS_DIR=$PERSISTENT_LOGGING_DIR/checkpoints
PERSISTENT_TENSORBOARD_DIR=$PERSISTENT_LOGGING_DIR/tensorboard

# N.B. mkdir does not overwrite if these directories already exist
mkdir -p $PERSISTENT_CHECKPOINTS_DIR
mkdir -p $PERSISTENT_TENSORBOARD_DIR

# ...but for best performance write checkpoints and tensorboard logs to local storage
LOCAL_LOGGING_DIR=/tmp/$SLURM_JOB_NAME/$SLURM_JOB_ID/logs
LOCAL_CHECKPOINTS_DIR=$LOCAL_LOGGING_DIR/checkpoints
LOCAL_TENSORBOARD_DIR=$LOCAL_LOGGING_DIR/tensorboard
mkdir -p $LOCAL_CHECKPOINTS_DIR
mkdir -p $LOCAL_TENSORBOARD_DIR

# Load software stack
module purge
module load cuda/10.1.243
conda activate ../env

# Start the nvidia-smi process in the background
NVIDIA_SMI_DELAY_SECONDS=60
nvidia-smi dmon --delay $NVIDIA_SMI_DELAY_SECONDS --options DT >> $PERSISTENT_LOGGING_DIR/nvidia-smi.log &
NVIDIA_SMI_PID=$!

# Start the nvdashboard server running in the background
NVDASHBOARD_PORT=8000
python -m jupyterlab_nvdashboard.server $NVDASHBOARD_PORT &
NVDASHBOARD_PID=$!

# Start the TensorBoard server running in the background
TENSORBOARD_PORT=6006
tensorboard --logdir $LOCAL_TENSORBOARD_DIR --port $TENSORBOARD_PORT --bind_all &
TENSORBOARD_PID=$!

# create the hostfile
HOSTFILE=$PERSISTENT_LOGGING_DIR/hostfile-$SLURM_JOB_ID.txt
scontrol show hostname $SLURM_JOB_NODELIST | awk -v slots=$SLURM_NTASKS_PER_NODE '{print $1 " slots="slots}' > $HOSTFILE

# start the training process in the background
horovodrun -np $SLURM_NTASKS \
       -hostfile $HOSTFILE \
       --verbose \
       python $TRAINING_SCRIPT \
           --data-dir $DATA_DIR \
           --read-checkpoints-from $PERSISTENT_CHECKPOINTS_DIR \
           --write-checkpoints-to  $LOCAL_CHECKPOINTS_DIR \
           --tensorboard-logging-dir $LOCAL_TENSORBOARD_DIR \
           --batch-size 160 

# kill off the monitoring processes
kill $NVIDIA_SMI_PID $NVDASHBOARD_PID $TENSORBOARD_PID

# make sure to get any new files written since last rsync 
rsync -a $LOCAL_CHECKPOINTS_DIR/ $PERSISTENT_CHECKPOINTS_DIR
rsync -a $LOCAL_TENSORBOARD_DIR/ $PERSISTENT_TENSORBOARD_DIR
