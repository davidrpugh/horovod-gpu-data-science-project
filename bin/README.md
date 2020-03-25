## Slurm

### Single node jobs

The `horovod-single-node-job.sh` job script can be broken down into a number of sections.

#### Slurm directives

When runnning a distributed training job with Horovod you will want to set the total number of 
tasks (`--ntasks`) equal to the total number of GPUs requested across all nodes and set the 
number of tasks per node (`--tasks-per-node`) equal to the number of NVIDIA V100 GPUs requested 
per node (`--gres=gpu:v100`). In the directives below we are requesting all 8 32 GB NVIDIA 
V100 GPUs available on a single node so we need to set `--nodes=1`, `--ntasks=8`, 
`--tasks-per-node=8` and `--gres=gpu:v100:8`.

In order to achieve top training performance, you need to request sufficient CPUs to feed the 
training data to the GPUs. A good rule of thumb is to request 5-6 CPUs per 32 GB NVIDIA V100 GPU.
To achieve this we set `--cpus-per-task=6` (given that `--tasks-per-node=8` this means we have 
requested all 48 CPUs on the node). We also provide a constraing on the type of CPU requested: 
`--constraint=cput_intel_platinum_8260` insures that all nodes have Intel "Cascade Lake" CPUs.

Finally, we also need to request sufficient CPU memory. A good rule of thumb is to request at 
least twice the CPU memory as available GPU memory. Since we have requested 8 x 32 GB = 256 GB 
of GPU memory we should request at least 512 GB of CPU memory. However, given that we are 
already requesting all of the GPUs and all of the CPUs on the node we might as well ask for all 
available memory on the node by setting `--mem=0`. 

Finally, we direct out the slurm output and error logs to files in particular directories. Note 
that `%x` refers to the Slurm job name (which we specify!) and `%j` refers to the Slurm job id 
(which Slurm specifies!). Thus our output and error files will be written into the directory 
`../results/$JOB_NAME/` and will contain the Slurm job id as part of the file name. This 
naming convention helps to keep our results directories neat and tidy.

```bash
#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100:8
#SBATCH --constraint=cpu_intel_platinum_8260
#SBATCH --partition=batch
#SBATCH --output=../results/%x/slurm-%j.out
#SBATCH --error=../results/%x/slurm-%j.err
```

#### Checkpointing

Efficient checkpointing can be a bit tricky. The main concern is that the worker responsible for 
writing the checkpoint files will be blocked from training while the checkpoint files are being 
written. Thus we should avoid writing checkpoints directly to `/ibex/(f)scratch` which is a 
shared file system whose performance can vary depending on overall IO load. Instead should write 
checkpoints files to local, on-node storage. However local, on-node storage is not persistent and  
will be wiped after the job terminates. So if we are going to write checkpoint files to local 
storage we need to periodically sync our checkpoint files with persistent storage (in a manner 
which will not block our training progress).

We follow the same procedure for our Tensorboard logs. Periodically syncing Tensorboard logs to 
`/ibex/(f)scratch` allows us to run Tensorboard on a login node while the training job is 
running in order to confirm that training is converging as expected.
 
```bash
...
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
...
HOROVODRUN_PID=$!

# asynchronous rsync of training logs between local and persistent storage
RSYNC_DELAY_SECONDS=600
HOROVODRUN_STATE=$(ps -h --pid $HOROVODRUN_PID -o state | head -n 1)
while [ "${HOROVODRUN_STATE}" != "" ]
    do
        rsync -a $LOCAL_CHECKPOINTS_DIR/ $PERSISTENT_CHECKPOINTS_DIR
        rsync -a $LOCAL_TENSORBOARD_DIR/ $PERSISTENT_TENSORBOARD_DIR
        sleep $RSYNC_DELAY_SECONDS
        HOROVODRUN_STATE=$(ps -h --pid $HOROVODRUN_PID -o state | head -n 1)
done
...
# make sure to get any new files written since last rsync 
rsync -a $LOCAL_CHECKPOINTS_DIR/ $PERSISTENT_CHECKPOINTS_DIR
rsync -a $LOCAL_TENSORBOARD_DIR/ $PERSISTENT_TENSORBOARD_DIR
```

#### Loading the software application stack

It is always good practice to explicitly load the software application stack inside the job script.
First we load the appropriate Cuda Toolkit module for the version of PyTorch we are using. Then we 
activate the Conda environment containing all the other software dependencies in such as NCCL, CUDNN, 
OpenMPI, and Horovod.

```bash
...
# Load software stack
module load cuda/10.1.243
conda activate ../env
...
```

#### GPU Resource Monitoring

Understanding GPU resource utilization is critical for performance tuning deep learning applications. 
At present accessing GPU resource utilization for an individual job is a bit challenging so we wanted 
to share a solution that we have useful. The basic idea is to launch `nvidia-smi dmon` as a 
background process prior to starting the training job (this way we don't block the training job from 
making progress) and to have `nvidia-smi dmon` append its logs to a file on persistent storage (so 
that the GPU resource utilization logs can be inspected whilst the training job is still running).

We also use a new tool called `jupyterlab-nvdashboard` developed by the NVIDIA RAPIDS team to provide 
a browser-based UI for GPU, CPU, memory, and IO resource monitoring. The server is started as a 
background process prior to launching your training job and is accessible from a web browser at the 
following URL
```
$IBEX_NODE_NAME.ibex.kaust.edu.sa:$NVDASHBOARD_PORT
``` 
where `$IBEX_NODE_NAME` is the name of the node on Ibex where your job is running.

```bash
...
# start the nvidia-smi process in the background
NVIDIA_SMI_DELAY_SECONDS=60
nvidia-smi dmon --delay $NVIDIA_SMI_DELAY_SECONDS --options DT >> $PERSISTENT_LOGGING_DIR/nvidia-smi.log &
NVIDIA_SMI_PID=$!

# start the nvdashboard server in the background
NVDASHBOARD_PORT=8889
python -m jupyterlab_nvdashboard.server $NVDASHBOARD_PORT &
NVDASHBOARD_PID=$!

...
# kill off the GPU monitoring processes
kill $NVIDIA_SMI_PID $NVDASHBOARD_PID
...
```

#### Running the training job

To launch the training job we use Horovod's builtin runner `horovodrun` (which is itself a thin wrapper 
around `mpirun`). Note that we set the number of processes `-np` equal to the total number of tasks 
`$SLURM_NTASKS` (which, you may recall, is also equal to the total number of GPUs being requested!). We 
also control the actual training script being launched using the `$TRAINING_SCRIPT` environment variable 
so that this same job script can be used for any (single-node) training job. Also note that we launch 
the training process in the background so that we can more easily support non-blocking (i.e., asynchronous) 
checkpointing.

```bash
...
# start the training process in the background
horovodrun -np $SLURM_NTASKS python $TRAINING_SCRIPT \
    --data-dir $DATA_DIR \
    --read-checkpoints-from $PERSISTENT_CHECKPOINTS_DIR \
    --write-checkpoints-to $LOCAL_CHECKPOINTS_DIR \
    --tensorboard-logging-dir $LOCAL_TENSORBOARD_DIR &
...
```

### Submitting jobs

Before submitting the job we need to make sure that our results directory exists as Slurm will try to create 
the `slurm-%j.out` and `slurm-%j.err` files in this directory before launching the our job and this process 
will fail if the directory doesn't already exist. Note also that we export two environment variables 
`$TRAINING_SCRIPT` and `DATA_DIR` from our shell environment on the login node to the shell environment on 
the compute node. This was a design choice that should allow the `horovod-single-node-job.sh` job script to 
be reused for arbitrary training jobs (at least those that following the workflow conventions discussed above).
Finally, we submit the job to Ibex using the `sbatch` command.
 
```bash
$ USER_EMAIL=your.name@kaust.edu.sa # don't forget to change this!
$ JOB_NAME=horovod-single-node-benchmark
$ mkdir ../results/$JOB_NAME
$ TRAINING_SCRIPT=../src/horovod-example/train.py
$ DATA_DIR=/local/reference/CV/ILSVR/classification-localization/data/jpeg
$ sbatch --job-name $JOB_NAME --mail-user $USER_EMAIL --mail-type=ALL --export TRAINING_SCRIPT=$TRAINING_SCRIPT,DATA_DIR=$DATA_DIR horovod-single-node-job.sh
```
