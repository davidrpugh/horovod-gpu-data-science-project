FROM nvidia/cuda:10.1-devel-ubuntu16.04

LABEL maintainer="pughdr <david.pugh@kaust.edu.sa>"

SHELL [ "/bin/bash", "--login", "-c" ]

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG username=al-khawarizmi
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

COPY environment.yml requirements.txt /tmp/
RUN chown $UID:$GID /tmp/environment.yml /tmp/requirements.txt

COPY postBuild docker/entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/postBuild /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/postBuild /usr/local/bin/entrypoint.sh

USER $USER

# install miniconda
ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env
ENV HOROVOD_CUDA_HOME $CUDA_HOME
ENV HOROVOD_NCCL_HOME $ENV_PREFIX
ENV HOROVOD_GPU_ALLREDUCE NCCL
ENV HOROVOD_GPU_BROADCAST NCCL
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda activate $ENV_PREFIX && \
    . /usr/local/bin/postBuild && \
    conda clean --all --yes

# use an entrypoint script to insure conda environment is properly activated at runtime
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# default command will be to launch JupyterLab server for development
CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0" ]
