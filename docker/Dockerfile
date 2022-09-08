FROM nvidia/cudagl:10.2-devel-ubuntu18.04

# https://github.com/facebookresearch/habitat-lab/blob/main/Dockerfile
# Setup basic packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# https://stackoverflow.com/questions/27701930/how-to-add-users-to-docker-container
RUN useradd -m -s /bin/bash -u 1000 ubuntu
USER ubuntu
WORKDIR /home/ubuntu

# Install conda
RUN curl -L -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh && \
    ~/miniconda3/bin/conda init && \
    ~/miniconda3/bin/conda clean -ya
ENV PATH /home/ubuntu/miniconda3/bin:$PATH

SHELL ["/bin/bash", "-c"]

RUN conda install -y python=3.7 && \
    conda install -y pytorch==1.5.1 cudatoolkit=10.2 -c pytorch && \
    conda install -y habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly && \
    conda clean -ya

RUN git clone --single-branch https://github.com/facebookresearch/habitat-sim && \
    cd habitat-sim && git checkout ccbfa32e0af6d2bfb5a3131f28f69b72f184e638 && \
    conda install cmake=3.14.0 patchelf ninja && conda clean -ya && \
    pip install -r requirements.txt && python setup.py install --bullet --headless && pip cache purge && \
    cd .. && rm -rf habitat-sim

# RUN cd ~/miniconda3/lib/python3.7/site-packages && \
#     hsim_bindings=$(find . -name "*habitat_sim_bindings*so") && \
#     ext_folder=$(dirname "${hsim_bindings}") && \
#     magnum_bindings=$(find . -name "*_magnum*so") && \
#     patchelf --set-rpath "\$ORIGIN/${ext_folder}" --force-rpath "${magnum_bindings}"

RUN git clone --single-branch https://github.com/facebookresearch/habitat-lab && \
    cd habitat-lab && git checkout 2ec4f6832422faebf20ca413b1ebf78547a4855d && \
    pip install gym==0.18.3 --no-deps && pip install -r requirements.txt && pip cache purge && \
    cd .. && rm -rf habitat-lab

RUN pip install \
    "lmdb>=0.98" \
    webdataset==0.1.40 \
    ifcfg \
    "moviepy>=1.0.1" \
    tensorboard \
    pybullet==3.0.4 && \
    pip cache purge

ENV MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
