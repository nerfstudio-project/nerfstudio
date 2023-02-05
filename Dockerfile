# Define base image.
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA architectures, required by tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ENV TCNN_CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libcgal-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -rf glog
# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Install Ceres-solver (required by colmap).
RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -rf ceres-solver

# Install colmap.
RUN git clone --branch 3.7 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
        	 -DCUDA_NVCC_FLAGS="--std c++14" && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -rf colmap
    
# Create non root user and setup environment.
RUN useradd -m -d /home/user -u 1000 user

# Switch to new uer and workdir.
USER 1000:1000
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install packages.
RUN python3.10 -m pip install --upgrade pip setuptools pathtools promise
# Install pytorch and submodules.
RUN python3.10 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# Install tynyCUDNN.
RUN python3.10 -m pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# Copy nerfstudio folder and give ownership to user.
ADD . /home/user/nerfstudio
USER root
RUN chown -R user:user /home/user/nerfstudio
USER 1000:1000

# Install nerfstudio dependencies.
RUN cd nerfstudio && \
    python3.10 -m pip install -e . && \
    cd ..

# Change working directory
WORKDIR /workspace

# Install nerfstudio cli auto completion and enter shell if no command was provided.
CMD ns-install-cli --mode install && /bin/bash

