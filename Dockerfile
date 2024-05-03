ARG USER_ID=1000
# Define base image.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ARG USER_ID

# metainformation
# LABEL org.opencontainers.image.version = "0.1.18"
LABEL org.opencontainers.image.source = "https://github.com/nerfstudio-project/nerfstudio"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:11.8.0-devel-ubuntu22.04"

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libceres-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    ninja-build \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    python-is-python3 \
    python3-venv \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install colmap.
RUN git clone --branch 3.9.1 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
             "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf colmap

# Create non root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u ${USER_ID} user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new uer and workdir.
USER ${USER_ID}
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install packages.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools pathtools promise pybind11
# Install pytorch and submodules
RUN python3.10 -m pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118

# Install tynyCUDNN (we need to set the target architectures as environment variable first).
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN python3.10 -m pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch
RUN python3.10 -m pip install --no-cache-dir pycolmap==0.6.1 pyceres==2.1 omegaconf

# Install hloc 1.4 as alternative feature detector and matcher option for nerfstudio.
RUN git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git && \
    cd Hierarchical-Localization && \
    git checkout v1.4 && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

# Copy nerfstudio folder and give ownership to user.
ADD . /home/user/nerfstudio
USER root
RUN chown -R user /home/user && \
    chmod -R go=u /home/user
USER ${USER_ID}

# Install nerfstudio dependencies.
RUN cd nerfstudio && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

# Change working directory
WORKDIR /workspace

# Install nerfstudio cli auto completion
RUN ns-install-cli --mode install

# Bash as default entrypoint.
CMD /bin/bash -l
