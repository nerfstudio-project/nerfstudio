ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
ARG USER_ID=1000
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION
ARG USER_ID

# metainformation
LABEL org.opencontainers.image.version = "0.1.18"
LABEL org.opencontainers.image.source = "https://github.com/nerfstudio-project/nerfstudio"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37

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
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*


# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j `nproc` && \
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
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf ceres-solver

# Install colmap.
RUN git clone --branch 3.8 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
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
RUN python3.10 -m pip install --upgrade pip setuptools pathtools promise pybind11
# Install pytorch and submodules
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3.10 -m pip install \
    torch==2.0.1+cu${CUDA_VER} \
    torchvision==0.15.2+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}
# Install tynyCUDNN (we need to set the target architectures as environment variable first).
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN python3.10 -m pip install git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch

# Install pycolmap, required by hloc.
RUN git clone --branch v0.4.0 --recursive https://github.com/colmap/pycolmap.git && \
    cd pycolmap && \
    python3.10 -m pip install . && \
    cd ..

# Install hloc master (last release (1.3) is too old) as alternative feature detector and matcher option for nerfstudio.
RUN git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git && \
    cd Hierarchical-Localization && \
    python3.10 -m pip install -e . && \
    cd ..

# Install pyceres from source
RUN git clone --branch v1.0 --recursive https://github.com/cvg/pyceres.git && \
    cd pyceres && \
    python3.10 -m pip install -e . && \
    cd ..

# Install pixel perfect sfm.
RUN git clone --recursive https://github.com/cvg/pixel-perfect-sfm.git && \
    cd pixel-perfect-sfm && \
    git reset --hard 40f7c1339328b2a0c7cf71f76623fb848e0c0357 && \
    git clean -df && \
    python3.10 -m pip install -e . && \
    cd ..

RUN python3.10 -m pip install omegaconf
# Copy nerfstudio folder and give ownership to user.
ADD . /home/user/nerfstudio
USER root
RUN chown -R user /home/user/nerfstudio
USER ${USER_ID}

# Install nerfstudio dependencies.
RUN cd nerfstudio && \
    python3.10 -m pip install -e . && \
    cd ..

# Change working directory
WORKDIR /workspace

# Install nerfstudio cli auto completion
RUN ns-install-cli --mode install

# Bash as default entrypoint.
CMD /bin/bash -l
