ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION

# Define username, user uid and gid
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

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

# Create non root user, add it to custom group and setup environment.
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -d /home/${USERNAME} --shell /usr/bin/bash 
# OPTIONAL
# If sudo privilages are not required comment below line
# Create simple password for user and add it to sudo group
# Update group so that it is not required to type password for commands: apt update/upgrade/install/remove
RUN echo "${USERNAME}:password" | chpasswd \
    && usermod -aG sudo ${USERNAME} \
    && echo "%sudo ALL=NOPASSWD:/usr/bin/apt-get update, /usr/bin/apt-get upgrade, /usr/bin/apt-get install, /usr/bin/apt-get remove" >> /etc/sudoers

# Create workspace folder and change ownership to new user
RUN mkdir /workspace && chown ${USER_UID}:${USER_GID} /workspace

# Switch to new user and workdir.
USER ${USER_UID}
WORKDIR /home/${USERNAME}

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"

# Upgrade pip and install packages.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools==69.5.1 pathtools promise pybind11 omegaconf

# Install pytorch and submodules
# echo "${CUDA_VERSION}" | sed 's/.$//' | tr -d '.' -- CUDA_VERSION -> delete last digit -> delete all '.'
RUN CUDA_VER=$(echo "${CUDA_VERSION}" | sed 's/.$//' | tr -d '.') && python3.10 -m pip install --no-cache-dir \
    torch==2.1.2+cu${CUDA_VER} \
    torchvision==0.16.2+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

# Install tiny-cuda-nn (we need to set the target architectures as environment variable first).
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN python3.10 -m pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# Install pycolmap, required by hloc.
RUN git clone --branch v0.4.0 --recursive https://github.com/colmap/pycolmap.git && \
    cd pycolmap && \
    python3.10 -m pip install --no-cache-dir . && \
    cd ..

# Install hloc 1.4 as alternative feature detector and matcher option for nerfstudio.
RUN git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git && \
    cd Hierarchical-Localization && \
    git checkout v1.4 && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

# Install pyceres from source
RUN git clone --branch v1.0 --recursive https://github.com/cvg/pyceres.git && \
    cd pyceres && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

# Install pixel perfect sfm.
RUN git clone --recursive https://github.com/cvg/pixel-perfect-sfm.git && \
    cd pixel-perfect-sfm && \
    git reset --hard 40f7c1339328b2a0c7cf71f76623fb848e0c0357 && \
    git clean -df && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

# Copy nerfstudio folder and give ownership to user.
COPY --chown=${USER_UID}:${USER_GID} . /home/${USERNAME}/nerfstudio

# Install nerfstudio dependencies.
RUN cd nerfstudio && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

# Switch to workspace folder and install nerfstudio cli auto completion
WORKDIR /workspace
RUN ns-install-cli --mode install

# Bash as default entrypoint.
CMD /bin/bash -l
# Force changing password on first container run
# Change line above: CMD /bin/bash -l -> CMD /bin/bash -l -c passwd && /usr/bin/bash -l
