FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Set the shell to /bin/bash instead of /bin/sh so conda works correctly.
# https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45
SHELL [ "/bin/bash", "--login", "-c" ]

# Prepare and empty machine for building
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    vim \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    wget

# Build and install ceres solver
RUN apt-get -y install \
    libatlas-base-dev \
    libsuitesparse-dev
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 1.14.0
RUN cd ceres-solver && \
	mkdir build && \
	cd build && \
	cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
	make -j4 && \
	make install

# Clear space # https://askubuntu.com/questions/179955/var-lib-apt-lists-is-huge
RUN rm -rf /var/lib/apt/lists/*

# Build and install COLMAP

# Note: This Dockerfile has been tested using COLMAP pre-release 3.6.
# Later versions of COLMAP (which will be automatically cloned as default) may
# have problems using the environment described thus far. If you encounter
# problems and want to install the tested release, then uncomment the branch
# specification in the line below
RUN git clone https://github.com/colmap/colmap.git #--branch 3.6

RUN cd colmap && \
	git checkout dev && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j4 && \
	make install

# Install Conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Install friends conda environment
COPY ./run_install_friends_conda.sh /tmp/run_install_friends_conda.sh
COPY ./requirements_friends.txt /tmp/requirements_friends.txt
RUN yes "Y" | bash /tmp/run_install_friends_conda.sh
