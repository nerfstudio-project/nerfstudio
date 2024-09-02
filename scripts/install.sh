#!/bin/bash

# Ensure conda is initialized
eval "$(conda shell.bash hook)"

echo "======Phase 1: Setting up Conda environment======"
conda create --name homee_nerfstudio -y python=3.8
conda activate homee_nerfstudio


echo "======Phase 2: Installing Nerfstudio dependencies======"
pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install setuptools==69.5.1
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# may take a while to build tinycudann...

echo "======Phase 3: Installing HLOC======"
pip install git+https://github.com/cvg/Hierarchical-Localization.git

echo "======Phase 4: Installing COLMAP dependencies======"
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

echo "======Phase 5: Building and installing COLMAP======"
git clone https://github.com/homee-ai/colmap.git
cd colmap
mkdir build
cd build
cmake .. -GNinja
ninja
sudo ninja install

cd ../..
echo "======Phase 6: Installing Nerfstudio======"
pip install -e .