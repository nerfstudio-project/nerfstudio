FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# setup
RUN apt-get update && apt-get install -y wget curl bzip2 git && apt-get clean

# miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

# conda environment
RUN conda create -n nerfstudio -y python=3.8

# PyTorch, torchvision, ninja 
RUN /bin/bash -c "source $CONDA_DIR/bin/activate nerfstudio && \
    python -m pip install --upgrade pip && \
    pip install ninja torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"

# cuda arquitectures
ENV TCNN_CUDA_ARCHITECTURES="70,75,80,86,89"

# tiny-cuda-nn 
RUN /bin/bash -c "source $CONDA_DIR/bin/activate nerfstudio && \
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# pixi 
RUN /bin/bash -c "source $CONDA_DIR/bin/activate nerfstudio && \
    curl -fsSL https://pixi.sh/install.sh | bash && \
    source ~/.bashrc && \
    pixi --version"

RUN echo "source activate nerfstudio" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]
CMD ["bash", "-l"]

# paso 16 desde el shell
# git clone https://github.com/nerfstudio-project/nerfstudio.git
# cd nerfstudio
# pixi run post-install
# pixi shell
# apt update
# apt install libgl1-mesa-glx -y
# pixi run train-example-nerf
# http://localhost:7007

# pip install gsplat==1.4.0
# python -c "import gsplat; print('gsplat installed successfully')"

# Process video data
# ns-process-data video --data ./pista.MP4 --sfm-tool hloc --output-dir ./processed
# Train model
# ns-train splatfacto --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.stop-screen-size-at=15000 --pipeline.model.use_scale_regularization=True --data ./processed
# view training
# ns-viewer --load-config outputs/processed/splatfacto/2025-03-11_210124/config.yml`
# http://0.0.0.0:7007
#  remplace 0.0.0.0 -> docker inspect container_id | grep "IPAddress"

# change swap memory to 15 gb (if needed)