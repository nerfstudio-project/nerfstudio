# pyRad :metal:

[![Documentation Status](https://readthedocs.com/projects/plenoptix-pyrad/badge/?version=latest)](https://plenoptix-pyrad.readthedocs-hosted.com/en/latest/?badge=latest)

The all-in-one repo for NeRFs

# Quickstart

#### 1. Installation: Setup the environment

This repository is tested with cuda 11.3
```
# Clone the repo
git clone --recurse-submodules git@github.com:ethanweber/pyrad.git

# Create the python environment
conda create --name pyrad python=3.8.13
conda activate pyrad
pip install -r environment/requirements.txt

# Install pyrad as a library
pip install -e .

# Install library with CUDA support. Change setup.py to `USE_CUDA = True` and then
python setup.py develop

# Running the test cases
pytest tests
```

#### 2. Getting the data

Download the original [NeRF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and unfold it in the following format:

```
├── data/
|   ├── blender/
|   |   ├── fern/
|   |   ├── lego/
...
```

#### 3. Training a model

```
# Run with default config
python scripts/run_train.py

# Run with config changes
python scripts/run_train.py machine_config.num_gpus=1
python scripts/run_train.py data.dataset.downscale_factor=1

# Run with different datasets
python scripts/run_train.py data/dataset=blender_lego
python scripts/run_train.py data/dataset=friends_TBBT-big_living_room

# Run with different datasets and config changes
python scripts/run_train.py data/dataset=friends_TBBT-big_living_room graph.network.far_plane=14

# [Experimental] Speed up the dataloading pipeline by caching DatasetInputs.
python scripts/run_data_preprocessor.py
# Then, specify using the cache.
python scripts/run_train.py ++data.dataset.use_cache=true
```

# Walk-through tour
In this quick tour, we will walk you through the core of training and building any NeRFs with pyrad.

#### Launching a training job
The entry point for training starts at `scripts/run_train.py`, which spawns instances of our `Trainer()` class (in `nerf/trainer.py`). The `Trainer()` is responsible for setting up the datasets and NeRF graph depending on the config specified. It will then run the usual train/val routine for a config-specified number of iterations. If you are planning on using our codebase to build a new NeRF method or to use an existing implementation, we've abstracted away the training routine in these two files and chances are you will not need to think of them again.

#### Graphs, Fields, and Modules
The actual NeRF graph definitions can be found in `nerf/graph/`. For instance, to implement the vanilla NeRF, we create a new [PICKOFF HERE]

#### 

# Feature List
#### :metal: [Hydra config structure](./configs/README.md)
#### :metal: [Logging, debugging utilities](./radiance/utils/README.md)
#### :metal: [Benchmarking, other tooling](./scripts/README.md)

#### :metal: Running other repos with our data

```
# nerf-pytorch
cd external
python run_nerf.py --config configs/chair.txt --datadir /path/to/pyrad/data/blender/chair

# jaxnerf
cd external
conda activate jaxnerf
python -m jaxnerf.train --data_dir=/path/to/pyrad/data/blender/chair --train_dir=/path/to/pyrad/outputs/blender_chair_jaxnerf --config=/path/to/pyrad/external/jaxnerf/configs/demo --render_every 100
```

#### :metal: Speeding up the code
Documentation for running the code with CUDA.
Please see https://github.com/NVlabs/tiny-cuda-nn for how to install tiny-cuda-nn.

```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

To run instant-ngp with tcnn, you can do the following. This is with the fox dataset.

```
python scripts/run_train.py --config-name=instant_ngp_tcnn.yaml data/dataset=instant_ngp_fox
```


#### :metal: Setting up Jupyter

```
python -m jupyter lab build
bash environments/run_jupyter.sh
```

# Benchmarked Model Architectures
| Method                                                                            | PSNR |
| --------------------------------------------------------------------------------- | ---- |
| [NeRF](https://arxiv.org/abs/2003.08934)                                          |      |
| [instant NGP](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) |      |
| [Mip NeRF](https://arxiv.org/abs/2103.13415)                                      |      |
