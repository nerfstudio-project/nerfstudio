<h1 align="center"> pyRad🤘 </h1>

<p align="center"> The all-in-one repo for NeRFs </p>

<p align="center"> 
    <a href="https://plenoptix-pyrad.readthedocs-hosted.com/en/latest/?badge=latest">
        <img alt="Documentation Status" src="https://readthedocs.com/projects/plenoptix-pyrad/badge/?version=latest">
    </a>
    <!-- TODO: add license and have it point to that -->
    <a href="https://github.com/plenoptix/pyrad/blob/master/LICENSE">
        <img alt="Documentation Status" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
    </a> 
    <!-- TODO: add version number badge -->
</p>

- [Quickstart](#quickstart)
- [Supported Features](#supported-features)
- [Benchmarked Model Architectures](#benchmarked-model-architectures)

# Quickstart

The quickstart will help you get started with the default vanilla nerf trained on the classic blender lego scene.
For more complex changes (e.g. running with your own data/ setting up a new NeRF graph, please see our [docs](https://plenoptix-pyrad.readthedocs-hosted.com/en/latest/quickstart/quick_tour.html).

#### 1. Installation: Setup the environment

This repository is tested with cuda 11.3

```
# Clone the repo
git clone --recurse-submodules git@github.com:plenoptix/pyrad.git

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

To run with all the defaults, e.g. vanilla nerf method with the blender lego image:

```
# Run with default config
python scripts/run_train.py
```

With support for [Hydra](https://hydra.cc/), you can run with other configurations by changing appropriate configs defined in `configs/` or by setting flags via command-line arguments:

```
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

#### 4. Visualizing training runs

If you run everything with the default configuration, by default, we use [TensorBoard](https://www.tensorflow.org/tensorboard) to log all training curves, test images, and other stats. Once the job is launched, you will be able to track training by launching the tensorboard in `outputs/blender_lego/vanilla_nerf/<timestamp>/<events.tfevents>`.

```
tensorboard --logdir outputs/blender_lego/vanilla_nerf/
```

#### 5. Rendering a trajectories during inference

TODO(ethan)

#### 6. In-depth guide

For a more in-depth tutorial on how to modify/implement your own NeRF Graph, please see our [walk-through](#).

# Supported Features

We provide the following support strucutures to make life easier for getting started with NeRFs. For a full description, please refer to our [features page](#).

If you are looking for a feature that is not currently supported, please do not hesitate to contact the Plenoptix team!

#### :metal: Support for [Hydra](https://hydra.cc/) config structure

#### :metal: Support for multiple logging interfaces

#### :metal: Built-in support for profiling code

#### :metal: Benchmarking scripts

#### :metal: Easily run other repos with our data

#### :metal: Speed up your code with Tiny Cuda

#### :metal: Support for Jupyter

# Benchmarked Model Architectures

| Method                                                                            | PSNR                     |
| --------------------------------------------------------------------------------- | ------------------------ |
| [NeRF](https://arxiv.org/abs/2003.08934)                                          | :hourglass_flowing_sand: |
| [instant NGP](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) | :hourglass_flowing_sand: |
| [Mip NeRF](https://arxiv.org/abs/2103.13415)                                      | :hourglass_flowing_sand: |
