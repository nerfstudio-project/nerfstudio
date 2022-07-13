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

This repository is tested with CUDA 11.3. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) before proceding.

<details>
<summary>Installing Conda</summary>

    This step is fairly self-explanatory, but here are the basic steps. You can also find countless tutorials online.

    ```
    cd /path/to/install/miniconda

    mkdir -p miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
    bash miniconda3/miniconda.sh -b -u -p miniconda3
    rm -rf miniconda/miniconda.sh
    ```

</details>

```
# Create the python environment
conda create --name pyrad python=3.8.13
conda activate pyrad

# Clone the repo
git clone git@github.com:plenoptix/pyrad.git

# Install dependencies
cd pyrad
pip install -r environment/requirements.txt

# Install pyrad as a library
pip install -e .

# Install library with CUDA support. Change setup.py to `USE_CUDA = True` and then
python setup.py develop

# Install tiny-cuda-nn (tcnn) to use with the graph_instant_ngp.yaml config
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Run the test cases
pytest tests
```

#### 2. Getting the data

Download the original [NeRF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and unfold it in the following format. This is for the blender dataset type. We support the major datasets and allow users to create their own dataset, described in detail [here](docs/tutorials/data_setup.rst).

```
|─ pyrad/
   ├─ data/
   |  ├─ blender/
   |     ├─ fern/
   |     ├─ lego/
         ...
      |- <dataset_format>/
         |- <scene>
         ...
```

#### 3. Training a model

To run with all the defaults, e.g. vanilla nerf method with the blender lego image

```
# Run a vanilla nerf model.
python scripts/run_train.py

# Run a faster version with instant ngp using tcnn (without the viewer).
python scripts/run_train.py --config-name=graph_instant_ngp.yaml

# Run with the viewer. However, you'll have to start the viewer server first. (See the docs.)
python scripts/run_train.py --config-name=graph_instant_ngp.yaml viewer.enable=true
```

With support for [Hydra](https://hydra.cc/), you can run with other configurations by changing appropriate configs defined in `configs/` or by setting flags via command-line arguments.

#### 4. Visualizing training runs

If you run everything with the default configuration, by default, we use [TensorBoard](https://www.tensorflow.org/tensorboard) to log all training curves, test images, and other stats. Once the job is launched, you will be able to track training by launching the tensorboard in `outputs/blender_lego/vanilla_nerf/<timestamp>/<events.tfevents>`.

```
tensorboard --logdir outputs

# or the following
export TENSORBOARD_PORT=<port>
bash environment/run_tensorboard.sh
```

#### 5. Rendering a trajectories during inference

```
python scripts/run_eval.py --method=traj --traj=spiral --output-filename=output.mp4 --config-name=graph_instant_ngp.yaml trainer.resume_train.load_dir=outputs/blender_lego/instant_ngp/2022-07-07_230905/checkpoints
```

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
