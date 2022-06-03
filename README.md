<h1 align="center"> pyRad🤘 </h1>

<p align="center"> The all-in-one repo for NeRFs </p>

<p align="center"> 
    <a href="https://plenoptix-pyrad.readthedocs-hosted.com/en/latest/?badge=latest">
        <img alt="Documentation Status" src="https://readthedocs.com/projects/plenoptix-pyrad/badge/?version=latest">
    </a>
    <!-- TODO: add license and have it point to that -->
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img alt="Documentation Status" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
    </a> 
    <!-- TODO: add version number badge -->
</p>


* [Quickstart](#quickstart)
* [Walk-through tour](#walk-through-tour)
* [Feature List](#feature-list)
* [Benchmarked Model Architectures](#benchmarked-model-architectures)

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
If you are looking to implemnet a new NeRF method or extend an existing one, you only need to edit files in `nerf/graph/`, `nerf/fields/`, `nerf/field_modules/`, `nerf/misc_modules/`. (TODO: restructuring)


The actual NeRF graph definitions can be found in `nerf/graph/`. For instance, to implement the vanilla NeRF, we create a new class that inherits the abstract Graph class. To fully implement the any new graph class, you will need to implement the following abstract methods defined in the skeleton code below. See also `nerf/graph/vanilla_nerf.py` for the full implementation.

```
class NeRFGraph(Graph):
    """Vanilla NeRF graph"""

    def __init__(self, intrinsics=None, camera_to_world=None, **kwargs) -> None:
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_fields(self):
        """
        Set all field related modules here
        """

    def populate_misc_modules(self):
        """
        Set all remaining modules here including: samplers, renderers, losses, and metrics
        """

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """
        Create a dictionary of parameters that are grouped according to different optimizers
        """

    def get_outputs(self, ray_bundle: RayBundle):
        """
        Takes in a Ray Bundle and returns a dictionary of outputs.
        """
       
    def get_loss_dict(self, outputs, batch):
        """
        Computes and returns the losses.
        """

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        """
        Writes the test image outputs.
        """
```

Note that the graph is composed of fields and modules. 

**Fields** (`nerf/fields/`) represents the actual radiance field of the NeRF and is composed of field modules (`nerf/field_modules/`). Here, we define the field as the part of the network that takes in point samples and any other conditioning, and outputs any of the `FieldHeadNames` (`nerf/field_modules/field_heads.py`). The **misc. modules** can be any module outside of the field that are needed by the NeRF (e.g. losses, samplers, renderers). 

To get started on a new NeRF implementation, you simply have to define all relevant modules and populate them in the graph. 

#### Dataset population TODO(ethan)

#### Config 
Now that you have the graph and dataset all set up, you're ready to create the config that you pass into our run train routine. Our config system is powered by [Hydra](https://hydra.cc/). All Hydra and machine related arguments are stored in `configs/default_setup.yaml`, as well as the defaults list. 
To set up the graph config, create a new yaml under `configs/`.

```
# configs/vanilla_nerf.yaml

defaults:
  - default_setup # inherit the basic yaml heirarchy
  - _self_

experiment_name: blender_lego
method_name: vanilla_nerf

graph:
    network:
        _target_: pyrad.nerf.graph.vanilla_nerf.NeRFGraph # set the target to the graph you defined

    # <insert any additional graph related overrides here>

data:
    # <insert any additional dataset related overrides here>
```

Once you have the config properly set up, you can begin training! Note, you can also pass in the config changes via command-line as shown above in the quick-start if you don't want to make a new config for a given job.

```
python scripts/run_train.py --config-name vanilla_nerf
```


# Feature List
#### :metal: [Hydra config structure](#)
#### :metal: [Logging, debugging utilities](#)
#### :metal: [Benchmarking, other tooling](#)

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
| Method                                                                            | PSNR                     |
| --------------------------------------------------------------------------------- | ------------------------ |
| [NeRF](https://arxiv.org/abs/2003.08934)                                          | :hourglass_flowing_sand: |
| [instant NGP](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) | :hourglass_flowing_sand: |
| [Mip NeRF](https://arxiv.org/abs/2103.13415)                                      | :hourglass_flowing_sand: |
