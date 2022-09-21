# Installation

This repository is tested with CUDA 11.3. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) before preceding.

</details>

Create the python environment

```bash
conda create --name nerfactory python=3.8.13
conda activate nerfactory
python -m pip install --upgrade pip
```

Clone the repo

```bash
git clone git@github.com:plenoptix/nerfactory.git
```

Install dependencies and nerfactory as a library

```bash
cd nerfactory
pip install -e .
```

Install tiny-cuda-nn (tcnn) to run instant_ngp

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Experimental: install tab completion for nerfactory (bash and zsh)

```bash
# If the CLI changes, we can also re-install to synchronize completions
python scripts/completions/configure.py install
```

# Downloading data

Download the original NeRF Blender dataset. We support the major datasets and allow users to create their own dataset, described in detail [here](https://plenoptix-nerfactory.readthedocs-hosted.com/en/latest/tutorials/data/index.html).

```
python scripts/downloads/download_data.py --dataset=blender
```

Use `--help` to view all currently available datasets. The resulting script should download and unpack the dataset as follows:

```
|─ nerfactory/
   ├─ data/
   |  ├─ blender/
   |     ├─ fern/
   |     ├─ lego/
         ...
      |- <dataset_format>/
         |- <scene>
         ...
```

# Training a model

To run with all the defaults, e.g. vanilla nerf method with the blender lego image

Run a vanilla nerf model.

```bash
python scripts/run_train.py
```

Run a faster version with instant ngp using tcnn (without the viewer).

```bash
python scripts/run_train.py --config-name=graph_instant_ngp.yaml
```

Run with the viewer. However, you'll have to start the viewer server first. (See [viewer docs](../tutorials/viewer/viewer_quickstart.md))

```bash
python scripts/run_train.py --config-name=graph_instant_ngp.yaml viewer.enable=true
```

With support for [Hydra](https://hydra.cc/), you can run with other configurations by changing appropriate configs defined in `configs/` or by setting flags via command-line arguments.

# Visualizing training runs

If you using a fast NeRF variant (ie. Instant-NGP), we reccomend using our viewer. See our [viewer docs](../tutorials/viewer/viewer_quickstart.md) for more details. The viewer will allow interactive visualization of training in realtime.

Additionally, if you run everything with the default configuration, by default, we use [TensorBoard](https://www.tensorflow.org/tensorboard) to log all training curves, test images, and other stats. Once the job is launched, you will be able to track training by launching the tensorboard in `outputs/blender_lego/vanilla_nerf/<timestamp>/<events.tfevents>`.

```bash
tensorboard --logdir outputs
```

# Rendering a Trajectory

To evaluate the trained NeRF, we provide an evaluation script that allows you to do benchmarking (see our [benchmarking workflow](../tooling/benchmarking.md)) or to render out the scene with a custom trajectory and save the output to a video.

```bash
python scripts/run_eval.py render-trajectory --load-config=outputs/blender_lego/instant_ngp/2022-07-07_230905/config.yml --traj=spiral --output-path=output.mp4
```

Please note, this quickstart allows you to preform everything in a headless manner. We also provide a web-based viewer that allows you to easily monitor training or render out trajectories. See our [viewer docs](../tutorials/viewer/viewer_quickstart.md) for more.

# FAQ

- [TinyCUDA installation errors out with cuda mismatch](tiny-cuda-error)

(tiny-cuda-error)=

#### TinyCUDA installation errors out with cuda mismatch

While installing tiny-cuda, you run into: `The detected CUDA version mismatches the version that was used to compile PyTorch (10.2). Please make sure to use the same CUDA versions.`

**Solution**:

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

- [Installation errors, File "setup.py" not found](pip-install-error)

(pip-install-error)=

#### Installation errors, File "setup.py" not found

When installing dependencies and nerfactory with `pip install -e .`, you run into: `ERROR: File "setup.py" not found. Directory cannot be installed in editable mode`

**Solution**:
This can be fixed by upgrading pip to the latest version:

```
python -m pip install --upgrade pip
```

- [Runtime errors, "len(sources) > 0".](cuda-sources-error)

(cuda-sources-error)=

#### Runtime errors, "len(sources) > 0".

When running `run_train.py `, an error occurs when installing cuda files in the backend code.

**Solution**:
This is a problem with not being able to detect the correct CUDA version, and can be fixed by updating the CUDA path environment variables:

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```