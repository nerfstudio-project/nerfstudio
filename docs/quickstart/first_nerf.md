# Training your first model

## Downloading data

Download datasets provided by nerfstudio. We support the major datasets and allow users to create their own dataset, described in detail [here](./custom_dataset.md).

```
ns-download-data --dataset=blender
ns-download-data --dataset=nerfstudio --capture=poster
```

:::{admonition} Tip
:class: info

Use `ns-download-data --help` to view all currently available datasets.
:::

The resulting script should download and unpack the dataset as follows:

```
|─ nerfstudio/
   ├─ data/
   |  ├─ blender/
   |     ├─ fern/
   |     ├─ lego/
         ...
      |- nerfstudio/
         |- poster
         ...
```

## Training a model

See which models are available.

```bash
ns-train --help
```

Run a vanilla nerf model.

```bash
ns-train vanilla-nerf
```

Run a nerfacto model.

```bash
ns-train nerfacto
```

Run a nerfacto model with different data and port.

```
ns-train nerfacto --vis viewer --data data/nerfstudio/poster --viewer.websocket-port 7007
```

Run a nerfacto model and load the latest checkpoint to resume training.

```
ns-train nerfacto --vis viewer --data data/nerfstudio/poster --trainer.load_dir outputs/data-nerfstudio-poster/nerfacto/{timestamp}/nerfstudio_models
```

:::{admonition} Warning
:class: warning

- You may have to change the ports, and be sure to forward the "websocket-port".
- All data configurations must go at the end. In this case, `nerfstudio-data` and all of its corresponding configurations come at the end after the model and viewer specification.
  :::

## Intro to nerfstudio CLI and Configs

Nerfstudio allows customization of training and eval configs from the CLI in a powerful way, but there are some things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following commands:

```bash
ns-train -h
```

```bash
ns-train nerfacto -h nerfstudio-data
```

```bash
ns-train nerfacto nerfstudio-data -h
```

In each of these examples, the -h applies to the previous subcommand (`ns-train`, `nerfacto`, and `nerfstudio-data`).

In the first example, we get the help menu for the `ns-train` script.

In the second example, we get the help menu for the `nerfacto` model.

In the third example, we get the help menu for the `nerfstudio-data` dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your arguments matters! Any optional arguments you discover from running

```bash
ns-train nerfacto -h nerfstudio-data
```

need to come directly after the `nerfacto` subcommand since these optional arguments only belong to the `nerfacto` subcommand:

```bash
ns-train nerfacto <nerfacto optional args> nerfstudio-data
```

Each script will have some other minor quirks (like the training script dataparser subcommand needing to come after the model subcommand), read up on them [here](../reference/cli/index.md).

## Visualizing training runs

If you are using a fast NeRF variant (ie. Nerfacto/Instant-NGP), we recommend using our viewer. See our [viewer docs](viewer_quickstart.md) for a tutorial on using the viewer. The viewer will allow interactive, real-time visualization of training.

For slower methods where the viewer is not recommended, we default to [Wandb](https://wandb.ai/site) to log all training curves, test images, and other stats. We also support logging with [Tensorboard](https://www.tensorflow.org/tensorboard). We pre-specify default `--vis` options depending on the model.

:::{admonition} Attention
:class: attention

- Currently we only support using a single viewer at a time.
- To toggle between Wandb, Tensorboard, or our Web-based Viewer, you can specify `--vis VIS_OPTION`, where `VIS_OPTION` is one of {viewer,wandb,tensorboard}.
  :::

#### Rendering videos

We also provide options to render out the scene of a trained model with a custom trajectory and save the output to a video.

```bash
ns-render --load-config={PATH_TO_CONFIG} --traj=spiral --output-path=renders/output.mp4
```

While we provide pre-specified trajectory options, `--traj={spiral, interp, filename}` we can also specify a custom trajectory if we specify `--traj=filename --camera-path-filename {PATH}`.

:::{admonition} Tip
:class: info
After running the training, the config path is logged to the terminal under "[base_config.py:263] Saving config to:"
:::

:::{admonition} See Also
:class: seealso
This quickstart allows you to preform everything in a headless manner.
We also provide a web-based viewer that allows you to easily monitor training or render out custom trajectories.
See our [viewer docs](viewer_quickstart.md) for more.
:::

## Evaluating Runs

Calculate the psnr of your trained model and save to a json file.

```bash
ns-eval --load-config={PATH_TO_CONFIG} --output-path=output.json
```

We also provide a train and evaluation script that allows you to do benchmarking on the classical Blender dataset (see our [benchmarking workflow](../developer_guides/debugging_tools/benchmarking.md)).

## Multi-GPU Training

Here we explain how to use multi-GPU training. This is the command to train the nerfacto model with 4 GPUs. We are using [PyTorch Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/beginner/dist_overview.html), so we scale the learning rate with the number of GPUs used. Plotting will only be done for the first process. Note that you may want to play around with both learning rate and `<X>_num_rays_per_batch` when using DDP. Below is a simple example for how you'd run the vanilla-nerf method with either 1 or 4 GPUs. We don't see much value at the moment for running the fast methods like nerfacto with more than one GPU.

```python
# 1 GPU (1x LR)
export CUDA_VISIBLE_DEVICES=0
ns-train vanilla-nerf \
  --machine.num-gpus 1 \
  --vis wandb \
  --optimizers.fields.optimizer.lr 5e-4

# 4 GPUs (4x LR)
export CUDA_VISIBLE_DEVICES=0,1,2,3
ns-train vanilla-nerf \
  --machine.num-gpus 4 \
  --vis wandb \
  --optimizers.fields.optimizer.lr 20e-4
```
