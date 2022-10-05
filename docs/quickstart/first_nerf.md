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

:::{admonition} Warning
:class: warning

* You may have to change the ports, and be sure to forward the "websocket-port".
* All data configurations must go at the end. In this case, `nerfstudio-data` and all of its corresponding configurations come at the end after the model and viewer specification.
  :::

## Intro to nerfstudio CLI and Configs
Nerfstudio allows for customizing your training runs, eval runs, configs from the CLI in a powerful way, but there are some things to understand.

The most demonstrative and helpful example of this in action is the difference in output between the following commands:
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

In the first example, this shows us the help menu for the `ns-train` script. 

In the second examples, we will get the help menu for the `nerfacto` model. 

In the third examples, we will get the help menu for the `nerfstudio-data` dataparser.

With our scripts, your arguments will apply to the previous subcommand in your command, and thus where you put your arguments matters! Any optional arguments you discover while doing
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
ns-render --load-config={PATH_TO_CONFIG} --traj=spiral --output-path=output.mp4
```

While we provide pre-specified trajectory options, `--traj={spiral, interp, filename}` we can also specify a custom trajectory if we specify `--traj=filename  --camera-path-filename {PATH}`. 

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