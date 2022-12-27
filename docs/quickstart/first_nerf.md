# Training your first model

## Train and run viewer

The following will train a _nerfacto_ model, our recommended model for real world scenes.

```bash
# Download some test data:
ns-download-data nerfstudio --capture-name=poster
# Train model
ns-train nerfacto --data data/nerfstudio/poster
```

If everything works, you should see training progress like the following:

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766069-cadfd34f-8833-4156-88b7-ad406d688fc0.png">
</p>

Navigating to the link at the end of the terminal will load the webviewer. If you are running on a remote machine, you will need to port forward the websocket port (defaults to 7007).

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766653-586a0daa-466b-4140-a136-6b02f2ce2c54.png">
</p>

:::{admonition} Note
:class: note

- You may have to change the port using `--viewer.websocket-port`.
- All data configurations must go at the end. In this case, `nerfstudio-data` and all of its corresponding configurations come at the end after the model and viewer specification.
  :::

## Resume from checkpoint / visualize existing run

It is possible to load a pretrained model by running

```bash
ns-train nerfacto --data data/nerfstudio/poster --load-dir {outputs/.../nerfstudio_models}
```

This will automatically start training. If you do not want it to train, add `--viewer.start-train False` to your training command.

## Exporting Results

Once you have a NeRF model you can either render out a video or export a point cloud.

### Render Video

First we must create a path for the camera to follow. This can be done in the viewer under the "RENDER" tab. Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA". This will set the first camera key frame. Continue to new viewpoints adding additional cameras to create the camera path. We provide other parameters to further refine your camera path. Once satisfied, press "RENDER" which will display a modal that contains the command needed to render the video. Kill the training job (or create a new terminal if you have lots of compute) and the command to generate the video.

Other video export options are available, learn more by running,

```bash
ns-render --help
```

### Generate Point Cloud

While NeRF models are not designed to generate point clouds, it is still possible. Navigate to the "EXPORT" tab in the 3D viewer and select "POINT CLOUD". If the crop option is selected, everything in the yellow square will be exported into a point cloud. Modify the settings as desired then run the command at the bottom of the panel in your command line.

Alternatively you can use the CLI without the viewer. Learn about the export options by running,

```bash
ns-export pointcloud --help
```

## Intro to nerfstudio CLI and Configs

Nerfstudio allows customization of training and eval configs from the CLI in a powerful way, but there are some things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following commands:

The following will list the supported models,

```bash
ns-train --help
```

Applying `--help` after the model specification will provide the model and training specific arguments.

```bash
ns-train nerfacto --help
```

At the end of the command you can specify the dataparser used. By default we use the _nerfstudio-data_ dataparser. We include other dataparsers such as _Blender_, _NuScenes_, ect. For a list of dataparse specific arguments, add `--help` to the end of the command,

```bash
ns-train nerfacto <nerfacto optional args> nerfstudio-data --help
```

Each script will have some other minor quirks (like the training script dataparser subcommand needing to come after the model subcommand), read up on them [here](../reference/cli/index.md).

## Tensorboard / WandB

We support three different methods to track training progress, using the viewer, [tensorboard](https://www.tensorflow.org/tensorboard), and [Weights and Biases](https://wandb.ai/site). You can specify which visualizer to use by appending `--vis {viewer, tensorboard, wandb}` to the training command. Note that only one may be used at a time. Additionally the viewer only works for methods that are fast (ie. nerfacto, instant-ngp), for slower methods like NeRF, use the other loggers.

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
