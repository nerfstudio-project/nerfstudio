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

## Resume from checkpoint

It is possible to load a pretrained model by running

```bash
ns-train nerfacto --data data/nerfstudio/poster --load-dir {outputs/.../nerfstudio_models}
```

## Visualize existing run

Given a pretrained model checkpoint, you can start the viewer by running

```bash
ns-viewer --load-config {outputs/.../config.yml}
```

## Exporting Results

Once you have a NeRF model you can either render out a video or export a point cloud.

### Render Video

First we must create a path for the camera to follow. This can be done in the viewer under the "RENDER" tab. Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA". This will set the first camera key frame. Continue to new viewpoints adding additional cameras to create the camera path. We provide other parameters to further refine your camera path. Once satisfied, press "RENDER" which will display a modal that contains the command needed to render the video. Kill the training job (or create a new terminal if you have lots of compute) and run the command to generate the video.

Other video export options are available, learn more by running

```bash
ns-render --help
```

### Generate Point Cloud

While NeRF models are not designed to generate point clouds, it is still possible. Navigate to the "EXPORT" tab in the 3D viewer and select "POINT CLOUD". If the crop option is selected, everything in the yellow square will be exported into a point cloud. Modify the settings as desired then run the command at the bottom of the panel in your command line.

Alternatively you can use the CLI without the viewer. Learn about the export options by running

```bash
ns-export pointcloud --help
```

## Intro to nerfstudio CLI and Configs

Nerfstudio allows customization of training and eval configs from the CLI in a powerful way, but there are some things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following commands:

The following will list the supported models

```bash
ns-train --help
```

Applying `--help` after the model specification will provide the model and training specific arguments.

```bash
ns-train nerfacto --help
```

At the end of the command you can specify the dataparser used. By default we use the _nerfstudio-data_ dataparser. We include other dataparsers such as _Blender_, _NuScenes_, ect. For a list of dataparse specific arguments, add `--help` to the end of the command

```bash
ns-train nerfacto <nerfacto optional args> nerfstudio-data --help
```

Each script will have some other minor quirks (like the training script dataparser subcommand needing to come after the model subcommand), read up on them [here](../reference/cli/index.md).

## Comet / Tensorboard / WandB / Viewer

We support four different methods to track training progress, using the viewer [tensorboard](https://www.tensorflow.org/tensorboard), [Weights and Biases](https://wandb.ai/site), and [Comet](https://comet.com/?utm_source=nerf&utm_medium=referral&utm_content=nerf_docs). You can specify which visualizer to use by appending `--vis {viewer, tensorboard, wandb, comet, viewer+wandb, viewer+tensorboard, viewer+comet}` to the training command. Simultaneously utilizing the viewer alongside wandb or tensorboard may cause stuttering issues during evaluation steps. The viewer only works for methods that are fast (ie. nerfacto, instant-ngp), for slower methods like NeRF, use the other loggers.

## Evaluating Runs

Calculate the psnr of your trained model and save to a json file.

```bash
ns-eval --load-config={PATH_TO_CONFIG} --output-path=output.json
```

We also provide a train and evaluation script that allows you to do benchmarking on the classical Blender dataset (see our [benchmarking workflow](../developer_guides/debugging_tools/benchmarking.md)).

## Multi-GPU Training

Here we explain how to use multi-GPU training. We are using [PyTorch Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/beginner/dist_overview.html), so gradients are averaged over devices. If the loss scales depend on sample size (usually not the case in our implementation), we need to scale the learning rate with the number of GPUs used. Plotting will only be done for the first process. Note that you may want to play around with both learning rate and `<X>_num_rays_per_batch` when using DDP. Below is a simple example for how you'd run the `nerfacto-big` method on the aspen scene (see above to download the data), with either 1 or 2 GPUs. The `nerfacto-big` method uses a larger model size than the `nerfacto` method, so it benefits more from multi-GPU training.

First, download the aspen scene.

```python
ns-download-data nerfstudio --capture-name=aspen
```

```python
# 1 GPU (8192 rays per GPU per batch)
export CUDA_VISIBLE_DEVICES=0
ns-train nerfacto-big --vis viewer+wandb --machine.num-devices 1 --pipeline.datamanager.train-num-rays-per-batch 4096 --data data/nerfstudio/aspen
```

You would observe about ~70k rays / sec on NVIDIA V100.

```
Step (% Done)       Train Iter (time)    ETA (time)           Train Rays / Sec
-----------------------------------------------------------------------------------
610 (0.61%)         115.968 ms           3 h, 12 m, 6 s       72.68 K
620 (0.62%)         115.908 ms           3 h, 11 m, 58 s      72.72 K
630 (0.63%)         115.907 ms           3 h, 11 m, 57 s      72.73 K
640 (0.64%)         115.937 ms           3 h, 11 m, 59 s      72.71 K
650 (0.65%)         115.853 ms           3 h, 11 m, 49 s      72.76 K
660 (0.66%)         115.710 ms           3 h, 11 m, 34 s      72.85 K
670 (0.67%)         115.797 ms           3 h, 11 m, 42 s      72.80 K
680 (0.68%)         115.783 ms           3 h, 11 m, 39 s      72.81 K
690 (0.69%)         115.756 ms           3 h, 11 m, 35 s      72.81 K
700 (0.70%)         115.755 ms           3 h, 11 m, 34 s      72.81 K
```

By having more GPUs in the training, you can allocate batch size to multiple GPUs and average their gradients.

```python
# 2 GPUs (4096 rays per GPU per batch, effectively 8192 rays per batch)
export CUDA_VISIBLE_DEVICES=0,1
ns-train nerfacto --vis viewer+wandb --machine.num-devices 2 --pipeline.datamanager.train-num-rays-per-batch 4096 --data data/nerfstudio/aspen
```

You would get improved throughput (~100k rays / sec on two NVIDIA V100).

```
Step (% Done)       Train Iter (time)    ETA (time)           Train Rays / Sec
-----------------------------------------------------------------------------------
1910 (1.91%)        79.623 ms            2 h, 10 m, 10 s      104.92 K
1920 (1.92%)        79.083 ms            2 h, 9 m, 16 s       105.49 K
1930 (1.93%)        79.092 ms            2 h, 9 m, 16 s       105.48 K
1940 (1.94%)        79.364 ms            2 h, 9 m, 42 s       105.21 K
1950 (1.95%)        79.327 ms            2 h, 9 m, 38 s       105.25 K
1960 (1.96%)        79.473 ms            2 h, 9 m, 51 s       105.09 K
1970 (1.97%)        79.334 ms            2 h, 9 m, 37 s       105.26 K
1980 (1.98%)        79.200 ms            2 h, 9 m, 23 s       105.38 K
1990 (1.99%)        79.264 ms            2 h, 9 m, 28 s       105.29 K
2000 (2.00%)        79.168 ms            2 h, 9 m, 18 s       105.40 K
```

During training, the "Train Rays / Sec" throughput represents the total number of training rays it processes per second, gradually increase the number of GPUs and observe how this throughput improves and eventually saturates.
