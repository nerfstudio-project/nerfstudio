# Benchmarking workflow

We make it easy to benchmark your new NeRF against the standard Blender dataset.

## Launching training on Blender dataset

To start, you will need to train your NeRF on each of the blender objects.
To launch training jobs automatically on each of these items, you can call:

```bash

./nerfstudio/scripts/benchmarking/launch_train_blender.sh -m {METHOD_NAME} [-s] [-v {VIS}] [{GPU_LIST}]
```

Simply replace the arguments in brackets with the correct arguments.

- `-m {METHOD_NAME}`: Name of the method you want to benchmark (e.g. `nerfacto`, `mipnerf`).
- `-s`: Launch a single job per GPU.
- `-v {VIS}`: Use another visualization than wandb, which is the default. Other options are comet & tensorboard.
- `{GPU_LIST}`: (optional) Specify the list of gpus you want to use on your machine space separated. for instance, if you want to use GPU's 0-3, you will need to pass in `0 1 2 3`. If left empty, the script will automatically find available GPU's and distribute training jobs on the available GPUs.

:::{admonition} Tip
:class: info

To view all the arguments and annotations, you can run `./nerfstudio/scripts/benchmarking/launch_train_blender.sh --help`
:::

A full example would be:

- Specifying gpus

  ```bash
  ./nerfstudio/scripts/benchmarking/launch_train_blender.sh -m nerfacto 0 1 2 3
  ```

- Automatically find available gpus
  ```bash
  ./nerfstudio/scripts/benchmarking/launch_train_blender.sh -m nerfacto
  ```

The script will automatically launch training on all of the items and save the checkpoints in an output directory with the experiment name and current timestamp.

## Evaluating trained Blender models

Once you have launched training, and training converges, you can test your method with `nerfstudio/scripts/benchmarking/launch_eval_blender.sh`.

Say we ran a benchmark on 08-10-2022 for `instant-ngp`. By default, the train script will save the benchmarks in the following format:

```
outputs
└───blender_chair_2022-08-10
|   └───instant-ngp
|       └───2022-08-10_172517
|           └───config.yml
|               ...
└───blender_drums_2022-08-10
|   └───instant-ngp
|       └───2022-08-10_172517
|           └───config.yml
|               ...
...
```

If we wanted to run the benchmark on all the blender data for the above example, we would run:

```bash

./nerfstudio/scripts/benchmarking/launch_eval_blender.sh -m instant-ngp -o outputs/ -t 2022-08-10_172517 [{GPU_LIST}]
```

The flags used in the benchmarking script are defined as follows:

- `-m`: config name (e.g. `instant-ngp`). This should be the same as what was passed in for -c in the train script.
- `-o`: base output directory for where all of the benchmarks are stored (e.g. `outputs/`). Corresponds to the `--output-dir` in the base `Config` for training.
- `-t`: timestamp of benchmark; also the identifier (e.g. `2022-08-10_172517`).
- `-s`: Launch a single job per GPU.
- `{GPU_LIST}`: (optional) Specify the list of gpus you want to use on your machine space separated. For instance, if you want to use GPU's 0-3, you will need to pass in `0 1 2 3`. If left empty, the script will automatically find available GPU's and distribute evaluation jobs on the available GPUs.

The script will simultaneously run the benchmarking across all the objects in the blender dataset and calculates the PSNR/FPS/other stats. The results are saved as .json files in the `-o` directory with the following format:

```
outputs
└───instant-ngp
|   └───blender_chair_2022-08-10_172517.json
|   |   blender_ficus_2022-08-10_172517.json
|   |   ...
```

:::{admonition} Warning
:class: warning

Since we are running multiple backgrounded processes concurrently with this script, please note the terminal logs may be messy.
:::
