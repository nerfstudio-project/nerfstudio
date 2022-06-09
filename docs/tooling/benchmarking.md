# Benchmarking workflow

We make it easy to benchmark your new NeRF against the standard Blender dataset.

#### Launching training on Blender

To start, you will need to train your NeRF on each of the blender objects.
To launch training jobs automatically on each of these items, you can call:

```
./scripts/benchmarking/launch_train_blender.sh <config-name> <list_of_gpus>
```

Simply replace the arguments in brackets with the correct arguments.
* `<config-name>`: name of the config you created for your new graph. For instance, if you have created the config `configs/graph_vanilla_nerf.yaml`, `<config-name>` will be `graph_vanilla_nerf`.
* `<list_of_gpus>`: specify the list of gpus you want to use on your machine space separated. for instance, if you want to use GPU's 0-3, you will need to pass in `0 1 2 3`. If left empty, the script will automatically find available GPU's and distribute training jobs on the available GPUs.

A full example would be:

```
./scripts/benchmarking/launch_train_blender.sh graph_vanilla_nerf 0 1 2 3
```

The script will automatically launch training on all of the items and save the checkpoints in an output directory with the current date. 
You will use the date as the reference when you run inference for benchmarking.

#### Benchmarking on Blender
Once you have launched training, and training converges, you can test your method with `scripts/benchmarking/run_benchmark.py`.

Modify the `BENCH` variable to specify which jobs you want benchmarked. 

```
BENCH = {
    "method": "graph_vanilla_nerf",     # change this to whatever you want
    "hydra_base_dir": "outputs/",       # change this to wherever hydra's default output directory is
    "benchmark_date": "05-26-2022",     # change this to the date you ran the benchmarking
    "object_list": ["mic", "ficus", "chair", "hotdog", "materials", "drums", "ship", "lego"],
}
```

Then you can simply run

```
python scripts/benchmarking/run_benchmark.py

```

The script will search for the most recent checkpoint according to the benchmark configuration specified via `BENCH`. It will run inference and calculate the PSNR according to the respective object. The output is then saved in the `hydra_base_dir` directory as `<benchmark_date>.json` (full path: `<hydra_base_dir>/<benchmark_date>.json`).