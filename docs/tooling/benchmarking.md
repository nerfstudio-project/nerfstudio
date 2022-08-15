# Benchmarking workflow


We make it easy to benchmark your new NeRF against the standard Blender dataset. 

(train)=
## Launching training on Blender dataset

To start, you will need to train your NeRF on each of the blender objects.
To launch training jobs automatically on each of these items, you can call:

```bash
./scripts/benchmarking/launch_train_blender.sh -c <config-name> -g <list-of-gpus>
```

Simply replace the arguments in brackets with the correct arguments.
* `-c`: name of the config you created for your new graph. For instance, if you have created the config `configs/graph_instant_ngp.yaml`, `<config-name>` will be `graph_instant_ngp`.
* `-g`: specify the list of gpus you want to use on your machine space separated. for instance, if you want to use GPU's 0-3, you will need to pass in `0 1 2 3`. If left empty, the script will automatically find available GPU's and distribute training jobs on the available GPUs.

A full example would be:

```bash
# with -g specifying gpus
./scripts/benchmarking/launch_train_blender.sh -c graph_instant_ngp -g 0 1 2 3

# without -g automatically finds available gpus
./scripts/benchmarking/launch_train_blender.sh -c graph_instant_ngp
```

The script will automatically launch training on all of the items and save the checkpoints in an output directory with the current date and timestamp.

(eval)=
## Evaluating trained Blender models
Once you have launched training, and training converges, you can test your method with `scripts/benchmarking/launch_eval_blender.sh`.

Say we ran a benchmark on 08-10-2022 for instant_ngp. By default, the train script will save the benchmarks in the following format:
```
outputs
└───blender_chair_08-10-2022
|   └───instant_ngp
|       └───2022-08-10_172517
|           └───run_train.log
|               ...
└───blender_drums_08-10-2022
|   └───instant_ngp
|       └───2022-08-10_172517
|           └───run_train.log
|               ...
...
```

If we wanted to run the benchmark on all the blender data for the above example, we would run:
```bash
./scripts/benchmarking/launch_eval_blender.sh -c graph_instant_ngp -o ./outputs/ -m 08 -d 10 -y 2022 -s 172517 -g 4 5 6 7
```

The flags used in the benchmarking script are defined as follows:
* `-c`: config name (e.g. `graph_instant_ngp`). This should be the same as what was passed in for -c in the train script.
* `-o`: base output directory for where all of the benchmarks are stored (e.g. `outputs/`). Corresponds to the hydra base dir.
* `-m`: month of benchmark (e.g. `08`) of format 02d. 
* `-d`: date of benchmark (e.g. `10`) of format 02d.
* `-y`: year of benchmark (e.g. `2022`).
* `-s`: seconds timestamp of benchmark (e.g. `172517`) of format 06d.
* `-g`: specifies the gpus to use and if not specified (no -g flag), will automaticaly search for available gpus.

The script will simultaneously run the benchmarking across all the objects in the blender dataset and calculates the PSNR/FPS/other stats. The results are saved as .json files in the `-o` directory with the following format:

```
outputs
└───instant_ngp
|   └───blender_chair_08-10-2022_172517.json
|   |   blender_ficus_08-10-2022_172517.json
|   |   ...
```