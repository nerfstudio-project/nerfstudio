# MattPort

# Quickstart

## Installation: Setup the environment

```
# Clone the repo
git clone --recurse-submodules git@github.com:ethanweber/mattport.git

# Create the python environment
conda create --name mattport python=3.8.13
conda activate mattport
pip install -r environment/requirements.txt

# Install mattport as a library
pip install -e .

# Running the test cases
pytest tests
```

## Getting the data

Download the original NeRF dataset and put it in the following format.

```
- data/
    - blender/
        - fern/
        - lego/
```

## Training a model

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
```

# Logging and profiling features

We provide many logging functionalities for timing and/or tracking losses during training. All of these loggers are configurable via `configs/logging.yml`

1. **Writer**: Logs losses and generated images during training to a specified output stream. Specify the type of writer (Tensorboard, local directory, Weights and Biases), and how often to log in the config.

2. **StatsTracker**: Computes select statistics and prints to the terminal for local debugging. Specify the kinds of statistics to be logged, how often to log, and the maximum history buffer size in the config.

3. **Profiler**: Computes the average total time of execution for any function with the `@profiler.time_function` decorator. Prints out the full profile at the termination of training or the program.

# Setting up Jupyter

```
python -m jupyter lab build
bash environments/run_jupyter.sh
```

# Tooling

1. One can use [autoenv](https://github.com/hyperupcall/autoenv) to make setting up the environment and environment variables easier. This will run the `.envrc` file upon entering the `/path/to/mattport` folder. It will also remove the environment parameters upon leaving.

```
# Install direnv.
sudo apt install direnv

# Add the following line to the bottom of your ~/.bashrc file.
eval "$(direnv hook bash)"

# Populate your .envrc with commands you want to run. Then, run the following to allow updates.
cd /path/to/mattport
direnv allow .
```

2. To run local **github actions**, you can run:

```
python scripts/debugging/run_actions.py
```

3. To run local **profiling** to get a flame graph, make sure [pyspy](https://github.com/benfred/py-spy) is installed and you can run:

```
pip install py-spy

## for flame graph
./scripts/debugging/profile.sh -t flame -o flame.svg -p scripts/run_train.py data/dataset=blender_lego

## for live view of functions
./scripts/debugging/profile.sh -t top -p scripts/run_train.py data/dataset=blender_lego
```

4. For debugging with a debugger.

```
ipython --pdb scripts/run_train.py
```

5. **Benchmarking** For launching training jobs automatically on blender dataset

```
./scripts/benchmarking/launch_train_blender.sh <gpu0> <gpu1> ... <gpu7>
```