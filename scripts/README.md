# Tooling

1. One can use [autoenv](https://github.com/hyperupcall/autoenv) to make setting up the environment and environment variables easier. This will run the `.envrc` file upon entering the `/path/to/pyrad` folder. It will also remove the environment parameters upon leaving.

```
# Install direnv.
sudo apt install direnv

# Add the following line to the bottom of your ~/.bashrc file.
eval "$(direnv hook bash)"

# Populate your .envrc with commands you want to run. Then, run the following to allow updates.
cd /path/to/pyrad
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

For testing specific methods, see scripts/benchmarking/run_benchmark.py.
Modify the `BENCH` variable to specify which jobs ("ckpt_dir") and methods ("method") you want benchmarked. Then run

```
python scripts/benchmarking/run_benchmark.py
```