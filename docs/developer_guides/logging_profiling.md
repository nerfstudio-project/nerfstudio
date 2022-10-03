# Loggers and Profilers

## Logging support

We provide integration with multiple logging interfaces to log images and statistics during training.
All of these loggers are configurable via the config:

```yaml
# e.g. configs/graphs_default.yaml

logging:
  steps_per_log: 10 # how often to log losses/images
  max_buffer_size: 20 # defines how many steps to average over (if reporting averages)
  writer: # definition of all writers you want to use
    TensorboardWriter:
      log_dir: './' # must specify the output path for all writers
    LocalWriter:
      log_dir: './'
      stats_to_track: [ITER_TRAIN_TIME, TRAIN_RAYS_PER_SEC, CURR_TEST_PSNR]
      max_log_size: 10
```

Currently, we provide support for the following loggers, which can be added to the `logging.writer` section of the config.

#### Tensorboard

To log training stats and images with [tensorboard](https://www.tensorflow.org/tensorboard):

```
TensorboardWriter:
    log_dir: "./" # the output path for which the event files are written
```

#### Weights and Biases

To log training stats and images with [wandb](https://wandb.ai/site):

```
WandbWriter:
    log_dir: "./" # the output path for which the run files are written
```

#### Local writer

```
LocalWriter:
    log_dir: "./" # if images are saved, where the images would be saved
    stats_to_track:
        [ITER_TRAIN_TIME, TRAIN_RAYS_PER_SEC, CURR_TEST_PSNR]
    max_log_size: 10
```

The `LocalWriter` simply outputs to the terminal. However, you can specify additional parameters to customize the logging.

- `stats_to_track`: all the stats that you want to print to the terminal (see list under `EventName` in `utils/writer.py`). You can add or remove any of the defined enums.
- `max_log_size`: how much content to print onto the screen (e.g. only print 10 lines on the screen at a time). If 0, will print everything without deleting any previous lines.

## Code profiling support

We provide built-in performance profiling capabilities to make it easier for you to debug and assess the performance of your code.

#### In-house profiler

You can use our built-in profiler by enabling the profiler in the config:

```yaml
# e.g. configs/graphs_default.yaml

logging:
  enable_profiler: True
```

The profiler computes the average total time of execution for any function with the `@profiler.time_function` decorator.
For instance, if you wanted to profile the total time it takes to generate rays given pixel and camera indices via the `RayGenerator` class, you would want to evaluate its `forward()` function. In that class, you would need to add the decorator to the function.

```python
class RayGenerator(nn.Module):

    ...

    @profiler.time_function     # add the profiler decorator before the function
    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        # implementation here
        ...
```

At termination of training or end of the training run, the profiler will print out the average execution time for all of the functions that have the profiler tag.

Use this profiler if there are specific functions that you want to measure the times for.

#### Profiling with PySpy

If you want to profile the entire codebase, we provide functionality to generate flame graphs or get real-time reports using [PySpy](https://github.com/benfred/py-spy).

Install PySpy

```bash
pip install py-spy
```

To perform the profiling, you can either specify that you want to generate a flame graph or generate a live-view of the profiler.

```bash
## for flame graph

./scripts/debugging/profile.sh -t flame -o flame.svg -p <python_command>

## for live view of functions

./scripts/debugging/profile.sh -t top -p <python_command>
```

In both cases, `<python_command>` can be replaced with any command you would normally run in the terminal. For instance, you can replace `<python_function>` with `scripts/train.py data/dataset=blender_lego`. Then the full command would be e.g.:

```bash
./scripts/debugging/profile.sh -t flame -o flame.svg -p scripts/train.py data/dataset=blender_lego
```
