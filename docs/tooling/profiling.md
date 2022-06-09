# Profiling and debugging

## Profilers
We provide built-in performance profiling capabilities to make it easier for you to debug and assess the performance of your code. 

#### In-house profiler
You can use our built-in profiler by enabling the profiler in the config:

```
# e.g. configs/graphs_default.yaml

logging:
    enable_profiler: True
```

The profiler computes the average total time of execution for any function with the `@profiler.time_function` decorator. 
For instance, if you wanted to profile the total time it takes to generate rays given pixel and camera indices via the `RayGenerator` class, you would want to evaluate its `forward()` function. In that class, you would need to add the decorator to the function.


```
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

```
pip install py-spy
```

To perform the profiling, you can either specify that you want to generate a flame graph or generate a live-view of the profiler.

```
## for flame graph

./scripts/debugging/profile.sh -t flame -o flame.svg -p <python_command>

## for live view of functions

./scripts/debugging/profile.sh -t top -p <python_command> 
```

In both cases, `<python_command>` can be replaced with any command you would normally run in the terminal. For instance, you can replace `<python_function>` with `scripts/run_train.py data/dataset=blender_lego`. Then the full command would be e.g.:

```
./scripts/debugging/profile.sh -t flame -o flame.svg -p scripts/run_train.py data/dataset=blender_lego
```

## Debugging