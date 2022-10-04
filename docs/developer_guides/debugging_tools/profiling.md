# Code profiling support

We provide built-in performance profiling capabilities to make it easier for you to debug and assess the performance of your code.

#### In-house profiler

You can use our built-in profiler. By default, it is enabled and will print at the termination of the program. You can disable it via CLI using the flag `--logging.no-enable-profiler`.


The profiler computes the average total time of execution for any function with the `@profiler.time_function` decorator.
For instance, if you wanted to profile the total time it takes to generate rays given pixel and camera indices via the `RayGenerator` class, you might want to time its `forward()` function. In that case, you would need to add the decorator to the function.

```python
"""nerfstudio/model_components/ray_generators.py""""

class RayGenerator(nn.Module):

    ...

    @profiler.time_function     # <-- add the profiler decorator before the function
    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        # implementation here
        ...
```

At termination of training or end of the training run, the profiler will print out the average execution time for all of the functions that have the profiler tag.

:::{admonition} Tip
:class: info

Use this profiler if there are *specific/individual functions* you want to measure the times for.
  :::


#### Profiling with PySpy

If you want to profile the entire codebase, consider using [PySpy](https://github.com/benfred/py-spy).

Install PySpy

```bash
pip install py-spy
```

To perform the profiling, you can either specify that you want to generate a flame graph or generate a live-view of the profiler.

- flame graph: with wandb logging and our inhouse logging disabled
    ```bash
    program="ns-train nerfacto -- --vis=wandb --logging.no-enable-profiler blender-data"
    py-spy record -o {PATH_TO_OUTPUT_SVG} $program
    ```
- top-down stats: running same program configuration as above
    ```bash
    py-spy top $program
    ```
    
:::{admonition} Attention
:class: attention

In defining `program`, you will need to add an extra `--` before you specify your program's arguments.
  :::