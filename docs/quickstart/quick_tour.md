# Walk-through tour

In this quick tour, we will walk you through the core of training and building any NeRFs with pyrad.

#### Launching a training job

The entry point for training starts at `scripts/run_train.py`, which spawns instances of our `Trainer()` class (in `nerf/trainer.py`). The `Trainer()` is responsible for setting up the datasets and NeRF graph depending on the config specified. It will then run the usual train/val routine for a config-specified number of iterations. If you are planning on using our codebase to build a new NeRF method or to use an existing implementation, we've abstracted away the training routine in these two files and chances are you will not need to think of them again.

#### Graphs, Fields, and Modules

If you are looking to implemnet a new NeRF method or extend an existing one, you only need to edit files in `nerf/graph/`, `nerf/fields/`, `nerf/field_modules/`, `nerf/misc_modules/`. (TODO: restructuring)

The actual NeRF graph definitions can be found in `nerf/graph/`. For instance, to implement the vanilla NeRF, we create a new class that inherits the abstract Graph class. To fully implement the any new graph class, you will need to implement the following abstract methods defined in the skeleton code below. See also `nerf/graph/vanilla_nerf.py` for the full implementation.

```
class NeRFGraph(Graph):
    """Vanilla NeRF graph"""

    def __init__(self, intrinsics=None, camera_to_world=None, **kwargs) -> None:
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_fields(self):
        """
        Set all field related modules here
        """

    def populate_misc_modules(self):
        """
        Set all remaining modules here including: samplers, renderers, losses, and metrics
        """

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """
        Create a dictionary of parameters that are grouped according to different optimizers
        """

    def get_outputs(self, ray_bundle: RayBundle):
        """
        Takes in a Ray Bundle and returns a dictionary of outputs.
        """

    def get_loss_dict(self, outputs, batch):
        """
        Computes and returns the losses.
        """

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        """
        Writes the test image outputs.
        """
```

Note that the graph is composed of fields and modules.

**Fields** (`nerf/fields/`) represents the actual radiance field of the NeRF and is composed of field modules (`nerf/field_modules/`). Here, we define the field as the part of the network that takes in point samples and any other conditioning, and outputs any of the `FieldHeadNames` (`nerf/field_modules/field_heads.py`). The **misc. modules** can be any module outside of the field that are needed by the NeRF (e.g. losses, samplers, renderers).

To get started on a new NeRF implementation, you simply have to define all relevant modules and populate them in the graph.

#### Dataset population TODO(ethan)

#### Config

Now that you have the graph and dataset all set up, you're ready to create the config that you pass into our run train routine. Our config system is powered by [Hydra](https://hydra.cc/). All Hydra and machine related arguments are stored in `configs/default_setup.yaml`, as well as the defaults list.
To set up the graph config, create a new yaml under `configs/`.

```
# configs/vanilla_nerf.yaml

defaults:
  - default_setup # inherit the basic yaml heirarchy
  - _self_

experiment_name: blender_lego
method_name: vanilla_nerf

graph:
    network:
        _target_: pyrad.graph.vanilla_nerf.NeRFGraph # set the target to the graph you defined

    # <insert any additional graph related overrides here>

data:
    # <insert any additional dataset related overrides here>
```

Once you have the config properly set up, you can begin training! Note, you can also pass in the config changes via command-line as shown above in the quick-start if you don't want to make a new config for a given job.

```
python scripts/run_train.py --config-name vanilla_nerf
```
