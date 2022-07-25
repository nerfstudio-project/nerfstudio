# Developing a Custom Model

In this quick tour, we will walk you through the core of training and building any NeRFs with pyrad.

The entry point for training starts at `scripts/run_train.py`, which spawns instances of our `Trainer()` class (in `engine/trainer.py`). The `Trainer()` is responsible for setting up the datasets and NeRF graph depending on the config specified. It will then run the usual train/val routine for a config-specified number of iterations. If you are planning on using our codebase to build a new NeRF method or to use an existing implementation, we've abstracted away the training routine in these two files and chances are you will not need to think of them again.

## Graphs, Modules, and Fields

If you are looking to implemnet a new NeRF method or extend an existing one, you may need to set up a new graph (in `graphs/`).

All NeRF graph definitions can be found in `graphs/`. The graph encapsulates all the modules in the NeRF method- from the MLP's to the samplers to the losses and optimizers. All graphs are composed of modules and fields.

### Modules

We can decompose all graphs into respective modules. For instance, we have {ref}`optimizers` which contains modules related to the loss, {ref}`renderers` which contains all rendering modules, {`graph/modules/`}, which includes samplers and ray generators. Additionally, we have {ref}`field_modules`, which will be discussed below.
In general, modules can be thought of as individual component parts of the graph that can be swapped in/out. We provide most of the standard modules, but if you are creating a new NeRF graph, you may have to define new modules.

### Fields

{ref}`fields` represents the "space", aka. the radiance field of the NeRF. Here, we define the field as the part of the network that takes in point samples and any other conditioning, and outputs any of the `FieldHeadNames` (`nerf/field_modules/field_heads.py`).

All fields are composed of modules continained within the `fields/` directory (e.g. `fields/modules/` or `fields/density_fields/`). We can think of field modules as modules that actually define or interact with the field.

To build a NeRF field, we therefore follow these steps:

1. define any relevant field modules
2. create a new class that extends the {ref}`fields` base class and use these field modules to compose the full field
3. implement the abstract functions `get_density()` and `get_outputs()`.

```python
class NeRFField(Field):
    """NeRF Field"""

    def __init__(self, ...) -> None:
        """Create boilerplate NeRF field."""

        super().__init__()

        # In the init method, we are instantiating all relevant field modules
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
        )
        ...

    def get_density(self, point_samples: PointSamples):
        """
        Computes and returns the densities.
        """

    def get_outputs(
        self, point_samples: PointSamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        """
        Computes and returns the outputs.
        """
```

### Putting it together

Now that we've gone over what modules and fields are, we are ready to create the graph!

To implement the vanilla NeRF, we create a new class that inherits the abstract Graph class. Similar to composing a new field, we can pull from all pre-defined fields and modules and use them to compose a new Graph.
We will then need to define the following abstract methods as seen in the skeleton code below. See also `graphs/vanilla_nerf.py` for the full implementation.

```python
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

## Creating the config

Now that you have the graph and dataset all set up, you'll need to create the config that you pass into our run train routine. Our config system is powered by [Hydra](https://hydra.cc/). All Hydra and machine related arguments are stored in `configs/default_setup.yaml`, as well as the defaults list.
To set up the graph config, create a new yaml under `configs/`.

```yaml
# configs/vanilla_nerf.yaml

defaults:
  - graph_default # inherit the basic yaml heirarchy
  - _self_

logging:
  # <insert logging definitions here>

trainer:
  # <override configurations for loading/training/saving model here>

experiment_name: blender_lego
method_name: vanilla_nerf

data:
  # <insert any dataset related overrides here>

graph:
  _target_: pyrad.graph.vanilla_nerf.NeRFGraph # set the target to the graph you defined
  # <insert any graph related overrides here>

optimizers:
  # <insert any optimizer related overrides here>
```

## Training and testing

Once you have the config properly set up, you can begin training! To begin the training process, just pass in the name of your new config to the training script:

```bash
python scripts/run_train.py --config-name vanilla_nerf
```

And now you have a brand-new NeRF in training! For testing and visualizing, simply refer to steps 4-5 in the [quickstart guide](https://github.com/plenoptix/pyrad#quickstart).

To help you get started, we also provide additional training tools such as profiling, logging, and debugging. Please refer to our [features guide](../tooling/index.rst).
