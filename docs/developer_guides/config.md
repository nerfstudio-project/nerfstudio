# Customizable configs

Our dataclass configs allow you to easily plug in different permutations of models, dataloaders, modules, etc.
and modify all parameters from a typed CLI supported by [tyro](https://pypi.org/project/tyro/).

### Base components

All basic, reusable config components can be found in `nerfstudio/configs/base_config.py`. The `Config` class at the bottom of the file is the upper-most config level and stores all of the sub-configs needed to get started with training.

You can browse this file and read the attribute annotations to see what configs are available and what each specifies.

### Creating new configs

If you are interested in creating a brand new model or data format, you will need to create a corresponding config with associated parameters you want to expose as configurable.

Let's say you want to create a new model called Nerfacto. You can create a new `Model` class that extends the base class as described [here](pipelines/models.ipynb). Before the model definition, you define the actual `NerfactoModelConfig` which points to the `NerfactoModel` class (make sure to wrap the `_target` classes in a `field` as shown below).

:::{admonition} Tip
:class: info

You can then enable type/auto complete on the config passed into the `NerfactoModel` by specifying the config type below the class definition.
:::

```python
"""nerfstudio/models/nerfacto.py"""

@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    ...

class NerfactoModel(Model):
     """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig
    ...
```

The same logic applies to all other custom configs you want to create. For more examples, you can see `nerfstudio/data/dataparsers/nerfstudio_dataparsers.py`, `nerfstudio/data/datamanagers.py`.

:::{admonition} See Also
:class: seealso

For how to create the actual data and model classes that follow the configs, please refer to [pipeline overview](pipelines/index.rst).
:::

### Updating method configs

If you are interested in creating a new model config, you will have to modify the `nerfstudio/configs/method_configs.py` file. This is where all of the configs for implemented models are housed. You can browse this file to see how we construct various existing models by modifying the `Config` class and specifying new or modified default components.

For instance, say we created a brand new model called Nerfacto that has an associated `NerfactoModelConfig`, we can specify the following new Config by overriding the pipeline and optimizers attributes appropriately.

```python
"""nerfstudio/configs/method_configs.py"""

method_configs["nerfacto"] = Config(
    method_name="nerfacto",
    pipeline=VanillaPipelineConfig(
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 14),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
)
```

After placing your new `Config` class into the `method_configs` dictionary, you can provide a description for the model by updating the `descriptions` dictionary at the top of the file.

### Modifying from CLI

Often times, you just want to play with the parameters of an existing model without having to specify a new one. You can easily do so via CLI. Below, we showcase some useful CLI commands.

- List out all existing models

  ```bash
  ns-train --help
  ```

- List out all exist configurable parameters for `{METHOD_NAME}`

  ```bash
  ns-train {METHOD_NAME} --help
  ```

- Change the train/eval dataset

  ```bash
  ns-train {METHOD_NAME} --data DATA_PATH
  ```

- Enable the viewer

  ```bash
  ns-train {METHOD_NAME} --vis viewer
  ```

- See what options are available for the specified dataparser (e.g. blender-data)

  ```bash
  ns-train {METHOD_NAME} {DATA_PARSER} --help
  ```

- Run with changed dataparser attributes and viewer on
  ```bash
  # NOTE: the dataparser and associated configurations go at the end of the command
  ns-train {METHOD_NAME} --vis viewer {DATA_PARSER} --scale-factor 0.5
  ```
