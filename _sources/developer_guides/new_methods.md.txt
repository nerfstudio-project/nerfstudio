# Adding a New Method

Nerfstudio aims to offer researchers a codebase that they can utilize to extend and develop novel methods. Our vision is for users to establish a distinct repository that imports nerfstudio and overrides pipeline components to cater to specific functionality requirements of the new approach. If any of the new features require modifications to the core nerfstudio repository and can be generally useful, we encourage you to submit a PR to enable others to benefit from it.

You can use the [nerfstudio-method-template](https://github.com/nerfstudio-project/nerfstudio-method-template) repository as a minimal guide to register your new methods. Examples are often the best way to learn, take a look at the [LERF](https://github.com/kerrj/lerf) repository for a good example of how to extend and use nerfstudio in your projects.

## File Structure

We recommend the following file structure:

```
├── my_method
│   ├── __init__.py
│   ├── my_config.py
│   ├── custom_pipeline.py [optional]
│   ├── custom_model.py [optional]
│   ├── custom_field.py [optional]
│   ├── custom_datamanger.py [optional]
│   ├── custom_dataparser.py [optional]
│   ├── ...
├── pyproject.toml
```

## Registering custom model with nerfstudio

In order to extend the Nerfstudio and register your own methods, you can package your code as a python package
and register it with Nerfstudio as a `nerfstudio.method_configs` entrypoint in the `pyproject.toml` file.
Nerfstudio will automatically look for all registered methods and will register them to be used
by methods such as `ns-train`.

First create a config file:

```python
"""my_method/my_config.py"""

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

MyMethod = MethodSpecification(
  config=TrainerConfig(
    method_name="my-method",
    pipeline=...
    ...
  ),
  description="Custom description"
)
```

Then create a `pyproject.toml` file. This is where the entrypoint to your method is set and also where you can specify additional dependencies required by your codebase.

```python
"""pyproject.toml"""

[project]
name = "my_method"

dependencies = [
    "nerfstudio" # you may want to consider pinning the version, ie "nerfstudio==0.1.19"
]

[tool.setuptools.packages.find]
include = ["my_method*"]

[project.entry-points.'nerfstudio.method_configs']
my-method = 'my_method.my_config:MyMethod'
```

finally run the following to register the method,

```
pip install -e .
```

When developing a new method you don't always want to install your code as a package.
Instead, you may use the `NERFSTUDIO_METHOD_CONFIGS` environment variable to temporarily register your custom method.

```
export NERFSTUDIO_METHOD_CONFIGS="my-method=my_method.my_config:MyMethod"
```

The `NERFSTUDIO_METHOD_CONFIGS` environment variable additionally accepts a function or derived class to temporarily register your custom method.

```python
"""my_method/my_config.py"""

from dataclasses import dataclass, field
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

def MyMethodFunc():
    return MethodSpecification(
      config=TrainerConfig(...)
      description="Custom description"
    )

@dataclass
class MyMethodClass(MethodSpecification):
    config: TrainerConfig = field(default_factory=lambda: TrainerConfig(...))
    description: str = "Custom description"
```

## Registering custom dataparser with nerfstudio

We also support adding new dataparsers in a similar way. In order to extend the NeRFstudio and register a customized dataparser, you can register it with Nerfstudio as a `nerfstudio.dataparser_configs` entrypoint in the `pyproject.toml` file. Nerfstudio will automatically look for all registered dataparsers and will register them to be used by methods such as `ns-train`.

You can declare the dataparser in the same config file:

```python
"""my_method/my_config.py"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from my_method.custom_dataparser import CustomDataparserConfig

MyDataparser = DataParserSpecification(config=CustomDataparserConfig())
```

Then add the following lines in the `pyproject.toml` file, where the entrypoint to the new dataparser is set.

```python
"""pyproject.toml"""

[project]
name = "my_method"

[project.entry-points.'nerfstudio.dataparser_configs']
custom-dataparser = 'my_method.my_config:MyDataparser'
```

finally run the following to register the dataparser.

```
pip install -e .
```

Similarly to the method development, you can also use environment variables to register dataparsers.
Use the `NERFSTUDIO_DATAPARSER_CONFIGS` environment variable:

```
export NERFSTUDIO_DATAPARSER_CONFIGS="my-dataparser=my_package.my_config:MyDataParser"
```

Same as with custom methods, `NERFSTUDIO_DATAPARSER_CONFIGS` environment variable additionally accepts a function or derived class to temporarily register your custom method.

## Running custom method

After registering your method you should be able to run the method with,

```
ns-train my-method --data DATA_DIR
```

## Adding to the _nerf.studio_ documentation

We invite researchers to contribute their own methods to our online documentation. You can find more information on how to do this {ref}`here<own_method_docs>`.
