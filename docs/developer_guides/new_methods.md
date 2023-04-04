# Adding a New Method

Nerfstudio aims to offer researchers a codebase that they can utilize to extend and develop novel methods. Our vision is for users to establish a distinct repository that imports nerfstudio and overrides pipeline components to cater to specific functionality requirements of the new approach. If any of the new features require modifications to the core nerfstudio repository and can be generally useful, we encourage you to submit a PR to enable others to benefit from it.

Examples are often the best way to learn, take a look at the [LERF](https://github.com/kerrj/lerf) repository for good example of how to use nerfstudio in your projects.

## File Structure

We recommend the following file structure:

```
├── my_method
│   ├── my_config.py
│   ├── custom_pipeline.py [optional]
│   ├── custom_model.py [optional]
│   ├── custom_field.py [optional]
│   ├── custom_datamanger.py [optional]
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

[project.entry-points.'nerfstudio.method_configs']
my-method = 'my_method.my_config:MyMethod'
```

finally run the following to register the method,

```
pip install -e .
```

## Running custom method

After registering your method you should be able to run the method with,

```
ns-train my-method --data DATA_DIR
```
