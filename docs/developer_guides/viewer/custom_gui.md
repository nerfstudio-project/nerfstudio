# Custom GUI

We provide support for custom viewer GUI elements that can be defined in any `nn.Module`. Although we don't have any specific use cases in mind, here are some examples of what can be achieved with this feature:

- Using text input to modify the rendering
- Logging numerical values to the viewer
- Using checkboxes to turn off and on losses
- Using a dropdown to switch between appearances

## Adding an Element

To define a custom element, create an instance of one of the provided classes in `nerfstudio.viewer.viewer_elements`, and assign it as a class variable in your `nn.Module`.

```python
from nerfstudio.viewer.viewer_elements import ViewerNumber

class MyClass(nn.Module):#must inherit from nn.Module
    def __init__(self):
        # Must be a class variable
        self.custom_value = ViewerNumber(name="My Value", default_value=1.0)
```
**Element Hierarchy**
The viewer recursively searches all `nn.Module` children of the base `Pipeline` object, and arranges parameters into folders based on their variable names.
For example, a `ViewerElement` defined in `pipeline.model.field` will be in the "Custom/model/field" folder in the GUI.

**Reading the value**
To read the value of a custom element, simply access its `value` attribute. In this case it will be `1.0` unless modified by the user in the viewer.

```python
current_value = self.custom_value.value
```

**Callbacks**
You can register a callback that will be called whenever a new value for your GUI element is available. For example, one can use a callback to update config parameters when elements are changed:
```python
def on_change_callback(handle: ViewerCheckbox) -> None:
    self.config.example_parameter = handle.value

self.custom_checkbox = ViewerCheckbox(
    name="Checkbox",
    default_value=False,
    cb_hook=on_change_callback,
)
```

**Thread safety**
Note that `ViewerElement` values can change asynchronously to model execution. So, it's best practice to store the value of a viewer element once at the beginning
of a forward pass and refer to the static variable afterwards.
```python
class MyModel(Model):
    def __init__(self):
        self.slider = ViewerSlider(name="Slider", default_value=0.5, min_value=0.0, max_value=1.0)

    def get_outputs(self,ray)
        slider_val = self.slider.value
        #self.slider.value could change after this, unsafe to use

```


**Writing to the element**
You can write to a viewer element in Python, which provides a convenient way to track values in your code without the need for comet/wandb/tensorboard or relying on `print` statements.

```python
self.custom_value.value = x
```

:::{admonition} Warning
:class: warning

Updating module state while training can have unexpected side effects. It is up to the user to ensure that GUI actions are safe. Conditioning on `self.training` can help determine whether effects are applied during forward passes for training or rendering.
:::

## Example Elements

```{image} imgs/custom_controls.png
:align: center
:width: 400
```

This was created with the following elements:

```python
from nerfstudio.viewer.viewer_elements import *

class MyModel(Model):
    def __init__(self):
        self.a = ViewerButton(name="My Button", cb_hook=self.handle_btn)
        self.b = ViewerNumber(name="Number", default_value=1.0)
        self.c = ViewerCheckbox(name="Checkbox", default_value=False)
        self.d = ViewerDropdown(name="Dropdown", default_value="A", options=["A", "B"])
        self.e = ViewerSlider(name="Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        self.f = ViewerText(name="Text", default_value="Hello World")
        self.g = ViewerVec3(name="3D Vector", default_value=(0.1, 0.7, 0.1))

        self.rgb_renderer = RGBRenderer()
...
class RGBRenderer(nn.Module):
    def __init__(self):
        #lives in "Custom/model/rgb_renderer" GUI folder
        self.a = ViewerRGB(name="F", default_value=(0.1, 0.7, 0.1))
...
```

For more information on the available classes and their arguments, refer to the [API documentation](../../reference/api/viewer.rst)
