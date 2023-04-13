# Custom GUI

We provide support for custom viewer GUI elements that can be defined in any `nn.Module`. Although we don't have any specific use cases in mind, here are some examples of what can be achieved with this feature:

- Using text input to modify the rendering
- Logging numerical values to the viewer
- Using checkboxes to turn off and on losses
- Using a dropdown to switch between appearances

## Adding an Element

To define a custom element, create an instance of one of the provided classes in `nerfstudio.viewer.server.viewer_elements`, and assign it as a class variable in your `nn.Module`.

```python
from nerfstudio.viewer.server.viewer_elements import ViewerNumber

class MyClass(nn.Module):
    def __init__():
        # Must be a class variable
        self.custom_value = ViewerNumber(name="My Value", default_value=1.0)
```

To read the value of a custom element, simply access its `value` attribute. In this case it will be `1.0` unless modified by the user in the viewer.

```python
current_value = self.custom_value.value
```

:::{admonition} Warning
:class: warning

Note that updating module state while training can have unexpected side effects. It is up to the user to ensure that GUI actions are safe. In most cases, we recommend wrapping GUI related logic in `if self.training:` to ensure that the effects are only applied during inference.
:::

## Example Elements

```{image} imgs/custom_controls.png
:align: center
:width: 400
```

This was created with the following elements, refer to the [API](../../reference/api/viewer.rst) for more details:

```python
from nerfstudio.viewer.server.viewer_elements import *

class MyModel(Model):
    def __init__():
        self.a = ViewerButton(name="My Button", call_fn=self.handle_btn)
        self.b = ViewerNumber(name="Number", default_value=1.0)
        self.c = ViewerCheckbox(name="Checkbox", default_value=False)
        self.d = ViewerDropdown(name="Dropdown", default_value="A", options=["A", "B"])
        self.e = ViewerSlider(name="Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        self.f = ViewerText(name="Text", default_value="Hello World")
        self.g = ViewerVec3(name="3D Vector", default_value=(0.1, 0.7, 0.1))

        self.rgb_renderer = RGBRenderer()
...
class RGBRenderer(nn.Module):
    def __init__():
        self.a = ViewerRGB(name="F", default_value=(0.1, 0.7, 0.1))
...
```
