# Python Viewer Control

Similar to [`ViewerElements`](./custom_gui.md), Nerfstudio includes supports a Python interface to the viewer through which you can:

* Set viewer camera pose and FOV
* Set viewer scene crop
* Retrieve the current viewer camera matrix
* Install listeners for click events inside the viewer window

## Usage

First, instantiate a `ViewerControl` object as a class variable inside a model file.
Just like `ViewerElements`, you can create an instance inside any class which inherits from `nn.Module`
and is contained within the `Pipeline` object (for example the `Model`)

```python
from nerfstudio.viewer.server.viewer_elements import ViewerControl

class MyModel(nn.Module):  # Must inherit from nn.Module
    def __init__(self):
        # Must be a class variable
        self.viewer_control = ViewerControl()  # no arguments
```
## Get Camera Matrix
To get the current camera intrinsics and extrinsics, use the `get_camera` function. This returns a `nerfstudio.cameras.cameras.Cameras` object. This object can be used to generate `RayBundles`, retrieve intrinsics and extrinsics, and more.

```python
from nerfstudio.viewer.server.viewer_elements import ViewerControl, ViewerButton

class MyModel(nn.Module):  # Must inherit from nn.Module
    def __init__(self):
        ...
        def button_cb(button):
            # example of using the get_camera function, pass img width and height
            # returns a Cameras object with 1 camera
            camera = self.viewer_control.get_camera(100,100)
            if camera is None:
                # returns None when the viewer is not connected yet
                return
            # get the camera pose
            camera_extrinsics_matrix = camera.camera_to_worlds[0,...]  # 3x4 matrix
            # generate image RayBundle
            bundle = camera.generate_rays(camera_indices=0)
            # Compute depth, move camera, or whatever you want
            ...
        self.viewer_button = ViewerButton(name="Dummy Button",cb_hook=button_cb)
```

## Set Camera Properties
You can set the viewer camera position and FOV from python. 
To set position, you must define a new camera position as well as a 3D "look at" point which the camera aims towards.
```python
from nerfstudio.viewer.server.viewer_elements import ViewerControl,ViewerButton

class MyModel(nn.Module):  # Must inherit from nn.Module
    def __init__(self):
        ...
        def aim_at_origin(button):
            # instant=False means the camera smoothly animates
            # instant=True means the camera jumps instantly to the pose
            self.viewer_control.set_pose(position=(1,1,1),look_at=(0,0,0),instant=False)
        self.viewer_button = ViewerButton(name="Dummy Button",cb_hook=button_cb)
```

## Scene Click Callbacks
We forward *single* clicks inside the viewer to the ViewerControl object, which you can use to interact with the scene. To do this, register a callback using `register_click_cb()`. The click is defined to be a ray that starts at the camera origin and passes through the click point on the screen, in world coordinates. 

```python
from nerfstudio.viewer.server.viewer_elements import ViewerControl,ViewerClick

class MyModel(nn.Module):  # must inherit from nn.Module
    def __init__(self):
        # Must be a class variable
        self.viewer_control = ViewerControl()  # no arguments
        def click_cb(click: ViewerClick):
            print(f"Click at {click.origin} in direction {click.direction}")
        self.viewer_control.register_click_cb(click_cb)
```

You can also use `unregister_click_cb()` to remove callbacks that are no longer needed. A good example is a "Click on Scene" button, that when pressed, would register a callback that would wait for the next click, and then unregister itself.
```python
    ...
    def button_cb(button: ViewerButton):
        def click_cb(click: ViewerClick):
            print(f"Click at {click.origin} in direction {click.direction}")
            self.viewer_control.unregister_click_cb(click_cb)
        self.viewer_control.register_click_cb(click_cb)
```

### Thread safety
Just like `ViewerElement` callbacks, click callbacks are asynchronous to training and can potentially interrupt a call to `get_outputs()`.

