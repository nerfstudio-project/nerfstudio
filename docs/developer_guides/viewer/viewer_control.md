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
from nerfstudio.viewer.viewer_elements import ViewerControl

class MyModel(nn.Module):  # Must inherit from nn.Module
    def __init__(self):
        # Must be a class variable
        self.viewer_control = ViewerControl()  # no arguments
```
## Get Camera Matrix
To get the current camera intrinsics and extrinsics, use the `get_camera` function. This returns a `nerfstudio.cameras.cameras.Cameras` object. This object can be used to generate `RayBundles`, retrieve intrinsics and extrinsics, and more.

```python
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerButton

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
from nerfstudio.viewer.viewer_elements import ViewerControl,ViewerButton

class MyModel(nn.Module):  # Must inherit from nn.Module
    def __init__(self):
        ...
        def aim_at_origin(button):
            # instant=False means the camera smoothly animates
            # instant=True means the camera jumps instantly to the pose
            self.viewer_control.set_pose(position=(1,1,1),look_at=(0,0,0),instant=False)
        self.viewer_button = ViewerButton(name="Dummy Button",cb_hook=button_cb)
```

## Scene Pointer Callbacks
We forward user interactions with the viewer to the `ViewerControl` object, which you can use to interact with the scene. 

We currently support:
 - `ViewerClick`: *single* clicks inside the viewer. The click is defined to be a ray that starts at the camera origin and passes through the click point on the screen, in world coordinates. 
 - `ViewerRectSelect`: drag to select a rectangle in the viewer screen. The rectangle is defined by two points (top-left and bottom-right corners) in normalized OpenCV screen coordinates.

To do this, register a callback using `register_pointer_cb()`. 

You can also use `unregister_pointer_cb()` to remove callbacks that are no longer needed. A good example is a "Click on Scene" button, that when pressed, would register a callback that would wait for the next click, and then unregister itself.

Note that the viewer can only listen to *one* scene pointer callback at a time. If you register a new callback, the old one will be unregistered! Be warned that if the callback includes GUI state changes (e.g., re-enabling a disabled button), they may be lost. You can ensure that the GUI state is restored by providing a `removed_cb` function that will be called after the callback is removed.

```python
from nerfstudio.viewer.viewer_elements import ViewerControl,ViewerClick

class MyModel(nn.Module):  # must inherit from nn.Module
    def __init__(self):
        # Must be a class variable
        self.viewer_control = ViewerControl()  # no arguments
        
        # Listen to clicks in the viewer...
        def pointer_click_cb(click: ViewerClick):
            print(f"Click at {click.origin} in direction {click.direction}, screen position {click.screen_pos}.")
        self.viewer_control.register_pointer_cb("click", pointer_click_cb)

        # Listen to rectangle selections in the viewer...
        def pointer_rect_cb(rect: ViewerRectSelect):
            print(f"Rectangular selection from {rect.min_bounds} to {rect.max_bounds}.")
        self.viewer_control.register_pointer_cb("click", pointer_rect_cb)

        ... 
        # Or make a button that, once pressed, listens to clicks in the viewer.
        def button_cb(button: ViewerButton):
            def pointer_click_cb(click: ViewerClick):
                ...
                self.viewer_control.unregister_pointer_cb()
            self.viewer_control.register_pointer_cb("click", pointer_click_cb)
        self.viewer_button = ViewerButton(name="Click on Scene", cb_hook=button_cb)

        # Or make a button that, once pressed, listens to clicks in the viewer.
        # Here, the button is disabled while it is listening to clicks.
        # The button will become enabled again if either:
        # - the callback is removed in `pointer_click_cb`, with the `unregister...`, or
        # - the callback is overridden by the viewer (to listen to another callback).
        def button_cb(button: ViewerButton):
            def pointer_click_cb(click: ViewerClick):
                ...
                self.viewer_control.unregister_pointer_cb()

            def pointer_click_removed_cb():
                self.viewer_button.set_disabled(False)

            self.viewer_button.set_disabled(True)
            self.viewer_control.register_pointer_cb(
                "click",
                cb=pointer_click_cb,
                removed_cb=pointer_click_removed_cb
            )
        self.viewer_button = ViewerButton(name="Click on Scene", cb_hook=button_cb)
```

### Thread safety
Just like `ViewerElement` callbacks, click callbacks are asynchronous to training and can potentially interrupt a call to `get_outputs()`.
