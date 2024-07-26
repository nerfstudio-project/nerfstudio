# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Viewer GUI elements for the nerfstudio viewer"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch
import viser.transforms as vtf
from typing_extensions import LiteralString, TypeVar
from viser import (
    GuiButtonGroupHandle,
    GuiButtonHandle,
    GuiDropdownHandle,
    GuiInputHandle,
    ScenePointerEvent,
    ViserServer,
)

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.utils import CameraState, get_camera

if TYPE_CHECKING:
    from nerfstudio.viewer.viewer import Viewer

TValue = TypeVar("TValue")
TString = TypeVar("TString", default=str, bound=str)


@dataclass
class ViewerClick:
    """
    Class representing a click in the viewer as a ray.
    """

    # the information here matches the information in the ClickMessage,
    # but we implement a wrapper as an abstraction layer
    origin: Tuple[float, float, float]
    """The origin of the click in world coordinates (center of camera)"""
    direction: Tuple[float, float, float]
    """
    The direction of the click if projected from the camera through the clicked pixel,
    in world coordinates
    """
    screen_pos: Tuple[float, float]
    """The screen position of the click in OpenCV screen coordinates, normalized to [0, 1]"""


@dataclass
class ViewerRectSelect:
    """
    Class representing a rectangle selection in the viewer (screen-space).

    The screen coordinates follow OpenCV image coordinates, with the origin at the top-left corner,
    but the bounds are also normalized to [0, 1] in both dimensions.
    """

    min_bounds: Tuple[float, float]
    """The minimum bounds of the rectangle selection in screen coordinates."""
    max_bounds: Tuple[float, float]
    """The maximum bounds of the rectangle selection in screen coordinates."""


class ViewerControl:
    """
    class for exposing non-gui controls of the viewer to the user
    """

    def _setup(self, viewer: Viewer):
        """
        Internal use only, setup the viewer control with the viewer state object

        Args:
            viewer: The viewer object (viewer.py)
        """
        self.viewer: Viewer = viewer
        self.viser_server: ViserServer = viewer.viser_server

    def set_pose(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        look_at: Optional[Tuple[float, float, float]] = None,
        instant: bool = False,
    ):
        """
        Set the camera position of the viewer camera.

        Args:
            position: The new position of the camera in world coordinates
            look_at: The new look_at point of the camera in world coordinates
            instant: If the camera should move instantly or animate to the new position
        """
        raise NotImplementedError()

    def set_fov(self, fov):
        """
        Set the FOV of the viewer camera

        Args:
            fov: The new FOV of the camera in degrees

        """
        raise NotImplementedError()

    def set_crop(self, min_point: Tuple[float, float, float], max_point: Tuple[float, float, float]):
        """
        Set the scene crop box of the viewer to the specified min,max point

        Args:
            min_point: The minimum point of the crop box
            max_point: The maximum point of the crop box

        """
        raise NotImplementedError()

    def get_camera(self, img_height: int, img_width: int, client_id: Optional[int] = None) -> Optional[Cameras]:
        """
        Returns the Cameras object representing the current camera for the viewer, or None if the viewer
        is not connected yet

        Args:
            img_height: The height of the image to get camera intrinsics for
            img_width: The width of the image to get camera intrinsics for
        """
        clients = self.viser_server.get_clients()
        if len(clients) == 0:
            return None
        if not client_id:
            client_id = list(clients.keys())[0]

        from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

        client = clients[client_id]
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(client.camera.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        camera_state = CameraState(
            fov=client.camera.fov, aspect=client.camera.aspect, c2w=c2w, camera_type=CameraType.PERSPECTIVE
        )
        return get_camera(camera_state, img_height, img_width)

    def register_click_cb(self, cb: Callable):
        """Deprecated, use register_pointer_cb instead."""
        CONSOLE.log("`register_click_cb` is deprecated, use `register_pointer_cb` instead.")
        self.register_pointer_cb("click", cb)

    @overload
    def register_pointer_cb(
        self,
        event_type: Literal["click"],
        cb: Callable[[ViewerClick], None],
        removed_cb: Optional[Callable[[], None]] = None,
    ): ...

    @overload
    def register_pointer_cb(
        self,
        event_type: Literal["rect-select"],
        cb: Callable[[ViewerRectSelect], None],
        removed_cb: Optional[Callable[[], None]] = None,
    ): ...

    def register_pointer_cb(
        self,
        event_type: Literal["click", "rect-select"],
        cb: Callable[[ViewerClick], None] | Callable[[ViewerRectSelect], None],
        removed_cb: Optional[Callable[[], None]] = None,
    ):
        """
        Add a callback which will be called when a scene pointer event is detected in the viewer.
        Scene pointer events include:
        - "click": A click event, which includes the origin and direction of the click
        - "rect-select": A rectangle selection event, which includes the screen bounds of the box selection

        The callback should take a ViewerClick object as an argument if the event type is "click",
        and a ViewerRectSelect object as an argument if the event type is "rect-select".

        Args:
            cb: The callback to call when a click or a rect-select is detected.
            removed_cb: The callback to run when the pointer event is removed.
        """
        from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

        def wrapped_cb(scene_pointer_msg: ScenePointerEvent):
            # Check that the event type is the same as the one we are interested in.
            if scene_pointer_msg.event_type != event_type:
                raise ValueError(f"Expected event type {event_type}, got {scene_pointer_msg.event_type}")

            if scene_pointer_msg.event_type == "click":
                origin = scene_pointer_msg.ray_origin
                direction = scene_pointer_msg.ray_direction
                screen_pos = scene_pointer_msg.screen_pos[0]
                assert (origin is not None) and (
                    direction is not None
                ), "Origin and direction should not be None for click event."
                origin = tuple([x / VISER_NERFSTUDIO_SCALE_RATIO for x in origin])
                assert len(origin) == 3
                pointer_event = ViewerClick(origin, direction, screen_pos)
            elif scene_pointer_msg.event_type == "rect-select":
                pointer_event = ViewerRectSelect(scene_pointer_msg.screen_pos[0], scene_pointer_msg.screen_pos[1])
            else:
                raise ValueError(f"Unknown event type: {scene_pointer_msg.event_type}")

            cb(pointer_event)  # type: ignore

        cb_overriden = False
        with warnings.catch_warnings(record=True) as w:
            # Register the callback with the viser server.
            self.viser_server.scene.on_pointer_event(event_type=event_type)(wrapped_cb)
            # If there exists a warning, it's because a callback was overriden.
            cb_overriden = len(w) > 0

        if cb_overriden:
            warnings.warn(
                "A ScenePointer callback has already been registered for this event type. "
                "The new callback will override the existing one."
            )

        # If there exists a cleanup callback after the pointer event is done, register it.
        if removed_cb is not None:
            self.viser_server.scene.on_pointer_callback_removed(removed_cb)

    def unregister_click_cb(self, cb: Optional[Callable] = None):
        """Deprecated, use unregister_pointer_cb instead. `cb` is ignored."""
        warnings.warn("`unregister_click_cb` is deprecated, use `unregister_pointer_cb` instead.")
        if cb is not None:
            # raise warning
            warnings.warn("cb argument is ignored in unregister_click_cb.")

        self.unregister_pointer_cb()

    def unregister_pointer_cb(self):
        """
        Remove a callback which will be called, when a scene pointer event is detected in the viewer.

        Args:
            cb: The callback to remove
        """
        self.viser_server.scene.remove_pointer_callback()

    @property
    def server(self):
        return self.viser_server


class ViewerElement(Generic[TValue]):
    """Base class for all viewer elements

    Args:
        name: The name of the element
        disabled: If the element is disabled
        visible: If the element is visible
    """

    def __init__(
        self,
        name: str,
        disabled: bool = False,
        visible: bool = True,
        cb_hook: Callable = lambda element: None,
    ) -> None:
        self.name = name
        self.gui_handle: Optional[Union[GuiInputHandle[TValue], GuiButtonHandle, GuiButtonGroupHandle]] = None
        self.disabled = disabled
        self.visible = visible
        self.cb_hook = cb_hook

    @abstractmethod
    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        """
        Returns the GuiInputHandle object which actually controls the parameter in the gui.

        Args:
            viser_server: The server to install the gui element into.
        """
        ...

    def remove(self) -> None:
        """Removes the gui element from the viewer"""
        if self.gui_handle is not None:
            self.gui_handle.remove()
            self.gui_handle = None

    def set_hidden(self, hidden: bool) -> None:
        """Sets the hidden state of the gui element"""
        assert self.gui_handle is not None
        self.gui_handle.visible = not hidden

    def set_disabled(self, disabled: bool) -> None:
        """Sets the disabled state of the gui element"""
        assert self.gui_handle is not None
        self.gui_handle.disabled = disabled

    def set_visible(self, visible: bool) -> None:
        """Sets the visible state of the gui element"""
        assert self.gui_handle is not None
        self.gui_handle.visible = visible

    @abstractmethod
    def install(self, viser_server: ViserServer) -> None:
        """Installs the gui element into the given viser_server"""
        ...


class ViewerButton(ViewerElement[bool]):
    """A button in the viewer

    Args:
        name: The name of the button
        cb_hook: The function to call when the button is pressed
        disabled: If the button is disabled
        visible: If the button is visible
    """

    gui_handle: GuiButtonHandle

    def __init__(self, name: str, cb_hook: Callable[[ViewerButton], Any], disabled: bool = False, visible: bool = True):
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=cb_hook)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.gui.add_button(label=self.name, disabled=self.disabled, visible=self.visible)

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        assert self.gui_handle is not None
        self.gui_handle.on_click(lambda _: self.cb_hook(self))


class ViewerParameter(ViewerElement[TValue], Generic[TValue]):
    """A viewer element with state

    Args:
        name: The name of the element
        default_value: The default value of the element
        disabled: If the element is disabled
        visible: If the element is visible
        cb_hook: Callback to call on update
    """

    gui_handle: GuiInputHandle

    def __init__(
        self,
        name: str,
        default_value: TValue,
        disabled: bool = False,
        visible: bool = True,
        cb_hook: Callable = lambda element: None,
    ) -> None:
        super().__init__(name, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.default_value = default_value

    def install(self, viser_server: ViserServer) -> None:
        """
        Based on the type provided by default_value, installs a gui element inside the given viser_server

        Args:
            viser_server: The server to install the gui element into.
        """
        self._create_gui_handle(viser_server)

        assert self.gui_handle is not None
        self.gui_handle.on_update(lambda _: self.cb_hook(self))

    @abstractmethod
    def _create_gui_handle(self, viser_server: ViserServer) -> None: ...

    @property
    def value(self) -> TValue:
        """Returns the current value of the viewer element"""
        if self.gui_handle is None:
            return self.default_value
        return self.gui_handle.value

    @value.setter
    def value(self, value: TValue) -> None:
        if self.gui_handle is not None:
            self.gui_handle.value = value
        else:
            self.default_value = value


IntOrFloat = TypeVar("IntOrFloat", int, float)


class ViewerSlider(ViewerParameter[IntOrFloat], Generic[IntOrFloat]):
    """A slider in the viewer

    Args:
        name: The name of the slider
        default_value: The default value of the slider
        min_value: The minimum value of the slider
        max_value: The maximum value of the slider
        step: The step size of the slider
        disabled: If the slider is disabled
        visible: If the slider is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name: str,
        default_value: IntOrFloat,
        min_value: IntOrFloat,
        max_value: IntOrFloat,
        step: IntOrFloat = 0.1,
        disabled: bool = False,
        visible: bool = True,
        cb_hook: Callable[[ViewerSlider], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert isinstance(default_value, (float, int))
        super().__init__(name, default_value, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.min = min_value
        self.max = max_value
        self.step = step
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.gui.add_slider(
            self.name,
            self.min,
            self.max,
            self.step,
            self.default_value,
            disabled=self.disabled,
            visible=self.visible,
            hint=self.hint,
        )


class ViewerText(ViewerParameter[str]):
    """A text field in the viewer

    Args:
        name: The name of the text field
        default_value: The default value of the text field
        disabled: If the text field is disabled
        visible: If the text field is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name: str,
        default_value: str,
        disabled: bool = False,
        visible: bool = True,
        cb_hook: Callable[[ViewerText], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert isinstance(default_value, str)
        super().__init__(name, default_value, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.gui.add_text(
            self.name, self.default_value, disabled=self.disabled, visible=self.visible, hint=self.hint
        )


class ViewerNumber(ViewerParameter[IntOrFloat], Generic[IntOrFloat]):
    """A number field in the viewer

    Args:
        name: The name of the number field
        default_value: The default value of the number field
        disabled: If the number field is disabled
        visible: If the number field is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    default_value: IntOrFloat

    def __init__(
        self,
        name: str,
        default_value: IntOrFloat,
        disabled: bool = False,
        visible: bool = True,
        cb_hook: Callable[[ViewerNumber], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert isinstance(default_value, (float, int))
        super().__init__(name, default_value, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.gui.add_number(
            self.name, self.default_value, disabled=self.disabled, visible=self.visible, hint=self.hint
        )


class ViewerCheckbox(ViewerParameter[bool]):
    """A checkbox in the viewer

    Args:
        name: The name of the checkbox
        default_value: The default value of the checkbox
        disabled: If the checkbox is disabled
        visible: If the checkbox is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name: str,
        default_value: bool,
        disabled: bool = False,
        visible: bool = True,
        cb_hook: Callable[[ViewerCheckbox], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert isinstance(default_value, bool)
        super().__init__(name, default_value, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.gui.add_checkbox(
            self.name, self.default_value, disabled=self.disabled, visible=self.visible, hint=self.hint
        )


TLiteralString = TypeVar("TLiteralString", bound=LiteralString)


class ViewerDropdown(ViewerParameter[TString], Generic[TString]):
    """A dropdown in the viewer

    Args:
        name: The name of the dropdown
        default_value: The default value of the dropdown
        options: The options of the dropdown
        disabled: If the dropdown is disabled
        visible: If the dropdown is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    gui_handle: Optional[GuiDropdownHandle[TString]]

    def __init__(
        self,
        name: str,
        default_value: TString,
        options: List[TString],
        disabled: bool = False,
        visible: bool = True,
        cb_hook: Callable[[ViewerDropdown], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert default_value in options
        super().__init__(name, default_value, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.options = options
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.gui.add_dropdown(
            self.name,
            self.options,
            self.default_value,
            disabled=self.disabled,
            visible=self.visible,
            hint=self.hint,  # type: ignore
        )

    def set_options(self, new_options: List[TString]) -> None:
        """
        Sets the options of the dropdown,

        Args:
            new_options: The new options. If the current option isn't in the new options, the first option is selected.
        """
        self.options = new_options
        if self.gui_handle is not None:
            self.gui_handle.options = new_options


class ViewerButtonGroup(ViewerParameter[TString], Generic[TString]):
    """A button group in the viewer. Unlike other fields, cannot be disabled.

    Args:
        name: The name of the button group
        visible: If the button group is visible
        options: The options of the button group
        cb_hook: Callback to call on update
    """

    gui_handle: GuiButtonGroupHandle

    def __init__(
        self,
        name: str,
        default_value: TString,
        options: List[TString],
        visible: bool = True,
        cb_hook: Callable[[ViewerDropdown], Any] = lambda element: None,
    ):
        super().__init__(name, disabled=False, visible=visible, default_value=default_value, cb_hook=cb_hook)
        self.options = options

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.gui.add_button_group(self.name, self.options, visible=self.visible)

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        assert self.gui_handle is not None
        self.gui_handle.on_click(lambda _: self.cb_hook(self))


class ViewerRGB(ViewerParameter[Tuple[int, int, int]]):
    """
    An RGB color picker for the viewer

    Args:
        name: The name of the color picker
        default_value: The default value of the color picker
        disabled: If the color picker is disabled
        visible: If the color picker is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name,
        default_value: Tuple[int, int, int],
        disabled=False,
        visible=True,
        cb_hook: Callable[[ViewerRGB], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert len(default_value) == 3
        super().__init__(name, default_value, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.gui.add_rgb(
            self.name, self.default_value, disabled=self.disabled, visible=self.visible, hint=self.hint
        )


class ViewerVec3(ViewerParameter[Tuple[float, float, float]]):
    """
    3 number boxes in a row to input a vector

    Args:
        name: The name of the vector
        default_value: The default value of the vector
        step: The step of the vector
        disabled: If the vector is disabled
        visible: If the vector is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name,
        default_value: Tuple[float, float, float],
        step=0.1,
        disabled=False,
        visible=True,
        cb_hook: Callable[[ViewerVec3], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert len(default_value) == 3
        super().__init__(name, default_value, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.step = step
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.gui.add_vector3(
            self.name, self.default_value, step=self.step, disabled=self.disabled, visible=self.visible, hint=self.hint
        )
