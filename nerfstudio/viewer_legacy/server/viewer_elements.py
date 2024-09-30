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

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, Tuple

from typing_extensions import TypeVar

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer_legacy.viser import GuiHandle, GuiSelectHandle, ViserServer
from nerfstudio.viewer_legacy.viser.messages import ClickMessage

if TYPE_CHECKING:
    from nerfstudio.viewer_legacy.server.control_panel import ControlPanel
    from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState


TValue = TypeVar("TValue")


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


class ViewerControl:
    """
    class for exposing non-gui controls of the viewer to the user
    """

    def __init__(self):
        # this should be a user-facing constructor, since it will be used inside the model/pipeline class
        self.click_cbs = []
        from nerfstudio.utils.rich_utils import CONSOLE

        CONSOLE.print(
            "WARNING: viewer_legacy.server.viewer_elements is deprecated and will be removed in the future. Use the new nerfstudio.viewer instead",
            style="bold yellow",
        )

    def _setup(self, viewer_state: ViewerLegacyState):
        """
        Internal use only, setup the viewer control with the viewer state object

        Args:
            viewer_state: The viewer state object (viewer_state.py)
        """
        self.viewer_state = viewer_state
        self.control_panel: ControlPanel = viewer_state.control_panel
        self.viser_server: ViserServer = viewer_state.viser_server

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
        assert hasattr(self, "viser_server"), "Called set_position on uninitialized ViewerControl"
        self.viser_server.set_camera(position=position, look_at=look_at, instant=instant)

    def set_fov(self, fov):
        """
        Set the FOV of the viewer camera

        Args:
            fov: The new FOV of the camera in degrees

        """
        assert hasattr(self, "viser_server"), "Called set_fov on uninitialized ViewerControl"
        self.viser_server.set_camera(fov=fov)

    def set_crop(self, min_point: Tuple[float, float, float], max_point: Tuple[float, float, float]):
        """
        Set the scene crop box of the viewer to the specified min,max point

        Args:
            min_point: The minimum point of the crop box
            max_point: The maximum point of the crop box

        """
        assert hasattr(self, "viser_server"), "Called set_crop on uninitialized ViewerControl"
        self.control_panel.crop_min = min_point
        self.control_panel.crop_max = max_point

    def get_camera(self, img_height: int, img_width: int) -> Optional[Cameras]:
        """
        Returns the Cameras object representing the current camera for the viewer, or None if the viewer
        is not connected yet

        Args:
            img_height: The height of the image to get camera intrinsics for
            img_width: The width of the image to get camera intrinsics for
        """
        return self.viewer_state.get_camera(img_height, img_width)

    def register_click_cb(self, cb: Callable):
        """
        Add a callback which will be called when a click is detected in the viewer.

        Args:
            cb: The callback to call when a click is detected.
                The callback should take a ViewerClick object as an argument
        """
        self.click_cbs.append(cb)

    def on_click(self, msg: ClickMessage):
        """
        Internal use only, register a click in the viewer which propagates to all self.click_cbs
        """
        click = ViewerClick(origin=msg.origin, direction=msg.direction)
        for c in self.click_cbs:
            c(click)


class ViewerElement(Generic[TValue]):
    """Base class for all viewer elements

    Args:
        name: The name of the element
        disabled: If the element is disabled
    """

    def __init__(
        self,
        name: str,
        disabled: bool = False,
        cb_hook: Callable = lambda element: None,
    ) -> None:
        self.name = name
        self.gui_handle: Optional[GuiHandle[TValue]] = None
        self.disabled = disabled
        self.cb_hook = cb_hook
        from nerfstudio.utils.rich_utils import CONSOLE

        CONSOLE.print(
            "WARNING: viewer_legacy.server.viewer_elements is deprecated and will be removed in the future. Use the new nerfstudio.viewer instead",
            style="bold yellow",
        )

    @abstractmethod
    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        """
        Returns the GuiHandle object which actually controls the parameter in the gui.

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
        self.gui_handle.set_hidden(hidden)

    def set_disabled(self, disabled: bool) -> None:
        """Sets the disabled state of the gui element"""
        assert self.gui_handle is not None
        self.gui_handle.set_disabled(disabled)

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
    """

    def __init__(self, name: str, cb_hook: Callable[[ViewerButton], Any], disabled: bool = False):
        super().__init__(name, disabled=disabled, cb_hook=cb_hook)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.add_gui_button(self.name)
        self.gui_handle.set_disabled(self.disabled)

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        assert self.gui_handle is not None
        self.gui_handle.on_update(lambda _: self.cb_hook(self))


class ViewerParameter(ViewerElement[TValue], Generic[TValue]):
    """A viewer element with state

    Args:
        name: The name of the element
        default_value: The default value of the element
        disabled: If the element is disabled
        cb_hook: Callback to call on update
    """

    def __init__(
        self,
        name: str,
        default_value: TValue,
        disabled: bool = False,
        cb_hook: Callable = lambda element: None,
    ) -> None:
        super().__init__(name, disabled=disabled, cb_hook=cb_hook)
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
        return self.gui_handle.get_value()

    @value.setter
    def value(self, value: TValue) -> None:
        if self.gui_handle is not None:
            self.gui_handle.set_value(value)
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
        cb_hook: Callable[[ViewerSlider], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert isinstance(default_value, (float, int))
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.min = min_value
        self.max = max_value
        self.step = step
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_slider(
            self.name, self.min, self.max, self.step, self.default_value, hint=self.hint
        )
        self.gui_handle.set_disabled(self.disabled)


class ViewerText(ViewerParameter[str]):
    """A text field in the viewer

    Args:
        name: The name of the text field
        default_value: The default value of the text field
        disabled: If the text field is disabled
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name: str,
        default_value: str,
        disabled: bool = False,
        cb_hook: Callable[[ViewerText], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert isinstance(default_value, str)
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_text(self.name, self.default_value, hint=self.hint)
        self.gui_handle.set_disabled(self.disabled)


class ViewerNumber(ViewerParameter[IntOrFloat], Generic[IntOrFloat]):
    """A number field in the viewer

    Args:
        name: The name of the number field
        default_value: The default value of the number field
        disabled: If the number field is disabled
        cb_hook: Callback to call on update
        hint: The hint text
    """

    default_value: IntOrFloat

    def __init__(
        self,
        name: str,
        default_value: IntOrFloat,
        disabled: bool = False,
        cb_hook: Callable[[ViewerNumber], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert isinstance(default_value, (float, int))
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_number(self.name, self.default_value, hint=self.hint)


class ViewerCheckbox(ViewerParameter[bool]):
    """A checkbox in the viewer

    Args:
        name: The name of the checkbox
        default_value: The default value of the checkbox
        disabled: If the checkbox is disabled
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name: str,
        default_value: bool,
        disabled: bool = False,
        cb_hook: Callable[[ViewerCheckbox], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert isinstance(default_value, bool)
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_checkbox(self.name, self.default_value, hint=self.hint)
        self.gui_handle.set_disabled(self.disabled)


TString = TypeVar("TString", default=str, bound=str)


class ViewerDropdown(ViewerParameter[TString], Generic[TString]):
    """A dropdown in the viewer

    Args:
        name: The name of the dropdown
        default_value: The default value of the dropdown
        options: The options of the dropdown
        disabled: If the dropdown is disabled
        cb_hook: Callback to call on update
        hint: The hint text
    """

    gui_handle: Optional[GuiSelectHandle[str]]

    def __init__(
        self,
        name: str,
        default_value: TString,
        options: List[TString],
        disabled: bool = False,
        cb_hook: Callable[[ViewerDropdown], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert default_value in options
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.options = options
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_select(
            self.name,
            self.options,  # type: ignore
            self.default_value,  # type: ignore
            hint=self.hint,
        )
        self.gui_handle.set_disabled(self.disabled)

    def set_options(self, new_options: List[TString]) -> None:
        """
        Sets the options of the dropdown,

        Args:
            new_options: The new options. If the current option isn't in the new options, the first option is selected.
        """
        self.options = new_options
        if self.gui_handle is not None:
            self.gui_handle.set_options(new_options)  # type: ignore


class ViewerButtonGroup(ViewerParameter[TString], Generic[TString]):
    """A button group in the viewer. Unlike other fields, cannot be disabled.

    Args:
        name: The name of the button group
        default_value: The default value of the button group
        options: The options of the button group
        cb_hook: Callback to call on update
    """

    gui_handle: Optional[GuiHandle[TString]]
    default_value: TString

    def __init__(
        self,
        name: str,
        default_value: TString,
        options: List[TString],
        cb_hook: Callable[[ViewerDropdown], Any] = lambda element: None,
    ):
        assert default_value in options
        super().__init__(name, default_value, disabled=False, cb_hook=cb_hook)
        self.options = options

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_button_group(self.name, self.options, self.default_value)  # type: ignore


class ViewerRGB(ViewerParameter[Tuple[int, int, int]]):
    """
    An RGB color picker for the viewer

    Args:
        name: The name of the color picker
        default_value: The default value of the color picker
        disabled: If the color picker is disabled
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name,
        default_value: Tuple[int, int, int],
        disabled=False,
        cb_hook: Callable[[ViewerRGB], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert len(default_value) == 3
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.add_gui_rgb(self.name, self.default_value, hint=self.hint)
        self.gui_handle.set_disabled(self.disabled)


class ViewerVec3(ViewerParameter[Tuple[float, float, float]]):
    """
    3 number boxes in a row to input a vector

    Args:
        name: The name of the vector
        default_value: The default value of the vector
        step: The step of the vector
        disabled: If the vector is disabled
        cb_hook: Callback to call on update
        hint: The hint text
    """

    def __init__(
        self,
        name,
        default_value: Tuple[float, float, float],
        step=0.1,
        disabled=False,
        cb_hook: Callable[[ViewerVec3], Any] = lambda element: None,
        hint: Optional[str] = None,
    ):
        assert len(default_value) == 3
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.step = step
        self.hint = hint

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.add_gui_vector3(self.name, self.default_value, self.step, hint=self.hint)
        self.gui_handle.set_disabled(self.disabled)
