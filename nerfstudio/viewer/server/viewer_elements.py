# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

""" Viewer GUI elements for the nerfstudio viewer """

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from nerfstudio.viewer.viser import GuiHandle, ViserServer

IntOrFloat = TypeVar("IntOrFloat", int, float)


class ViewerElement:
    """Base class for all viewer elements

    Args:
        name: The name of the element
        disabled: If the element is disabled
    """

    def __init__(self, name: str, disabled: bool = False, cb_hook: Callable = lambda element: None):
        self.name = name
        self.gui_handle: Optional[GuiHandle] = None
        self.disabled = disabled
        self.cb_hook = cb_hook

    @abstractmethod
    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        """
        Returns the GuiHandle object which actually controls the parameter in the gui.

        Args:
            viser_server: The server to install the gui element into.
        """
        ...

    def remove(self):
        """Removes the gui element from the viewer"""
        if self.gui_handle is not None:
            self.gui_handle.remove()
            self.gui_handle = None

    def set_hidden(self, hidden: bool):
        """Sets the hidden state of the gui element"""
        self.gui_handle.set_hidden(hidden)

    def set_disabled(self, disabled: bool):
        """Sets the disabled state of the gui element"""
        self.gui_handle.set_disabled(disabled)

    @abstractmethod
    def install(self, viser_server: ViserServer) -> None:
        """Installs the gui element into the given viser_server"""
        ...


class ViewerButton(ViewerElement):
    """A button in the viewer

    Args:
        name: The name of the button
        call_fn: The function to call when the button is pressed
        disabled: If the button is disabled
    """

    def __init__(self, name: str, call_fn: Callable, disabled: bool = False):
        super().__init__(name, disabled=disabled, cb_hook=call_fn)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.add_gui_button(self.name)
        self.gui_handle.set_disabled(self.disabled)

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        def call_fn(_: GuiHandle):  # pylint: disable=unused-argument
            self.cb_hook(self)

        assert self.gui_handle is not None
        self.gui_handle.on_update(call_fn)


class ViewerParameter(ViewerElement):
    """A viewer element with state

    Args:
        name: The name of the element
        default_value: The default value of the element
        disabled: If the element is disabled
        cb_hook: Callback to call on update
    """

    def __init__(self, name: str, default_value: Any, disabled: bool = False, cb_hook: Callable = lambda element: None):
        super().__init__(name, disabled=disabled, cb_hook=cb_hook)
        self.def_value = default_value

    def install(self, viser_server: ViserServer) -> None:
        """
        Based on the type provided by default_value, installs a gui element inside the given viser_server

        Args:
            viser_server: The server to install the gui element into.
        """
        self._create_gui_handle(viser_server)

        def update_fn(_: GuiHandle):
            self.cb_hook(self)

        assert self.gui_handle is not None
        self.gui_handle.on_update(update_fn)

    @abstractmethod
    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        ...

    @property
    def value(self) -> Any:
        """Returns the current value of the viewer element"""
        if self.gui_handle is None:
            return self.def_value
        return self.gui_handle.get_value()

    @value.setter
    def value(self, value: Any):
        if self.gui_handle is not None:
            self.gui_handle.set_value(value)
        else:
            self.def_value = value


class ViewerSlider(ViewerParameter):
    """A slider in the viewer

    Args:
        name: The name of the slider
        default_value: The default value of the slider
        min_value: The minimum value of the slider
        max_value: The maximum value of the slider
        step: The step size of the slider
        disabled: If the slider is disabled
        cb_hook: Callback to call on update
    """

    def __init__(
        self,
        name: str,
        default_value: IntOrFloat,
        min_value: IntOrFloat,
        max_value: IntOrFloat,
        step: IntOrFloat = 0.1,
        disabled: bool = False,
        cb_hook: Callable = lambda element: None,
    ):
        assert isinstance(default_value, (float, int))
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.min = min_value
        self.max = max_value
        self.step = step

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_slider(self.name, self.min, self.max, self.step, self.def_value)
        self.gui_handle.set_disabled(self.disabled)


class ViewerText(ViewerParameter):
    """A text field in the viewer

    Args:
        name: The name of the text field
        default_value: The default value of the text field
        disabled: If the text field is disabled
        cb_hook: Callback to call on update
    """

    def __init__(
        self,
        name: str,
        default_value: str,
        disabled: bool = False,
        cb_hook: Callable = lambda element: None,
    ):
        assert isinstance(default_value, str)
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_text(self.name, self.def_value)
        self.gui_handle.set_disabled(self.disabled)


class ViewerNumber(ViewerParameter):
    """A number field in the viewer

    Args:
        name: The name of the number field
        default_value: The default value of the number field
        disabled: If the number field is disabled
        cb_hook: Callback to call on update
    """

    def __init__(
        self,
        name: str,
        default_value: IntOrFloat,
        disabled: bool = False,
        cb_hook: Callable = lambda element: None,
    ):
        assert isinstance(default_value, (float, int))
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_number(self.name, self.def_value)


class ViewerCheckbox(ViewerParameter):
    """A checkbox in the viewer

    Args:
        name: The name of the checkbox
        default_value: The default value of the checkbox
        disabled: If the checkbox is disabled
        cb_hook: Callback to call on update
    """

    def __init__(
        self,
        name: str,
        default_value: bool,
        disabled: bool = False,
        cb_hook: Callable = lambda element: None,
    ):
        assert isinstance(default_value, bool)
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_checkbox(self.name, self.def_value)
        self.gui_handle.set_disabled(self.disabled)


class ViewerDropdown(ViewerParameter):
    """A dropdown in the viewer

    Args:
        name: The name of the dropdown
        default_value: The default value of the dropdown
        options: The options of the dropdown
        disabled: If the dropdown is disabled
        cb_hook: Callback to call on update
    """

    def __init__(
        self,
        name: str,
        default_value: str,
        options: List[str],
        disabled: bool = False,
        cb_hook: Callable = lambda element: None,
    ):
        assert default_value in options
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.options = options

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_select(self.name, self.options, self.def_value)
        self.gui_handle.set_disabled(self.disabled)

    def set_options(self, new_options: List[str]) -> None:
        """
        Sets the options of the dropdown,
        new_options: The new options. if the current option isn't in the new options, the first option is selected
        """
        self.options = new_options
        self.gui_handle.set_options(new_options)


class ViewerRGB(ViewerParameter):
    """
    An RGB color picker for the viewer
    """

    def __init__(
        self,
        name,
        default_value: Tuple[int, int, int],
        disabled=False,
        cb_hook: Callable = lambda element: None,
    ):
        assert len(default_value) == 3
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.add_gui_rgb(self.name, self.def_value)
        self.gui_handle.set_disabled(self.disabled)


class ViewerVec3(ViewerParameter):
    """
    3 number boxes in a row to input a vector
    """

    def __init__(
        self,
        name,
        default_value: Tuple[float, float, float],
        step=0.1,
        disabled=False,
        cb_hook: Callable = lambda element: None,
    ):
        assert len(default_value) == 3
        super().__init__(name, default_value, disabled=disabled, cb_hook=cb_hook)
        self.step = step

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.add_gui_vector3(self.name, self.def_value, self.step)
        self.gui_handle.set_disabled(self.disabled)
