from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from nerfstudio.viewer.viser import GuiHandle, ViserServer


class ViewerElement:
    def __init__(self, name: str, disabled=False):
        self.name = name
        self.gui_handle: GuiHandle = None
        self.disabled = disabled

    @abstractmethod
    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        """
        Returns the GuiHandle object which actually controls the parameter in the gui
        """
        ...

    def remove(self):
        self.gui_handle.remove()
        self.gui_handle = None

    @abstractmethod
    def install(self, viser_server: ViserServer) -> None:
        ...


class ViewerButton(ViewerElement):
    def __init__(self, name: str, call_fn: Callable, disabled=False):
        super().__init__(name, disabled=disabled)
        self.fn = call_fn

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.add_gui_button(self.name, disabled=self.disabled)

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        def call_fn(handle):
            print("call_fn called")
            self.fn()

        self.gui_handle.on_update(call_fn)


class ViewerParameter(ViewerElement):
    def __init__(self, name: str, default_value, disabled=False, cb_hook: Callable = lambda: None):
        super().__init__(name, disabled=disabled)
        self.cur_value = default_value
        self.cb_hook = cb_hook

    def install(self, viser_server: ViserServer) -> None:
        """
        Based on the type provided by default_value, installs a gui element inside the given viser_server
        """
        self._create_gui_handle(viser_server)

        def update_fn(handle):
            self.cur_value = handle.get_value()
            self.cb_hook()

        self.gui_handle.on_update(update_fn)

    @property
    def value(self):
        return self.cur_value


class ViewerSlider(ViewerParameter):
    def __init__(self, name: str, default_value, min_value, max_value, step, disabled=False, **kwargs):
        assert isinstance(default_value, float) or isinstance(default_value, int)
        super().__init__(name, default_value, disabled=disabled,**kwargs)
        self.min = min_value
        self.max = max_value
        self.step = step

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_slider(
            self.name, self.min, self.max, self.step, self.cur_value, disabled=self.disabled
        )


class ViewerText(ViewerParameter):
    def __init__(self, name, default_value, disabled=False,**kwargs):
        assert isinstance(default_value, str)
        super().__init__(name, default_value, disabled=disabled,**kwargs)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_text(self.name, self.cur_value, disabled=self.disabled)


class ViewerNumber(ViewerParameter):
    def __init__(self, name, default_value, disabled=False,**kwargs):
        assert isinstance(default_value, float) or isinstance(default_value, int)
        super().__init__(name, default_value, disabled=disabled,**kwargs)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_number(self.name, self.cur_value, disabled=self.disabled)


class ViewerCheckbox(ViewerParameter):
    def __init__(self, name, default_value, disabled=False,**kwargs):
        assert isinstance(default_value, bool)
        super().__init__(name, default_value, disabled=disabled,**kwargs)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_checkbox(self.name, self.cur_value, disabled=self.disabled)


class ViewerDropdown(ViewerParameter):
    def __init__(self, name, default_value, options: List, disabled=False,**kwargs):
        assert default_value in options
        super().__init__(name, default_value, disabled=disabled,**kwargs)
        self.options = options

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_select(self.name, self.options, self.cur_value, disabled=self.disabled)


class ViewerVec3(ViewerParameter):
    def __init__(self, name, default_value: Tuple[float, float, float], step, disabled=False,**kwargs):
        assert len(default_value) == 3
        super().__init__(name, default_value, disabled=disabled,**kwargs)
        self.step = step

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.add_gui_vector3(self.name, self.cur_value, self.step, disabled=self.disabled)
