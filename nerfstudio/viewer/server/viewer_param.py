from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from nerfstudio.viewer.viser import GuiHandle, ViserServer


class ViewerParameter:
    def __init__(self, name: str, default_value, min_value=None, max_value=None):
        self.name = name
        self.param_type = type(default_value)
        self.cur_value = default_value
        self.gui_handle = None

    def create_gui_element(self, viser_server: ViserServer) -> GuiHandle:
        """
        Based on the type provided by default_value, installs a gui element inside the given viser_server
        """
        if self.param_type == str:
            self.gui_handle = viser_server.add_gui_text(self.name, self.cur_value)
        elif self.param_type == int or self.param_type == float:
            self.gui_handle = viser_server.add_gui_number(self.name, self.cur_value)
        else:
            raise RuntimeError("Unsupported datatype given to ViewerParameter")

        def update_fn(handle):
            self.cur_value = handle.get_value()

        self.gui_handle.on_update(update_fn)

    @property
    def value(self):
        return self.cur_value
