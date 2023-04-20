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

""" Control panel for the viewer """
from collections import defaultdict
from typing import Callable, DefaultDict, List, Tuple

import torch

from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer.server.viewer_elements import (
    ViewerCheckbox,
    ViewerDropdown,
    ViewerElement,
    ViewerNumber,
    ViewerRGB,
    ViewerSlider,
    ViewerVec3,
)
from nerfstudio.viewer.viser import ViserServer


class ControlPanel:
    """
    Initializes the control panel with all the elements
    Args:
        time_enabled: whether or not the time slider should be enabled
        rerender_cb: a callback that will be called when the user changes a parameter that requires a rerender
            (eg train speed, max res, etc)
        crop_update_cb: a callback that will be called when the user changes the crop parameters
        update_output_cb: a callback that will be called when the user changes the output render
    """

    def __init__(self, time_enabled: bool, rerender_cb: Callable, crop_update_cb: Callable, update_output_cb: Callable):
        # elements holds a mapping from tag: [elements]
        self._elements_by_tag: DefaultDict[str, List[ViewerElement]] = defaultdict(lambda: [])

        self._train_speed = ViewerDropdown(
            "Train Speed", "Balanced", ["Fast", "Balanced", "Slow"], cb_hook=lambda han: self._train_speed_cb()
        )
        self._output_render = ViewerDropdown(
            "Output Render",
            "not set",
            ["not set"],
            cb_hook=lambda han: [self.update_control_panel(), rerender_cb(han), update_output_cb(han)],
        )
        self._colormap = ViewerDropdown("Colormap", "default", ["default"], cb_hook=rerender_cb)
        self._invert = ViewerCheckbox("| Invert", False, cb_hook=rerender_cb)
        self._normalize = ViewerCheckbox("| Normalize", False, cb_hook=rerender_cb)
        self._min = ViewerNumber("| Min", 0.0, cb_hook=rerender_cb)
        self._max = ViewerNumber("| Max", 1.0, cb_hook=rerender_cb)
        self._train_util = ViewerSlider("Train Util", 0.85, 0, 1, 0.05)
        self._max_res = ViewerSlider("Max Res", 512, 64, 2048, 100, cb_hook=rerender_cb)
        self._crop_viewport = ViewerCheckbox(
            "Crop Viewport",
            False,
            cb_hook=lambda han: [self.update_control_panel(), rerender_cb(han), crop_update_cb(han)],
        )
        self._background_color = ViewerRGB("| Background color", (38, 42, 55), cb_hook=crop_update_cb)
        self._crop_min = ViewerVec3("| Crop Min", (-1, -1, -1), 0.05, cb_hook=crop_update_cb)
        self._crop_max = ViewerVec3("| Crop Max", (1, 1, 1), 0.05, cb_hook=crop_update_cb)
        self._time = ViewerSlider("Time", 0.0, 0.0, 1.0, 0.01, cb_hook=rerender_cb)
        self._time_enabled = time_enabled

        self.add_element(self._train_speed)
        self.add_element(self._output_render)
        self.add_element(self._colormap)
        # colormap options
        self.add_element(self._invert, additional_tags=("colormap",))
        self.add_element(self._normalize, additional_tags=("colormap",))
        self.add_element(self._min, additional_tags=("colormap",))
        self.add_element(self._max, additional_tags=("colormap",))

        self.add_element(self._train_util)
        self.add_element(self._max_res)
        self.add_element(self._crop_viewport)
        # Crop options
        self.add_element(self._background_color, additional_tags=("crop",))
        self.add_element(self._crop_min, additional_tags=("crop",))
        self.add_element(self._crop_max, additional_tags=("crop",))

        self.add_element(self._time, additional_tags=("time",))

    def _train_speed_cb(self) -> None:
        """Callback for when the train speed is changed"""
        if self.train_speed == "Fast":
            self._train_util.value = 0.95
            self._max_res.value = 256
        elif self.train_speed == "Balanced":
            self._train_util.value = 0.85
            self._max_res.value = 512
        elif self.train_speed == "Slow":
            self._train_util.value = 0.5
            self._max_res.value = 1024

    def install(self, viser_server: ViserServer) -> None:
        """Installs the control panel on the viser server

        Args:
            viser_server: the viser server
        """
        for e in self._elements_by_tag["all"]:
            e.install(viser_server)
        self.update_control_panel()

    def update_output_options(self, new_options: List[str]):
        """
        Args:
            new_options: a list of new output options
        """
        self._output_render.set_options(new_options)

    def add_element(self, e: ViewerElement, additional_tags: Tuple[str] = tuple()) -> None:
        """Adds an element to the control panel

        Args:
            e: the element to add
            additional_tags: additional tags to add to the element for selection
        """
        self._elements_by_tag["all"].append(e)
        for t in additional_tags:
            self._elements_by_tag[t].append(e)

    def update_control_panel(self) -> None:
        """
        Sets elements to be hidden or not based on the current state of the control panel
        """
        self._colormap.set_disabled(self.output_render == "rgb")
        for e in self._elements_by_tag["colormap"]:
            e.set_hidden(self.output_render == "rgb")
        for e in self._elements_by_tag["crop"]:
            e.set_hidden(not self.crop_viewport)
        self._time.set_hidden(not self._time_enabled)

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        colormap_options = []
        if dimensions == 3:
            colormap_options = [viewer_utils.ColormapTypes.DEFAULT.value]
        if dimensions == 1 and dtype == torch.float:
            colormap_options = [c.value for c in list(viewer_utils.ColormapTypes)[1:]]

        self._colormap.set_options(colormap_options)

    @property
    def train_speed(self) -> str:
        """Returns the current train speed setting"""
        return self._train_speed.value

    @property
    def output_render(self) -> str:
        """Returns the current output render"""
        return self._output_render.value

    @property
    def colormap(self) -> str:
        """Returns the current colormap"""
        return self._colormap.value

    @property
    def colormap_invert(self) -> bool:
        """Returns the current colormap invert setting"""
        return self._invert.value

    @property
    def colormap_normalize(self) -> bool:
        """Returns the current colormap normalize setting"""
        return self._normalize.value

    @property
    def colormap_min(self) -> float:
        """Returns the current colormap min setting"""
        return self._min.value

    @property
    def colormap_max(self) -> float:
        """Returns the current colormap max setting"""
        return self._max.value

    @property
    def train_util(self) -> float:
        """Returns the current train util setting"""
        return self._train_util.value

    @property
    def max_res(self) -> int:
        """Returns the current max res setting"""
        return self._max_res.value

    @property
    def crop_viewport(self) -> bool:
        """Returns the current crop viewport setting"""
        return self._crop_viewport.value

    @crop_viewport.setter
    def crop_viewport(self, value: bool):
        """Sets the crop viewport setting"""
        self._crop_viewport.value = value

    @property
    def crop_min(self) -> Tuple[float, float, float]:
        """Returns the current crop min setting"""
        return self._crop_min.value

    @crop_min.setter
    def crop_min(self, value: Tuple[float, float, float]):
        """Sets the crop min setting"""
        self._crop_min.value = value

    @property
    def crop_max(self) -> Tuple[float, float, float]:
        """Returns the current crop max setting"""
        return self._crop_max.value

    @crop_max.setter
    def crop_max(self, value: Tuple[float, float, float]):
        """Sets the crop max setting"""
        self._crop_max.value = value

    @property
    def background_color(self) -> Tuple[int, int, int]:
        """Returns the current background color"""
        return self._background_color.value

    @background_color.setter
    def background_color(self, value: Tuple[int, int, int]):
        """Sets the background color"""
        self._background_color.value = value

    @property
    def time(self) -> float:
        """Returns the current background color"""
        return self._time.value

    @time.setter
    def time(self, value: float):
        """Sets the background color"""
        self._time.value = value
