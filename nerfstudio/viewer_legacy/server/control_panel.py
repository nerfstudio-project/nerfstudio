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

"""Control panel for the viewer"""

from collections import defaultdict
from typing import Callable, DefaultDict, List, Tuple, get_args

import torch

from nerfstudio.utils.colormaps import ColormapOptions, Colormaps
from nerfstudio.viewer_legacy.server.viewer_elements import (
    ViewerButtonGroup,
    ViewerCheckbox,
    ViewerDropdown,
    ViewerElement,
    ViewerNumber,
    ViewerRGB,
    ViewerSlider,
    ViewerVec3,
)
from nerfstudio.viewer_legacy.viser import ViserServer


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

    def __init__(
        self,
        viser_server: ViserServer,
        time_enabled: bool,
        rerender_cb: Callable,
        crop_update_cb: Callable,
        update_output_cb: Callable,
        update_split_output_cb: Callable,
    ):
        # elements holds a mapping from tag: [elements]
        self.viser_server = viser_server
        self._elements_by_tag: DefaultDict[str, List[ViewerElement]] = defaultdict(lambda: [])

        self._train_speed = ViewerButtonGroup(
            name="Train Speed  ",
            default_value="Balanced",
            options=["Slow", "Balanced", "Fast"],
            cb_hook=lambda han: self._train_speed_cb(),
        )
        self._output_render = ViewerDropdown(
            "Output Render",
            "not set",
            ["not set"],
            cb_hook=lambda han: [self.update_control_panel(), update_output_cb(han), rerender_cb(han)],
            hint="The output to render",
        )
        self._colormap = ViewerDropdown[Colormaps](
            "Colormap", "default", ["default"], cb_hook=rerender_cb, hint="The colormap to use"
        )
        self._invert = ViewerCheckbox("Invert", False, cb_hook=rerender_cb, hint="Invert the colormap")
        self._normalize = ViewerCheckbox("Normalize", True, cb_hook=rerender_cb, hint="Normalize the colormap")
        self._min = ViewerNumber("Min", 0.0, cb_hook=rerender_cb, hint="Min value of the colormap")
        self._max = ViewerNumber("Max", 1.0, cb_hook=rerender_cb, hint="Max value of the colormap")

        self._split = ViewerCheckbox(
            "Enable",
            False,
            cb_hook=lambda han: [self.update_control_panel(), rerender_cb(han)],
            hint="Render two outputs",
        )
        self._split_percentage = ViewerSlider(
            "Split Percentage", 0.5, 0.0, 1.0, 0.01, cb_hook=rerender_cb, hint="Where to split"
        )
        self._split_output_render = ViewerDropdown(
            "Output Render Split",
            "not set",
            ["not set"],
            cb_hook=lambda han: [self.update_control_panel(), update_split_output_cb(han), rerender_cb(han)],
            hint="The second output",
        )
        # Hack: spaces are after at the end of the names to make them unique
        self._split_colormap = ViewerDropdown[Colormaps](
            "Colormap ", "default", ["default"], cb_hook=rerender_cb, hint="Colormap of the second output"
        )
        self._split_invert = ViewerCheckbox(
            "Invert ", False, cb_hook=rerender_cb, hint="Invert the colormap of the second output"
        )
        self._split_normalize = ViewerCheckbox(
            "Normalize ", True, cb_hook=rerender_cb, hint="Normalize the colormap of the second output"
        )
        self._split_min = ViewerNumber(
            "Min ", 0.0, cb_hook=rerender_cb, hint="Min value of the colormap of the second output"
        )
        self._split_max = ViewerNumber(
            "Max ", 1.0, cb_hook=rerender_cb, hint="Max value of the colormap of the second output"
        )

        self._train_util = ViewerSlider(
            "Train Util",
            default_value=0.85,
            min_value=0.0,
            max_value=1,
            step=0.05,
            hint="Target training utilization, 0.0 is slow, 1.0 is fast. Doesn't affect final render quality",
        )
        self._max_res = ViewerSlider(
            "Max Res", 512, 64, 2048, 100, cb_hook=rerender_cb, hint="Maximum resolution to render in viewport"
        )
        self._crop_viewport = ViewerCheckbox(
            "Enable ",
            False,
            cb_hook=lambda han: [self.update_control_panel(), crop_update_cb(han), rerender_cb(han)],
            hint="Crop the scene to a specified box",
        )
        self._background_color = ViewerRGB(
            "Background color", (38, 42, 55), cb_hook=crop_update_cb, hint="Color of the background"
        )
        self._crop_min = ViewerVec3(
            "Crop Min", (-1, -1, -1), 0.05, cb_hook=crop_update_cb, hint="Minimum value of the crop"
        )
        self._crop_max = ViewerVec3(
            "Crop Max", (1, 1, 1), 0.05, cb_hook=crop_update_cb, hint="Maximum value of the crop"
        )
        self._time = ViewerSlider("Time", 0.0, 0.0, 1.0, 0.01, cb_hook=rerender_cb, hint="Time to render")
        self._time_enabled = time_enabled

        self.add_element(self._train_speed)
        self.add_element(self._train_util)
        with self.viser_server.gui_folder("Render Options"):
            self.add_element(self._max_res)
            self.add_element(self._output_render)
            self.add_element(self._colormap)
            # colormap options
            with self.viser_server.gui_folder(" "):
                self.add_element(self._invert, additional_tags=("colormap",))
                self.add_element(self._normalize, additional_tags=("colormap",))
                self.add_element(self._min, additional_tags=("colormap",))
                self.add_element(self._max, additional_tags=("colormap",))

        # split options
        with self.viser_server.gui_folder("Split Screen"):
            self.add_element(self._split)

            self.add_element(self._split_percentage, additional_tags=("split",))
            self.add_element(self._split_output_render, additional_tags=("split",))
            self.add_element(self._split_colormap, additional_tags=("split",))
            with self.viser_server.gui_folder("  "):
                self.add_element(self._split_invert, additional_tags=("split_colormap",))
                self.add_element(self._split_normalize, additional_tags=("split_colormap",))
                self.add_element(self._split_min, additional_tags=("split_colormap",))
                self.add_element(self._split_max, additional_tags=("split_colormap",))

        with self.viser_server.gui_folder("Crop Viewport"):
            self.add_element(self._crop_viewport)

            # Crop options
            self.add_element(self._background_color, additional_tags=("crop",))
            self.add_element(self._crop_min, additional_tags=("crop",))
            self.add_element(self._crop_max, additional_tags=("crop",))

        self.add_element(self._time, additional_tags=("time",))

    def _train_speed_cb(self) -> None:
        """Callback for when the train speed is changed"""
        if self._train_speed.value == "Fast":
            self._train_util.value = 0.95
            self._max_res.value = 256
        elif self._train_speed.value == "Balanced":
            self._train_util.value = 0.85
            self._max_res.value = 512
        elif self._train_speed.value == "Slow":
            self._train_util.value = 0.5
            self._max_res.value = 1024

    def update_output_options(self, new_options: List[str]):
        """
        Args:
            new_options: a list of new output options
        """
        self._output_render.set_options(new_options)
        self._split_output_render.set_options(new_options)
        self._split_output_render.value = new_options[-1]

    def add_element(self, e: ViewerElement, additional_tags: Tuple[str, ...] = tuple()) -> None:
        """Adds an element to the control panel

        Args:
            e: the element to add
            additional_tags: additional tags to add to the element for selection
        """
        self._elements_by_tag["all"].append(e)
        for t in additional_tags:
            self._elements_by_tag[t].append(e)
        e.install(self.viser_server)

    def update_control_panel(self) -> None:
        """
        Sets elements to be hidden or not based on the current state of the control panel
        """
        self._colormap.set_disabled(self.output_render == "rgb")
        for e in self._elements_by_tag["colormap"]:
            e.set_hidden(self.output_render == "rgb")
        for e in self._elements_by_tag["split_colormap"]:
            e.set_hidden(not self._split.value or self.split_output_render == "rgb")
        for e in self._elements_by_tag["crop"]:
            e.set_hidden(not self.crop_viewport)
        self._time.set_hidden(not self._time_enabled)
        self._split_percentage.set_hidden(not self._split.value)
        self._split_output_render.set_hidden(not self._split.value)
        self._split_colormap.set_hidden(not self._split.value)
        self._split_colormap.set_disabled(self.split_output_render == "rgb")

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        self._colormap.set_options(_get_colormap_options(dimensions, dtype))

    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the split colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        self._split_colormap.set_options(_get_colormap_options(dimensions, dtype))

    @property
    def output_render(self) -> str:
        """Returns the current output render"""
        return self._output_render.value

    @property
    def split_output_render(self) -> str:
        """Returns the current output for the split render"""
        return self._split_output_render.value

    @property
    def split(self) -> bool:
        """Returns whether the split is enabled"""
        return self._split.value

    @property
    def split_percentage(self) -> float:
        """Returns the percentage of the screen to split"""
        return self._split_percentage.value

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

    @property
    def colormap_options(self) -> ColormapOptions:
        """Returns the current colormap options"""
        return ColormapOptions(
            colormap=self._colormap.value,
            normalize=self._normalize.value,
            colormap_min=self._min.value,
            colormap_max=self._max.value,
            invert=self._invert.value,
        )

    @property
    def split_colormap_options(self) -> ColormapOptions:
        """Returns the current colormap options"""
        return ColormapOptions(
            colormap=self._split_colormap.value,
            normalize=self._split_normalize.value,
            colormap_min=self._split_min.value,
            colormap_max=self._split_max.value,
            invert=self._split_invert.value,
        )


def _get_colormap_options(dimensions: int, dtype: type) -> List[Colormaps]:
    """
    Given the number of dimensions and data type, returns a list of available colormap options
    to use with the visualize() function.

    Args:
        dimensions: the number of dimensions of the render
        dtype: the data type of the render
    Returns:
        a list of available colormap options
    """
    colormap_options: List[Colormaps] = []
    if dimensions == 3:
        colormap_options = ["default"]
    if dimensions == 1 and dtype == torch.float:
        colormap_options = [c for c in list(get_args(Colormaps)) if c not in ("default", "pca")]
    if dimensions > 3:
        colormap_options = ["pca"]
    return colormap_options
