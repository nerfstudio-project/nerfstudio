from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List, Tuple

from nerfstudio.viewer.server.viewer_param import *


class ControlPanel:
    def __init__(self, rerender_cb: Callable, crop_update_cb: Callable):
        """
        Initializes the control panel with all the elements
        Args:
            rerender_cb: a callback that will be called when the user changes a parameter that requires a rerender
                (eg train speed, max res, etc)
            crop_update_cb: a callback that will be called when the user changes the crop parameters
        """
        # elements holds a mapping from tag: [elements]
        self.elements_by_tag: DefaultDict[str, List[ViewerElement]] = defaultdict(lambda: [])
        self.elements_by_name: Dict[str, ViewerElement] = {}
        self.add_element(
            ViewerDropdown("Train Speed", "Balanced", ["Fast", "Balanced", "Slow"], cb_hook=self.train_speed_cb)
        )
        self.add_element(
            ViewerDropdown(
                "Output Render", "rgb", ["rgb", "depth"], cb_hook=lambda: [self.update_control_panel(), rerender_cb()]
            )
        )
        self.add_element(ViewerDropdown("Colormap", "default", ["default"], cb_hook=rerender_cb))
        # colormap options
        self.add_element(ViewerCheckbox("Invert", False), additional_tags=("colormap",))
        self.add_element(ViewerCheckbox("Normalize", False), additional_tags=("colormap",))
        self.add_element(ViewerNumber("Min", 0.0), additional_tags=("colormap",))
        self.add_element(ViewerNumber("Max", 1.0), additional_tags=("colormap",))

        self.add_element(ViewerSlider("Train Util", 0.9, 0, 1, 0.05))
        self.add_element(ViewerSlider("Max Res", 1024, 64, 2048, 100, cb_hook=rerender_cb))
        self.add_element(
            ViewerCheckbox(
                "Crop Viewport", False, cb_hook=lambda: [self.update_control_panel(), rerender_cb(), crop_update_cb()]
            )
        )
        # Crop options
        self.add_element(
            ViewerRGB(
                "Background color",
                (38, 42, 55),
                cb_hook=crop_update_cb,
            ),
            additional_tags=("crop",),
        )
        self.add_element(
            ViewerVec3(
                "Crop Min",
                (-1, -1, -1),
                0.05,
                cb_hook=crop_update_cb,
            ),
            additional_tags=("crop",),
        )
        self.add_element(
            ViewerVec3(
                "Crop Max",
                (1, 1, 1),
                0.05,
                cb_hook=crop_update_cb,
            ),
            additional_tags=("crop",),
        )

    def train_speed_cb(self):
        print("inside train speed cb")
        if self.train_speed == "Fast":
            self._get_element_by_name("Train Util").set_value(0.95)
        elif self.train_speed == "Balanced":
            self._get_element_by_name("Train Util").set_value(0.85)
        elif self.train_speed == "Slow":
            self._get_element_by_name("Train Util").set_value(0.5)

    def install(self, viser_server: ViserServer):
        for e in self.elements_by_name.values():
            e.install(viser_server)
        self.update_control_panel()

    def update_output_options(self, new_options: List[str]):
        raise NotImplementedError("Not implemented yet")

    def add_element(self, e: ViewerElement, additional_tags: Tuple[str] = tuple()):
        """
        Adds an element to the control panel
        """
        self.elements_by_tag["all"].append(e)
        for t in additional_tags:
            self.elements_by_tag[t].append(e)
        assert e.name not in self.elements_by_name
        self.elements_by_name[e.name] = e

    def update_control_panel(self):
        """
        Sets elements to be hidden or not based on the current state of the control panel
        """
        self._get_element_by_name("Colormap").set_disabled(self.output_render == "rgb")
        for e in self.elements_by_tag["colormap"]:
            e.set_hidden(self.output_render == "rgb")
        for e in self.elements_by_tag["crop"]:
            e.set_hidden(not self.crop_viewport)

    def _get_element_by_name(self, name):
        return self.elements_by_name[name]

    @property
    def train_speed(self):
        return self._get_element_by_name("Train Speed").value

    @property
    def output_render(self):
        return self._get_element_by_name("Output Render").value

    @property
    def colormap(self):
        return self._get_element_by_name("Colormap").value

    @property
    def colormap_invert(self):
        return self._get_element_by_name("Invert").value

    @property
    def colormap_normalize(self):
        return self._get_element_by_name("Normalize").value

    @property
    def colormap_min(self):
        return self._get_element_by_name("Min").value

    @property
    def colormap_max(self):
        return self._get_element_by_name("Max").value

    @property
    def train_util(self):
        return self._get_element_by_name("Train Util").value

    @property
    def max_res(self):
        return self._get_element_by_name("Max Res").value

    @property
    def crop_viewport(self):
        return self._get_element_by_name("Crop Viewport").value

    @property
    def crop_min(self):
        return self._get_element_by_name("Crop Min").value

    @property
    def crop_max(self):
        return self._get_element_by_name("Crop Max").value

    @property
    def background_color(self):
        return self._get_element_by_name("Background color").value
