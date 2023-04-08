from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List, Tuple

from nerfstudio.viewer.server.viewer_param import *


class ControlPanel:
    def __init__(self, rerender_cb: Callable):
        """
        Initializes the control panel with all the elements
        Args:
             rerender_cb: a callback that will be called when the user changes a parameter that requires a rerender
                (eg train speed, max res, etc)
        """
        # elements holds a mapping from tag: [elements]
        self.elements_by_tag: DefaultDict[str, List[ViewerElement]] = defaultdict(lambda: [])
        self.elements_by_name: Dict[str, ViewerElement] = {}
        self.add_element(ViewerDropdown("Train Speed", "Balanced", ["Fast", "Balanced", "Slow"]))
        self.add_element(ViewerDropdown("Output Render", "rgb", ["rgb"], cb_hook=rerender_cb))
        self.add_element(
            ViewerDropdown(
                "Colormap", "default", ["default"], cb_hook=lambda: [self.update_control_panel(), rerender_cb()]
            )
        )
        # colormap options
        self.add_element(ViewerCheckbox("Invert", False), additional_tags=("colormap",))
        self.add_element(ViewerCheckbox("Normalize", False), additional_tags=("colormap",))
        self.add_element(ViewerNumber("Min", 0.0), additional_tags=("colormap",))
        self.add_element(ViewerNumber("Max", 1.0), additional_tags=("colormap",))

        self.add_element(ViewerSlider("Train Util", 0.9, 0, 1, 0.05))
        self.add_element(ViewerSlider("Max Res.", 500, 100, 2000, 100, cb_hook=rerender_cb))
        self.add_element(ViewerCheckbox("Crop Viewport", False, cb_hook=self.update_control_panel))
        # Crop options
        self.add_element(ViewerVec3("Crop Min", (0, 0, 0), 0.05), additional_tags=("crop",))
        self.add_element(ViewerVec3("Crop Max", (1, 1, 1), 0.05), additional_tags=("crop",))
        # TODO add background color

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
        for e in self.elements_by_tag["colormap"]:
            e.set_hidden(self.output_render == "rgb")
        for e in self.elements_by_tag["crop"]:
            e.set_hidden(self.crop_viewport)

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
        return self._get_element_by_name("Max Res.").value

    @property
    def crop_viewport(self):
        return self._get_element_by_name("Crop Viewport").value

    @property
    def crop_min(self):
        return self._get_element_by_name("Crop Min").value

    @property
    def crop_max(self):
        return self._get_element_by_name("Crop Max").value
