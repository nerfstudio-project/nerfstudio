# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""
Code for the UI components to be used in the viewer.
This is for synchronizing the web UI views with the state stored 
in the BridgeServer (server.py).

Leva, our frontend framework for widgets, supports the following:
# Number
# Range (slider)
# Color (color picker)
# Boolean (toggle)
# Interval
"""


from nerfactory.viewer.server.state.node import Node


class WidgetNode(Node):
    """
    The base class Widget.
    """
    __slots__ = ["widget"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)s
        self.widget = None


class Widget:
    """TODO"""


class Number(Widget):
    """
    The Number widget.
    """

    __slots__ = ["name", "value"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None
        self.value = None


class Slider(Widget):
    """TODO"""

class ColorPicker(Widget):
    """TODO"""

class Color(Widget):
    """TODO"""