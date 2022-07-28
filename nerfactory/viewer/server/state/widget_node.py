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