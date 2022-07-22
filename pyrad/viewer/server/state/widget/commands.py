from dataclasses import dataclass

from matplotlib.widgets import Widget


@dataclass
class SetWidget:
    __slots__ = ["path"]
    widget: Widget
    path: str

    def lower(self):
        return {"type": "set_widget", "widget": self.widget.lower(), "path": self.path.lower()}
