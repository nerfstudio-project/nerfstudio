import os

from . import commands
from . import geometry
from . import visualizer
from . import transformations
from . import animation
from .visualizer import ViewerWindow, Visualizer


def viewer_assets_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "viewer"))
