# image stuff
import cv2
import imageio
import mediapy as media

# ipython stuff
from IPython import get_ipython

# plotly stuff
import plotly.graph_objects as go


def setup_ipynb():
    """
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    https://stackoverflow.com/questions/35595766/matplotlib-line-magic-causes-syntaxerror-in-python-script
    This gets reference to the InteractiveShell instance
    """
    try:
        from IPython import get_ipython

        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
        get_ipython().run_line_magic("matplotlib", "inline")
        return True
    except:
        return False


setup_ipynb()
