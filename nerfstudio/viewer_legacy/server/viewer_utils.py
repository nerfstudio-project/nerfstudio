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


"""Code to interface with the `vis/` (the JS viewer)."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
from nerfstudio.utils.io import load_from_json


def get_viewer_version() -> str:
    """Return the version of the viewer."""
    json_filename = os.path.join(os.path.dirname(__file__), "../app/package.json")
    version = load_from_json(Path(json_filename))["version"]
    return version


def get_viewer_url(websocket_port: int) -> str:
    """Generate URL for the viewer.

    Args:
        websocket_port: port to connect to the viewer
    Returns:
        URL to the viewer
    """
    version = get_viewer_version()
    websocket_url = f"ws://localhost:{websocket_port}"
    return f"https://viewer.nerf.studio/versions/{version}/?websocket_url={websocket_url}"


class IOChangeException(Exception):
    """Basic camera exception to interrupt viewer"""


class SetTrace:
    """Basic trace function"""

    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.settrace(None)


def is_port_open(port: int):
    """Returns True if the port is open.

    Args:
        port: Port to check.

    Returns:
        True if the port is open, False otherwise.
    """
    try:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _ = sock.bind(("", port))
        sock.close()
        return True
    except OSError:
        return False


def get_free_port(default_port: Optional[int] = None):
    """Returns a free port on the local machine. Try to use default_port if possible.

    Args:
        default_port: Port to try to use.

    Returns:
        A free port on the local machine.
    """
    if default_port is not None:
        if is_port_open(default_port):
            return default_port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    return port


def update_render_aabb(
    crop_viewport: bool, crop_min: Tuple[float, float, float], crop_max: Tuple[float, float, float], model: Model
):
    """
    update the render aabb box for the viewer:

    Args:
        crop_viewport: whether to crop the viewport
        crop_min: min of the crop box
        crop_max: max of the crop box
        model: the model to render
    """

    if crop_viewport:
        crop_min_tensor = torch.tensor(crop_min, dtype=torch.float32)
        crop_max_tensor = torch.tensor(crop_max, dtype=torch.float32)

        if isinstance(model.render_aabb, SceneBox):
            model.render_aabb.aabb[0] = crop_min_tensor
            model.render_aabb.aabb[1] = crop_max_tensor
        else:
            model.render_aabb = SceneBox(aabb=torch.stack([crop_min_tensor, crop_max_tensor], dim=0))
    else:
        model.render_aabb = None
