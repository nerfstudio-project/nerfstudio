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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
from torch import nn


@dataclass
class CameraState:
    """A dataclass for storing the camera state."""

    fov: float
    """ The field of view of the camera. """
    aspect: float
    """ The aspect ratio of the image. """
    c2w: Float[torch.Tensor, "3 4"]
    """ The camera matrix. """


def get_camera(
    camera_state: CameraState, image_height: int, image_width: Optional[Union[int, float]] = None
) -> Cameras:
    """Returns the camera intrinsics matrix and the camera to world homogeneous matrix.

    Args:
        camera_state: the camera state
        image_size: the size of the image (height, width)
    """
    # intrinsics
    fov = camera_state.fov
    aspect = camera_state.aspect
    if image_width is None:
        image_width = aspect * image_height
    pp_w = image_width / 2.0
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov / 2.0)
    intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]], dtype=torch.float32)

    camera_type = CameraType.PERSPECTIVE

    camera = Cameras(
        fx=intrinsics_matrix[0, 0],
        fy=intrinsics_matrix[1, 1],
        cx=pp_w,
        cy=pp_h,
        camera_type=camera_type,
        camera_to_worlds=camera_state.c2w.to(torch.float32),
        times=torch.tensor([0.0], dtype=torch.float32),
    )
    return camera


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


def parse_object(
    obj: Any,
    type_check,
    tree_stub: str,
) -> List[Tuple[str, Any]]:
    """
    obj: the object to parse
    type_check: recursively adds instances of this type to the output
    tree_stub: the path down the object tree to this object

    Returns:
        a list of (path/to/object, obj), which represents the path down the object tree
        along with the object itself
    """

    def add(ret: List[Tuple[str, Any]], ts: str, v: Any):
        """
        helper that adds to ret, and if v exists already keeps the tree stub with
        the shortest path
        """
        for i, (t, o) in enumerate(ret):
            if o == v:
                if len(t.split("/")) > len(ts.split("/")):
                    ret[i] = (ts, v)
                return
        ret.append((ts, v))

    if not hasattr(obj, "__dict__"):
        return []
    ret = []
    # get a list of the properties of the object, sorted by whether things are instances of type_check
    obj_props = [(k, getattr(obj, k)) for k in dir(obj)]
    for k, v in obj_props:
        if k[0] == "_":
            continue
        new_tree_stub = f"{tree_stub}/{k}"
        if isinstance(v, type_check):
            add(ret, new_tree_stub, v)
        elif isinstance(v, nn.Module):
            if v is obj:
                # some nn.Modules might contain infinite references, e.g. consider foo = nn.Module(), foo.bar = foo
                # to stop infinite recursion, we skip such attributes
                continue
            lower_rets = parse_object(v, type_check, new_tree_stub)
            # check that the values aren't already in the tree
            for ts, o in lower_rets:
                add(ret, ts, o)
    return ret
