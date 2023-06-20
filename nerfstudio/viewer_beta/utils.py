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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model


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
    intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]]).float()

    camera_type = CameraType.PERSPECTIVE

    camera = Cameras(
        fx=intrinsics_matrix[0, 0],
        fy=intrinsics_matrix[1, 1],
        cx=pp_w,
        cy=pp_h,
        camera_type=camera_type,
        camera_to_worlds=camera_state.c2w,
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
