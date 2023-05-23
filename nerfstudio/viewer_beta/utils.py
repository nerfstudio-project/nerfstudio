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
from typing import List, Optional, Union

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from nerfstudio.cameras.cameras import Cameras, CameraType


@dataclass
class CameraState:
    """A dataclass for storing the camera state."""

    fov: float
    """ The field of view of the camera. """
    aspect: float
    """ The aspect ratio of the image. """
    c2w: Float[torch.Tensor, "3 4"]
    """ The camera matrix. """


def get_chunks(
    lst: NDArray, num_chunks: Optional[int] = None, size_of_chunk: Optional[int] = None
) -> List[List[float]]:
    """Returns list of n elements, containing a sublist.

    Args:
        lst: List to be chunked up
        num_chunks: number of chunks to split list into
        size_of_chunk: size of each chunk
    """
    if num_chunks:
        assert not size_of_chunk
        size = len(lst) // num_chunks
    elif size_of_chunk:
        assert not num_chunks
        size = size_of_chunk
    else:
        assert False, "Either `num_chunks` or `size_of_chunk` must be set"

    chunks = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i : i + size])
    return chunks


def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


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
    focal_length = three_js_perspective_camera_focal_length(fov, image_height)
    intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]]).float()

    # camera_to_world_h = torch.stack(
    #     [
    #         camera_to_world_h[0, :],
    #         camera_to_world_h[2, :],
    #         camera_to_world_h[1, :],
    #         camera_to_world_h[3, :],
    #     ],
    #     dim=0,
    # )

    # camera_to_world = camera_to_world_h[:3, :]
    # camera_to_world = torch.stack(
    #     [
    #         camera_to_world[0, :],
    #         camera_to_world[2, :],
    #         camera_to_world[1, :],
    #     ],
    #     dim=0,
    # )

    camera_type = CameraType.PERSPECTIVE

    camera = Cameras(
        fx=intrinsics_matrix[0, 0],
        fy=intrinsics_matrix[1, 1],
        cx=intrinsics_matrix[0, 2],
        cy=intrinsics_matrix[1, 2],
        camera_type=camera_type,
        camera_to_worlds=camera_state.c2w,
        times=torch.tensor([0.0], dtype=torch.float32),
    )
    return camera
