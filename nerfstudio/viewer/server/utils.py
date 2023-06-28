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

"""Generic utility functions
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from typing_extensions import assert_never

from nerfstudio.viewer.viser.messages import CameraMessage


def get_chunks(
    lst: Union[List[float], Tuple[float, ...]], num_chunks: Optional[int] = None, size_of_chunk: Optional[int] = None
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


def get_intrinsics_matrix_and_camera_to_world_h(
    camera_message: CameraMessage, image_height: int, image_width: Optional[Union[int, float]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the camera intrinsics matrix and the camera to world homogeneous matrix.

    Args:
        camera_object: a Camera object.
        image_size: the size of the image (height, width)
    """
    # intrinsics
    fov = camera_message.fov
    aspect = camera_message.aspect
    if image_width is None:
        image_width = aspect * image_height
    pp_w = image_width / 2.0
    pp_h = image_height / 2.0
    if camera_message.camera_type in ("perspective", "fisheye"):
        focal_length = three_js_perspective_camera_focal_length(fov, image_height)
        intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]]).float()
    elif camera_message.camera_type == "equirectangular":
        render_aspect = camera_message.render_aspect
        if aspect < render_aspect:
            intrinsics_matrix = torch.tensor(
                [[pp_w, 0, pp_w], [0, image_width / render_aspect, pp_h], [0, 0, 1]]
            ).float()
        else:
            intrinsics_matrix = torch.tensor(
                [[image_height * render_aspect / 2, 0, pp_w], [0, pp_h * 2, pp_h], [0, 0, 1]]
            ).float()
    else:
        assert_never(camera_message.camera_type)

    # extrinsics
    camera_to_world_h = torch.tensor(get_chunks(camera_message.matrix, size_of_chunk=4)).T.float()
    camera_to_world_h = torch.stack(
        [
            camera_to_world_h[0, :],
            camera_to_world_h[2, :],
            camera_to_world_h[1, :],
            camera_to_world_h[3, :],
        ],
        dim=0,
    )

    return intrinsics_matrix, camera_to_world_h
