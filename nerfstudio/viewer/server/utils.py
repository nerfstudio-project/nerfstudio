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

"""Generic utility functions
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def get_chunks(
    lst: List[float], num_chunks: Optional[int] = None, size_of_chunk: Optional[int] = None
) -> List[List[float]]:
    """Returns list of n elements, constaining a sublist.

    Args:
        lst: List to be chunked up
        num_chunks: number of chunks to split list into
        size_of_chunk: size of each chunk
    """
    if num_chunks:
        assert not size_of_chunk
        size = len(lst) // num_chunks
    if size_of_chunk:
        assert not num_chunks
        size = size_of_chunk
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
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


def get_intrinsics_matrix_and_camera_to_world_h(
    camera_object: Dict[str, Any], image_height: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the camera intrinsics matrix and the camera to world homogeneous matrix.

    Args:
        camera_object: a Camera object.
        image_size: the size of the image (height, width)
    """
    # intrinsics
    fov = camera_object["fov"]
    aspect = camera_object["aspect"]
    image_width = aspect * image_height
    pp_w = image_width / 2.0
    pp_h = image_height / 2.0
    focal_length = three_js_perspective_camera_focal_length(fov, image_height)
    intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]]).float()

    # extrinsics
    camera_to_world_h = torch.tensor(get_chunks(camera_object["matrix"], size_of_chunk=4)).T.float()
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
