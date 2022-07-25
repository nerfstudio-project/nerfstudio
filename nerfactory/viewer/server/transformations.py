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

"""
Some helpful functions for creating geometric transformations.
"""

import math
from typing import List, Optional

import numpy as np


def get_unit_vector(
    data: List[float], axis: Optional[List[float]] = None, out: Optional[List[float]] = None
) -> Optional[np.ndarray]:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        data: direction in which we want to calculate the unit vector for
        axis: the axis along which to get the unit vector
        out: buffer in which we write the unit vector out to (if not returning)
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    return None


def get_translation_matrix(direction: List[float]) -> np.ndarray:
    """Return matrix to translate by direction vector.

    Args:
        direction: specifying direction of translation
    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M


def get_rotation_matrix(angle: float, direction: List[float], point: Optional[List[float]] = None) -> np.ndarray:
    """Return matrix to rotate about axis defined by point and direction.

    Args:
        angle: angle of rotation
        direction: specifying axis of rotation
        point: if specified, rotation not around origin
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = get_unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        [[0.0, -direction[2], direction[1]], [direction[2], 0.0, -direction[0]], [-direction[1], direction[0], 0.0]]
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M
