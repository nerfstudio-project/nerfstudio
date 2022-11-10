# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Defines an image that can be batched with the default nerfstudio collate fn, even if the images
aren't of the same height and width.
"""

from typing import List

import torch


# pylint: disable=too-few-public-methods
class BasicImages:
    """This is a very primitive struct for holding images, especially for when these images
    are of different heights / widths.

    The purpose of this is to have a special struct wrapping around a list so that the
    nerfstudio_collate fn and other parts of the code recognise this as a struct to leave alone
    instead of reshaping or concatenating into a single tensor (since this will likely be used
    for cases where we have images of different sizes and shapes).

    This only has one batch dimension and will likely be replaced down the line with some
    TensorDataclass alternative that supports arbitrary batches.
    """

    def __init__(self, images: List):
        assert isinstance(images, List)
        assert not images or isinstance(
            images[0], torch.Tensor
        ), f"Input should be a list of tensors, not {type(images[0]) if isinstance(images, List) else type(images)}"
        self.images = images

    def to(self, device):
        """Move the images to the given device."""
        assert isinstance(device, torch.device)
        return BasicImages([image.to(device) for image in self.images])
