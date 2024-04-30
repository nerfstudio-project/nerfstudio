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

"""Pixel sampling utils such as eroding of valid masks that we sample from."""

import math
from typing import List

import torch
from jaxtyping import Float
from torch import Tensor


def dilate(tensor: Float[Tensor, "bs 1 H W"], kernel_size=3) -> Float[Tensor, "bs 1 H W"]:
    """Dilate a tensor with 0s and 1s. 0s will be be expanded based on the kernel size.

    Args:
        kernel_size: Size of the pooling region. Dilates/contracts 1 pixel if kernel_size is 3.
    """

    unique_vals = torch.unique(tensor)
    if any(val not in (0, 1) for val in unique_vals) or tensor.dtype != torch.float32:
        raise ValueError("Input tensor should contain only values 0 and 1, and should have dtype torch.float32.")

    return torch.nn.functional.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)


def erode(tensor: Float[Tensor, "bs 1 H W"], kernel_size=3) -> Float[Tensor, "bs 1 H W"]:
    """Erode a tensor with 0s and 1s. 1s will be expanded based on the kernel size.

    Args:
        kernel_size: Size of the pooling region. Erodes/expands 1 pixel if kernel_size is 3.
    """

    unique_vals = torch.unique(tensor)
    if any(val not in (0, 1) for val in unique_vals) or tensor.dtype != torch.float32:
        raise ValueError("Input tensor should contain only values 0 and 1, and should have dtype torch.float32.")

    x = 1 - dilate(1 - tensor, kernel_size=kernel_size)
    # set edge pixels to 0
    p = (kernel_size - 1) // 2
    x[:, :, :p, :] *= 0
    x[:, :, :, :p] *= 0
    x[:, :, -p:, :] *= 0
    x[:, :, :, -p:] *= 0
    return x


def erode_mask(tensor: Float[Tensor, "bs 1 H W"], pixel_radius: int = 1):
    """Erode a mask. Expands 1 values to nearby pixels with a max pooling operation.
    A pixel radius of 1 will also extend the 1s along the diagonal.

    Args:
        pixel_radius: The number of pixels away from valid pixels (1s) that we may sample.
    """
    kernel_size = 1 + 2 * pixel_radius
    return erode(tensor, kernel_size=kernel_size)


def divide_rays_per_image(num_rays_per_batch: int, num_images: int) -> List[int]:
    """Divide the batch of rays per image. Finds the optimal number of rays per image such that
    it's still divisible by 2 and sums to the total number of rays.

    Args:
        num_rays_per_batch: Number of rays in the batch.
        num_images: Number of images in the batch.

    Returns:
        num_rays_per_image: A list of the number of rays per image.
    """
    num_rays_per_image = num_rays_per_batch / num_images
    residual = num_rays_per_image % 2
    num_rays_per_image_under = int(num_rays_per_image - residual)
    num_rays_per_image_over = int(num_rays_per_image_under + 2)
    num_images_under = math.ceil(num_images * (1 - residual / 2))
    num_images_over = num_images - num_images_under
    num_rays_per_image = num_images_under * [num_rays_per_image_under] + num_images_over * [num_rays_per_image_over]
    num_rays_per_image[-1] += num_rays_per_batch - sum(num_rays_per_image)
    return num_rays_per_image
