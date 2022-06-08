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
Code for sampling pixels.
"""

import random

import torch


def collate_image_dataset_batch(batch, num_rays_per_batch, keep_full_image: bool = False):
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.
    """
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch
    if "mask" in batch:
        nonzero_indices = torch.nonzero(batch["mask"][..., 0], as_tuple=False)
        chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_per_batch)
        indices = nonzero_indices[chosen_indices]
    else:
        indices = torch.floor(
            torch.rand((num_rays_per_batch, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

    c, y, x = [i.flatten() for i in torch.split(indices, 1, dim=-1)]
    image = batch["image"][c, y, x]
    if "mask" in batch:
        mask = batch["mask"][c, y, x]
    if "semantics" in batch:
        semantics = batch["semantics"][c, y, x]
    assert image.shape == (num_rays_per_batch, 3), image.shape

    # Needed to correct the random indices to their actual camera idx locations.
    local_indices = indices.clone()
    indices[:, 0] = batch["image_idx"][c]
    collated_batch = {
        "local_indices": local_indices,  # local to the batch returned
        "indices": indices,  # with the abs camera indices
        "image": image,
    }
    if "mask" in batch:
        collated_batch["mask"] = mask
    if "semantics" in batch:
        collated_batch["semantics"] = semantics

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's."""

    def __init__(self, num_rays_per_batch, keep_full_image=False) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def sample(self, image_batch):
        """Sample an image batch and return a pixel batch."""
        pixel_batch = collate_image_dataset_batch(
            image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
        )
        return pixel_batch
