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
Code for sampling pixels.
"""

import random
from typing import Dict

import torch


def collate_image_dataset_batch(batch: Dict, num_rays_per_batch: int, sample_all_pixels: bool = False):
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        sample_all_pixels: if True, will sample all pixels in the image
    """
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    if sample_all_pixels:
        collated_batch = {key: value for key, value in batch.items() if key != "image_idx" and value is not None}
        collated_batch["indices"] = torch.stack(
            torch.meshgrid(
                torch.arange(num_images, device=device),
                torch.arange(image_height, device=device),
                torch.arange(image_width, device=device),
            ),
            dim=-1,
        )
    else:
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

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None}

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

    return collated_batch


def collate_image_dataset_batch_list(batch: Dict, num_rays_per_batch: int, sample_all_pixels: bool = False):
    """
    Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
    a list.

    We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
    The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
    since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        sample_all_pixels: if True, will sample all the pixels in the image
    """

    device = batch["image"][0].device
    num_images = len(batch["image"])

    # only sample within the mask, if the mask is in the batch
    all_indices = []
    all_images = []

    if sample_all_pixels:
        # sample all the pixels
        raise NotImplementedError("sample_all_pixels is not implemented for collate_image_dataset_batch_list")
    else:
        # sample a subset of the pixels
        if "mask" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                nonzero_indices = torch.nonzero(batch["mask"][i][..., 0], as_tuple=False)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_in_batch)
                indices = nonzero_indices[chosen_indices]
                indices = torch.cat([torch.full((num_rays_in_batch, 1), i, device=device), indices], dim=-1)
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = torch.floor(
                    torch.rand((num_rays_in_batch, 3), device=device)
                    * torch.tensor([1, image_height, image_width], device=device)
                ).long()
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

    indices = torch.cat(all_indices, dim=0)

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key != "image_idx" and key != "image" and key != "mask" and value is not None
    }

    collated_batch["image"] = torch.cat(all_images, dim=0)

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    return collated_batch


class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
    """

    def __init__(self, num_rays_per_batch: int) -> None:
        self.num_rays_per_batch = num_rays_per_batch

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample(self, image_batch: Dict, sample_all_pixels: bool = False) -> Dict:
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
            sample_all_pixels: if True, will sample all pixels in the image
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictioary so we don't modify the original
            pixel_batch = collate_image_dataset_batch_list(image_batch, self.num_rays_per_batch, sample_all_pixels=sample_all_pixels)
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = collate_image_dataset_batch(image_batch, self.num_rays_per_batch, sample_all_pixels=sample_all_pixels)
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


def collate_image_dataset_batch_equirectangular(batch: Dict, num_rays_per_batch: int):
    """
    Operates on a batch of equirectangular images and samples pixels to use for
    generating rays. Rays will be generated uniformly on the sphere.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
    """
    # TODO(kevinddchen): make more DRY
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch
    if "mask" in batch:
        # TODO(kevinddchen): implement this
        raise NotImplementedError("Masking not implemented for equirectangular images.")

    # We sample theta uniformly in [0, 2*pi]
    # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
    # This is done by inverse transform sampling.
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/
    num_images_rand = torch.rand(num_rays_per_batch, device=device)
    phi_rand = torch.acos(1 - 2 * torch.rand(num_rays_per_batch, device=device)) / torch.pi
    theta_rand = torch.rand(num_rays_per_batch, device=device)
    indices = torch.floor(
        torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
        * torch.tensor([num_images, image_height, image_width], device=device)
    ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None}

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    return collated_batch


class EquirectangularPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
    """

    # overrides base method
    def sample(self, image_batch: Dict):

        pixel_batch = collate_image_dataset_batch_equirectangular(image_batch, self.num_rays_per_batch)
        return pixel_batch
