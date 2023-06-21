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

"""
Code for sampling pixels.
"""

import random
from typing import Dict, Optional, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.utils import profiler


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.kwargs = kwargs
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        if isinstance(mask, torch.Tensor):
            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
            indices = nonzero_indices[chosen_indices]
        else:
            indices = torch.floor(
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def collate_image_dataset_batch(
        self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False, indices: torch.Tensor = None
    ):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        if indices is None:
            indices = self.collate_image_dataset_indices(batch, num_rays_per_batch)
        need_interp = torch.is_floating_point(indices)

        c, y, x = torch.unbind(indices, dim=-1)
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        c = c.long()
        if need_interp:
            collated_batch = {
                key: _multiple_bilinear_sample(value, c, y, x)
                for key, value in batch.items()
                if key != "image_idx" and value is not None
            }
        else:
            collated_batch = {
                key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
            }

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(
        self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False, indices: torch.Tensor = None
    ):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_images = []
        if indices is None:
            indices = self.collate_image_dataset_indices_list(batch, num_rays_per_batch)
        need_interp = torch.is_floating_point(indices)

        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            start_ray = num_rays_in_batch * i

            idx = indices[start_ray : start_ray + num_rays_in_batch]
            if need_interp:
                all_images.append(_multiple_bilinear_sample(batch["image"], i, idx[:, 1], idx[:, 2]))
            else:
                all_images.append(batch["image"][i, idx[:, 1], idx[:, 2]])

        c, y, x = torch.unbind(indices, dim=-1)
        c = c.long()
        if need_interp:
            collated_batch = {
                key: _multiple_bilinear_sample(value, c, y, x)
                for key, value in batch.items()
                if key not in ["image_idx", "image", "mask"] and value is not None
            }
        else:
            collated_batch = {
                key: value[c, y, x]
                for key, value in batch.items()
                if key not in ["image_idx", "image", "mask"] and value is not None
            }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_indices(self, batch: Dict, num_rays_per_batch: int):
        """
        Does the same as collate_image_dataset_batch, except it only produces indices.

        Warning: camera indices are based on ordering in batch. Use batch["image_idx"][indices[:, 0]]
        to get corrected indices (fully equivalent to batch['indices']).

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            indices = self.sample_method(
                num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
            )
        else:
            indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        return indices

    def collate_image_dataset_indices_list(self, batch: Dict, num_rays_per_batch: int):
        """
        Does the same as collate_image_dataset_indices, except it will operate over a list of images / masks inside
        a list.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
        """
        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        masks = batch.get("mask", None)

        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            image_height, image_width, _ = batch["image"][i].shape

            start_ray = num_rays_in_batch * i
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - start_ray

            idx = self.sample_method(
                num_rays_in_batch, 1, image_height, image_width, mask=masks[i] if masks else None, device=device
            )
            idx[:, 0] = i
            all_indices.append(idx)

        indices = torch.cat(all_indices, dim=0)

        return indices

    @profiler.time_function
    def sample(self, image_batch: Dict, indices: torch.Tensor = None):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image, indices=indices
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image, indices=indices
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch

    def sample_indices(self, image_batch: Dict):
        """Sample an image batch and return the indices of a pixel batch.
        Equivalent to batch["indices"] where batch is produced by sample()

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            # I don't think this copy is necessary?
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            idx = self.collate_image_dataset_indices_list(image_batch, self.num_rays_per_batch)
        elif isinstance(image_batch["image"], torch.Tensor):
            idx = self.collate_image_dataset_indices(image_batch, self.num_rays_per_batch)
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return idx


class EquirectangularPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor):
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
        patch_size: side length of patch. This must be consistent in the method
        config in order for samples to be reshaped into patches correctly.
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.patch_size = kwargs["patch_size"]
        num_rays = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)
        super().__init__(num_rays, keep_full_image, **kwargs)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)

    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor):
            # Note: if there is a mask, sampling reduces back to uniform sampling
            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            sub_bs = batch_size // (self.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.patch_size, image_width - self.patch_size],
                device=device,
            )

            indices = indices.view(sub_bs, 1, 1, 3).broadcast_to(sub_bs, self.patch_size, self.patch_size, 3).clone()

            yys, xxs = torch.meshgrid(
                torch.arange(self.patch_size, device=device), torch.arange(self.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices


def _multiple_bilinear_sample(im, c, y, x):
    y_max = im.shape[1] - 1
    x_max = im.shape[2] - 1
    c = c.long()
    y_floor = y.long()
    y_ceil = torch.clamp(y_floor + 1, max=y_max)
    x_floor = x.long()
    x_ceil = torch.clamp(x_floor + 1, max=x_max)
    corners = torch.stack(
        [im[c, y_floor, x_floor], im[c, y_ceil, x_floor], im[c, y_floor, x_ceil], im[c, y_ceil, x_ceil]], dim=1
    )
    remain_x = x - x_floor
    remain_y = y - y_floor
    remain_comp_x = 1 - remain_x
    remain_comp_y = 1 - remain_y
    multipliers = (
        torch.stack(
            [
                remain_comp_x * remain_comp_y,
                remain_x * remain_comp_y,
                remain_comp_x * remain_y,
                remain_x * remain_y,
            ],
            dim=1,
        )
        .reshape(-1, 4, 1)
        .to(im.device)
    )
    return torch.sum(corners * multipliers, dim=1)
