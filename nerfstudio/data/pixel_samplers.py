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
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.utils.pixel_sampling_utils import divide_rays_per_image, erode_mask


@dataclass
class PixelSamplerConfig(InstantiateConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""
    ignore_mask: bool = False
    """Whether to ignore the masks when sampling."""
    fisheye_crop_radius: Optional[float] = None
    """Set to the radius (in pixels) for fisheye cameras."""
    rejection_sample_mask: bool = True
    """Whether or not to use rejection sampling when sampling images with masks"""
    max_num_iterations: int = 100
    """If rejection sampling masks, the maximum number of times to sample"""


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PixelSamplerConfig

    def __init__(self, config: PixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        # Possibly override some values if they are present in the kwargs dictionary
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.config.is_equirectangular = self.kwargs.get("is_equirectangular", self.config.is_equirectangular)
        self.config.fisheye_crop_radius = self.kwargs.get("fisheye_crop_radius", self.config.fisheye_crop_radius)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

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
        indices = (
            torch.rand((batch_size, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            if self.config.rejection_sample_mask:
                num_valid = 0
                for _ in range(self.config.max_num_iterations):
                    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
                    chosen_indices_validity = mask[..., 0][c, y, x].bool()
                    num_valid = int(torch.sum(chosen_indices_validity).item())
                    if num_valid == batch_size:
                        break
                    else:
                        replacement_indices = (
                            torch.rand((batch_size - num_valid, 3), device=device)
                            * torch.tensor([num_images, image_height, image_width], device=device)
                        ).long()
                        indices[~chosen_indices_validity] = replacement_indices

                if num_valid != batch_size:
                    warnings.warn(
                        """
                        Masked sampling failed, mask is either empty or mostly empty.
                        Reverting behavior to non-rejection sampling. Consider setting
                        pipeline.datamanager.pixel-sampler.rejection-sample-mask to False
                        or increasing pipeline.datamanager.pixel-sampler.max-num-iterations
                        """
                    )
                    self.config.rejection_sample_mask = False
                    nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                    chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                    indices = nonzero_indices[chosen_indices]
            else:
                nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                indices = nonzero_indices[chosen_indices]

        return indices

    def sample_method_equirectangular(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
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

    def sample_method_fisheye(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # Rejection sampling.
            valid: Optional[torch.Tensor] = None
            indices = None
            while True:
                samples_needed = batch_size if valid is None else int(batch_size - torch.sum(valid).item())

                # Check if done!
                if samples_needed == 0:
                    break

                rand_samples = torch.rand((samples_needed, 2), device=device)
                # Convert random samples to radius and theta.
                radii = self.config.fisheye_crop_radius * torch.sqrt(rand_samples[:, 0])
                theta = 2.0 * torch.pi * rand_samples[:, 1]

                # Convert radius and theta to x and y.
                x = (radii * torch.cos(theta) + image_width // 2).long()
                y = (radii * torch.sin(theta) + image_height // 2).long()
                sampled_indices = torch.stack(
                    [torch.randint(0, num_images, size=(samples_needed,), device=device), y, x], dim=-1
                )

                # Update indices.
                if valid is None:
                    indices = sampled_indices
                    valid = (
                        (sampled_indices[:, 1] >= 0)
                        & (sampled_indices[:, 1] < image_height)
                        & (sampled_indices[:, 2] >= 0)
                        & (sampled_indices[:, 2] < image_width)
                    )
                else:
                    assert indices is not None
                    not_valid = ~valid
                    indices[not_valid, :] = sampled_indices
                    valid[not_valid] = (
                        (sampled_indices[:, 1] >= 0)
                        & (sampled_indices[:, 1] < image_height)
                        & (sampled_indices[:, 2] >= 0)
                        & (sampled_indices[:, 2] < image_width)
                    )
            assert indices is not None

        assert indices.shape == (batch_size, 3)
        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                indices = self.sample_method(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
        else:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            else:
                indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
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

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
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

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []
        all_depth_images = []

        assert num_rays_per_batch % 2 == 0, "num_rays_per_batch must be divisible by 2"
        num_rays_per_image = divide_rays_per_image(num_rays_per_batch, num_images)

        if "mask" in batch:
            for i, num_rays in enumerate(num_rays_per_image):
                image_height, image_width, _ = batch["image"][i].shape

                indices = self.sample_method(
                    num_rays, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                if "depth_image" in batch:
                    all_depth_images.append(batch["depth_image"][i][indices[:, 1], indices[:, 2]])

        else:
            for i, num_rays in enumerate(num_rays_per_image):
                image_height, image_width, _ = batch["image"][i].shape
                if self.config.is_equirectangular:
                    indices = self.sample_method_equirectangular(num_rays, 1, image_height, image_width, device=device)
                else:
                    indices = self.sample_method(num_rays, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                if "depth_image" in batch:
                    all_depth_images.append(batch["depth_image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key not in ("image_idx", "image", "mask", "depth_image") and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)
        if "depth_image" in batch:
            collated_batch["depth_image"] = torch.cat(all_depth_images, dim=0)

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


@dataclass
class PatchPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: PatchPixelSampler)
    """Target class to instantiate."""
    patch_size: int = 32
    """Side length of patch. This must be consistent in the method
    config in order for samples to be reshaped into patches correctly."""


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: PatchPixelSamplerConfig

    def __init__(self, config: PatchPixelSamplerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.config.patch_size = self.kwargs.get("patch_size", self.config.patch_size)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)

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
        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            sub_bs = batch_size // (self.config.patch_size**2)
            half_patch_size = int(self.config.patch_size / 2)
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=half_patch_size)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=sub_bs)
            indices = nonzero_indices[chosen_indices]

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys - half_patch_size
            indices[:, ..., 2] += xxs - half_patch_size

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)
        else:
            sub_bs = batch_size // (self.config.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
                device=device,
            )

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices


@dataclass
class PairPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PairPixelSampler."""

    _target: Type = field(default_factory=lambda: PairPixelSampler)
    """Target class to instantiate."""
    radius: int = 2
    """max distance between pairs of pixels."""


class PairPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples pair of pixels from 'image_batch's. Samples pairs of pixels from
        from the images randomly within a 'radius' distance apart. Useful for pair-based losses.

    Args:
        config: the PairPixelSamplerConfig used to instantiate class
    """

    def __init__(self, config: PairPixelSamplerConfig, **kwargs) -> None:
        self.config = config
        self.radius = self.config.radius
        super().__init__(self.config, **kwargs)
        self.rays_to_sample = self.config.num_rays_per_batch // 2

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: Optional[int],
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        rays_to_sample = self.rays_to_sample
        if batch_size is not None:
            assert (
                int(batch_size) % 2 == 0
            ), f"PairPixelSampler can only return batch sizes in multiples of two (got {batch_size})"
            rays_to_sample = batch_size // 2

        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=self.radius)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=rays_to_sample)
            indices = nonzero_indices[chosen_indices]
        else:
            s = (rays_to_sample, 1)
            ns = torch.randint(0, num_images, s, dtype=torch.long, device=device)
            hs = torch.randint(self.radius, image_height - self.radius, s, dtype=torch.long, device=device)
            ws = torch.randint(self.radius, image_width - self.radius, s, dtype=torch.long, device=device)
            indices = torch.concat((ns, hs, ws), dim=1)

        pair_indices = torch.hstack(
            (
                torch.zeros(rays_to_sample, 1, device=device, dtype=torch.long),
                torch.randint(-self.radius, self.radius, (rays_to_sample, 2), device=device, dtype=torch.long),
            )
        )
        pair_indices += indices
        indices = torch.hstack((indices, pair_indices)).view(rays_to_sample * 2, 3)
        return indices
