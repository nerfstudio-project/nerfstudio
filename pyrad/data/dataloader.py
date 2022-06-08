"""
Data loader.
"""

from typing import Dict, List, Tuple

from torchtyping import TensorType
from pyrad.data.image_dataset import ImageDataset
from pyrad.data.image_sampler import ImageSampler
from pyrad.data.pixel_sampler import PixelSampler
from pyrad.cameras.cameras import get_camera
from pyrad.cameras.rays import RayBundle

import random
from pyrad.utils.misc import get_dict_to_torch


class TrainDataloader:
    def __init__(self, image_sampler: ImageSampler, pixel_sampler: PixelSampler):
        self.image_sampler = image_sampler
        self.pixel_sampler = pixel_sampler
        self.iter_image_sampler = iter(self.image_sampler)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> Tuple[TensorType["num_rays"], Dict]:
        self.count += 1
        image_batch = next(self.iter_image_sampler)
        batch = self.pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        return ray_indices, batch


class EvalDataloader:
    def __init__(
        self, image_dataset: ImageDataset, intrinsics, camera_to_world, num_rays_per_chunk: int, device="cpu", **kwargs
    ):
        super().__init__()
        self.image_dataset = image_dataset
        self.intrinsics = intrinsics
        self.camera_to_world = camera_to_world
        self.num_rays_per_chunk = num_rays_per_chunk
        self.device = device

    def get_data_from_image_idx(self, image_idx) -> Tuple[RayBundle, Dict]:
        intrinsics = self.intrinsics[image_idx].to(self.device)
        camera_to_world = self.camera_to_world[image_idx].to(self.device)
        camera = get_camera(intrinsics, camera_to_world, camera_index=image_idx)
        ray_bundle = camera.get_camera_ray_bundle(device=self.device)
        ray_bundle.num_rays_per_chunk = self.num_rays_per_chunk
        batch = self.image_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device)
        return ray_bundle, batch


class FixedIndicesEvalDataloader(EvalDataloader):
    def __init__(
        self,
        image_dataset: ImageDataset,
        intrinsics,
        camera_to_world,
        num_rays_per_chunk: int,
        image_indices: List[int],
        device="cpu",
        **kwargs
    ):
        super().__init__(image_dataset, intrinsics, camera_to_world, num_rays_per_chunk, device, **kwargs)
        self.image_indices = image_indices
        self.num_rays_per_chunk = num_rays_per_chunk
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            ray_bundle.num_rays_per_chunk = self.num_rays_per_chunk
            self.count += 1
            return ray_bundle, batch
        else:
            raise StopIteration


class RandIndicesEvalDataloader(EvalDataloader):
    def __init__(
        self, image_dataset: ImageDataset, intrinsics, camera_to_world, num_rays_per_chunk: int, device="cpu", **kwargs
    ):
        super().__init__(image_dataset, intrinsics, camera_to_world, num_rays_per_chunk, device, **kwargs)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < 1:
            image_indices = range(len(self.camera_to_world))
            image_idx = random.choice(image_indices)
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            self.count += 1
            return ray_bundle, batch
        else:
            raise StopIteration
