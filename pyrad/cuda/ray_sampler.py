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

from typing import Optional, Tuple

import torch
from torchtyping import TensorType

import pyrad.cuda_v2 as pyrad_cuda
from pyrad.cameras.rays import Frustums, RayBundle, RaySamples
from pyrad.fields.occupancy_fields.occupancy_grid import DensityGrid
from pyrad.graphs.modules.ray_sampler import Sampler


class NGPSpacedSampler(Sampler):
    """Sampler that matches Instant-NGP paper."""

    def __init__(
        self,
        num_samples: int,
        density_field: Optional[DensityGrid] = None,
        density_threshold: float = 1e-4,
    ) -> None:
        super().__init__(num_samples)
        self.num_samples = num_samples
        self.density_field = density_field
        self.density_threshold = density_threshold

    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError("For NGP we fused ray samples and occupancy check together. Please call forward() directly.")

    # pylint: disable=arguments-differ
    def forward(
        self,
        ray_bundle: RayBundle,
        aabb: TensorType[2, 3],
        num_samples: Optional[int] = None,
    ) -> Tuple[RaySamples, TensorType]:
        """Generate ray samples"""
        num_samples = num_samples or self.num_samples

        aabb = aabb.flatten()
        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        t_min, t_max = pyrad_cuda.ray_aabb_intersect(rays_o, rays_d, aabb)

        if self.training:
            # TODO(ruilongli): * 16 is for original impl.
            # needs to deal with loss because not all rays will
            # be processed.
            max_samples_per_batch = len(rays_o) * 16
        else:
            max_samples_per_batch = len(rays_o) * num_samples

        packed_info, positions, dirs, deltas, ts = pyrad_cuda.raymarching_train(
            rays_o,
            rays_d,
            t_min,
            t_max,
            self.density_field.center,
            self.density_field.num_cascades,
            self.density_field.resolution,
            self.density_field.density_bitfield,
            max_samples_per_batch,
            num_samples,
            0.0,
        )
        total_samples = max(packed_info[:, -1].sum(), 1)
        positions = positions[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]

        zeros = torch.zeros_like(positions[:, :1])
        # TODO(ruilongli): check why this fails at cascades=2
        # print(rays_o.shape, max_samples_per_batch, num_samples)
        # torch.cuda.synchronize()

        ray_samples = RaySamples(
            frustums=Frustums(origins=positions, directions=dirs, starts=zeros, ends=zeros, pixel_area=zeros),
            deltas=deltas,
            ts=ts,
        )
        return ray_samples, packed_info
