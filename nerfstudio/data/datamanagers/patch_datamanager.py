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

from __future__ import annotations

#from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Type

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PatchPixelSampler

@dataclass
class PatchDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: PatchDataManager)
    """Target class to instantiate."""

class PatchDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: InputDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PatchPixelSampler(*args, **kwargs,patch_size=16)
