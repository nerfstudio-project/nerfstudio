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
Some point based datastructures.
"""
from dataclasses import dataclass
from typing import Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[str, torch.device]


@dataclass
class Points3D:
    """Struct that holds 3D point information from SfM"""

    ids: Int[Tensor, "*batch 1"]
    xyzs: Float[Tensor, "*batch 3"]
    rgbs: Float[Tensor, "*batch 3"]
    errors: Float[Tensor, "*batch 1"]


@dataclass
class Gaussians3D(TensorDataclass):
    """3D Gaussian objects"""

    positions: Float[Tensor, "*batch 3"]
    rgbs: Float[Tensor, "*batch 3"]
    opacity: Float[Tensor, "*batch 1"]
    quat: Float[Tensor, "*batch 4"]
    scale: Float[Tensor, "*batch 3"]

    def __len__(self) -> int:
        num_gaussians = torch.numel(self.positions) // self.positions.shape[-1]
        return num_gaussians
