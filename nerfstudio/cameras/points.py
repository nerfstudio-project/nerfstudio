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
from typing import Union, Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor

from nerfstudio.utils.poses import quaterion_to_rotation

from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.cameras.camera_utils import _EPS

TORCH_DEVICE = Union[str, torch.device]


@dataclass
class Points3D:
    """Struct that holds 3D point information from SfM"""

    ids: Int[Tensor, "*batch 1"]
    xyzs: Float[Tensor, "*batch 3"]
    rgbs: Float[Tensor, "*batch 3"]
    errors: Float[Tensor, "*batch 1"]


@dataclass
class Gaussians3D:
    """3D Gaussian objects"""

    positions: Float[Tensor, "*batch 3"]
    rgbs: Float[Tensor, "*batch 3"]
    opacity: Float[Tensor, "*batch 1"]
    quat: Float[Tensor, "*batch 4"] = None
    scale: Float[Tensor, "*batch 3"] = None
    scale_activation: Literal["abs", "exp"] = "exp"
    covariance2D: Float[Tensor, "*batch 2 2"] = None

    def __len__(self) -> int:
        num_gaussians = torch.numel(self.positions) // self.positions.shape[-1]
        return num_gaussians

    def get_covariance_3D(self) -> Float[Tensor, "*batch 3 3"]:
        """Transforms a 3D rotation (quaternion) and scale to a 3D covariance matrix Σ = R@S@S.T@R.T

        TODO: speed this up using: https://github.com/graphdeco-inria/gaussian-splatting/blob/9d977c702bc8819bfec26992f541dc6162074267/scene/gaussian_model.py#L27
              or cuda implementation.
        """
        R = quaterion_to_rotation(self.quat)
        if self.scale_activation == "abs":
            _scale = self.scale.abs() + _EPS
        elif self.scale_activation == "exp":
            _scale = trunc_exp(self.scale)  # not sure if clipping ranges are best here
        S = torch.diag_embed(_scale)  # type: ignore
        RS = torch.bmm(R, S)
        RSSR = torch.bmm(RS, RS.permute(0, 2, 1))
        return RSSR

    @property
    def rotation(self):
        """Returns rotation matrices"""
        return quaterion_to_rotation(self.quat)
