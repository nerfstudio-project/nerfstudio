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
3D Gaussian Splatting model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

from torch.nn import Parameter
import torch
from torch import Tensor

from jaxtyping import Float
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.cameras.cameras import Cameras

from nerfstudio.utils import poses as pose_utils

from nerfstudio.cameras.points import Gaussians3D


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """3D Gaussian Splatting Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)


class GaussianSplattingModel(Model):
    """3D Gaussian Splatting Model

    Args:
        config: configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.cameras: Cameras = self.kwargs["cameras"]
        self.gaussians: Gaussians3D = self.kwargs["gaussians"]  # initialized 3D gaussians from SfM

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

        return {}

    def get_outputs(self, idx: int, image: Tensor):
        """Projects 3D gaussians to 2D image"""
        camera: Cameras = self.cameras[idx]  # current camera object to which gaussians are rendered onto
        c2w = camera.camera_to_worlds
        c2w = pose_utils.to4x4(c2w)
        w2c = torch.linalg.inv(c2w)

        gaussians_2d_image = self.project_gaussians_torch(w2c=w2c)
        print(gaussians_2d_image.covariance.shape)
        # TODO: render 2d gaussians to get rgb image

        rgb = None
        outputs = {
            "rgb": rgb,
        }
        return outputs

    def project_gaussians_torch(self, w2c: Float[Tensor, "4 4"]):
        """PyTorch implementation of 3D gaussian projection.

        Args:
            w2c: 4x4 homogenous world to camera matrix of current render view
        """

        # TODO: culling of bad gaussians before projection

        # 1. points from world coord to camera coord i.e. view space
        positions = torch.cat(
            (self.gaussians.positions, torch.ones(self.gaussians.positions.shape[0], 1)), dim=1
        ).float()  # to homogenous
        points_camera = torch.matmul(positions, w2c.T)[:, :3]

        # 2. Linear jacobian of view matrix
        jacobian = self.jacobian_torch(points_camera)

        JW = torch.matmul(jacobian, w2c[:3, :3].unsqueeze(dim=0))
        JWC = torch.bmm(JW, self.gaussians.get_gaussian_cov())
        gaussian_2d_cov = torch.bmm(JWC, JW.permute(0, 2, 1))[:, :2, :2]  # projected gaussians

        # 3. camera to image frame i.e. projection space
        points_image = torch.zeros_like(points_camera)
        points_image[:, 0] = points_camera[:, 0] / points_camera[:, 2]
        points_image[:, 1] = points_camera[:, 1] / points_camera[:, 2]
        points_image[:, 2] = points_camera.norm(dim=-1)

        gaussian_2d_image = Gaussians3D(
            positions=points_image,
            rgbs=self.gaussians.rgbs,  # .sigmoid()
            opacity=self.gaussians.opacity,  # .sigmoid()
            covariance=gaussian_2d_cov,
        )

        return gaussian_2d_image

    def jacobian_torch(self, positions: Float[Tensor, "batch 3"]):
        """PyTorch implementation of affine approximation of the projective transformation."""

        # The following models the steps outlined by equations 29
        # and 31 in "EWA Splatting" (Zwicker et al., 2002).
        # Additionally considers aspect / scaling of viewport.
        # Transposes used to account for row-/column-major conventions.
        # computeCov2D function from original repo.

        # focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        # 0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        # 0, 0, 0);

        # Is this even correct? Should check since original code has focal length and no normalization
        normalizer = 1.0 / (positions[:, 0] ** 2 + positions[:, 1] ** 2 + positions[:, 2] ** 2).sqrt()
        _res = [
            1 / positions[:, 2],
            torch.zeros_like(positions[:, 0]),
            -positions[:, 0] / (positions[:, 2] ** 2),
            torch.zeros_like(positions[:, 0]),
            1 / positions[:, 2],
            -positions[:, 1] / (positions[:, 2] ** 2),
            normalizer * positions[:, 0],
            normalizer * positions[:, 1],
            normalizer * positions[:, 2],
        ]
        return torch.stack(_res, dim=-1).reshape(-1, 3, 3)

    def rasterize_gaussians(self, means3D, means2D, shs, colors, opacities, scales, rotations, cov3d, **kwargs):
        return None

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}

    def forward(self, idx: int, image: Tensor):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        return self.get_outputs(idx, image)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        callbacks = []

        return callbacks
