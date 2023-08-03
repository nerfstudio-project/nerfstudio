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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union, Dict, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils import poses as pose_utils


@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    trans_l2_penalty: float = 1e-3
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        else:
            assert_never(self.config.mode)

    def forward(
        self,
        indices: Int[Tensor, "num_cameras"],
    ) -> Float[Tensor, "num_cameras 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            if self.non_trainable_camera_indices.device != self.pose_adjustment.device:
                self.non_trainable_camera_indices = self.non_trainable_camera_indices.to(self.pose_adjustment.device)
            outputs[0][self.non_trainable_camera_indices] = torch.eye(4, device=self.pose_adjustment.device)[:3, :4]

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)

    def apply_to_raybundle(self, raybundle: RayBundle) -> RayBundle:
        """Apply the pose correction to the raybundle"""
        assert raybundle.camera_indices is not None
        if self.config.mode != "off":
            correction_matrices = self(raybundle.camera_indices.squeeze())  # type: ignore
            raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
            raybundle.directions = torch.bmm(correction_matrices[:, :3, :3], raybundle.directions[..., None]).squeeze()
        return raybundle

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            loss_dict["camera_opt_regularizer"] = (
                self.pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.trans_l2_penalty
                + self.pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
            )

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            metrics_dict["camera_opt_translation"] = self.pose_adjustment[:, :3].norm()
            metrics_dict["camera_opt_rotation"] = self.pose_adjustment[:, 3:].norm()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0


@dataclass
class DeblurCameraOptimizerConfig(CameraOptimizerConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: DeblurCameraOptimizer)
    """The target class to be instantiated."""

    num_samples: int = 7
    """The number of samples to use for deblurring."""

    blur_l2_penalty: float = 1e-4
    """The L2 penalty for the blur parameters."""


class DeblurCameraOptimizer(CameraOptimizer):
    config: DeblurCameraOptimizerConfig

    def __init__(self, config: DeblurCameraOptimizerConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.blur_adjustment = torch.nn.Parameter(torch.zeros((self.num_cameras, 6), device=self.device))

    def sample_blur_correction(self, indices: Int[Tensor, "num_cameras"]):
        """
        Samples the blur parameters and returns transformation matrices for the origins and directions
        """
        times = (
            torch.linspace(0, 1, self.config.num_samples, device=indices.device)
            .unsqueeze(1)
            .repeat(indices.shape[0] // self.config.num_samples, 1)
        )
        # add a small random jittter to each sample time to avoid aliasing
        # w = 1 / (self.config.num_samples - 1)
        # times += torch.rand_like(times) * w - w / 2
        blur_adjs = self.blur_adjustment[indices, :] * times
        # normal seems to work worse
        # blur_adjs = self.blur_adjustment[indices, :] * torch.normal(
        #     torch.zeros((indices.shape[0], 1), device=indices.device),  # mean
        #     torch.ones((indices.shape[0], 1), device=indices.device),  # std
        # )
        return exp_map_SO3xR3(blur_adjs)

    def apply_to_raybundle(self, ray_bundle: RayBundle) -> RayBundle:
        assert ray_bundle.camera_indices is not None
        if self.blur_adjustment.device != ray_bundle.origins.device:
            self.blur_adjustment = self.blur_adjustment.to(ray_bundle.origins.device)

        optimized_bundle = super().apply_to_raybundle(ray_bundle)
        if self.config.num_samples == 1:
            return optimized_bundle

        # duplicate optimized_bundle num_samples times and stack
        def repeatfn(x):
            return x.repeat_interleave(self.config.num_samples, 0)

        ray_bundle = optimized_bundle._apply_fn_to_fields(repeatfn)
        camera_ids = ray_bundle.camera_indices.squeeze()
        # Multiply in the camera optimization deltas to the rays
        blur_deltas = self.sample_blur_correction(camera_ids)
        ray_bundle.origins = ray_bundle.origins + blur_deltas[:, :, 3]
        ray_bundle.directions = torch.bmm(
            blur_deltas[:, :, :3], ray_bundle.directions.view(*ray_bundle.directions.shape, 1)
        )[..., 0]
        return ray_bundle

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        super().get_loss_dict(loss_dict)
        if self.config.mode != "off":
            # add a loss penalizing norm of blur adjustment
            loss_dict["blur_regularizer"] = self.blur_adjustment.norm(dim=-1).mean() * self.config.blur_l2_penalty
