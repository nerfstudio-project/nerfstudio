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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import diff_rast
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss, distortion_loss, interlevel_loss, orientation_loss,
    pred_normal_loss, scale_gradients_by_distance_squared)
from nerfstudio.model_components.ray_samplers import (ProposalNetworkSampler,
                                                      UniformSampler)
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer,
                                                   NormalsRenderer,
                                                   RGBRenderer)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, gaussian_utils
from nerfstudio.utils import poses as pose_utils


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""
    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    initial_ptc_size = 100000
    max_ptc_size = 100000
    render_accumulation = False
    use_aabb = True
    initial_gaussian_scales = -3
    prune_density_after = 500
    prune_density_every = 100
    prune_density_until = 15000
    one_up_sh_every = 1000
    lambda_ssim = 0.2

class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def populate_modules(self):
        # TODO (jake-austin): clean this up, this is transplanted code across all the implementation functions
        self.active_sh_degree = 0
        self.max_sh_degree = 3

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.max_points = self.config.max_ptc_size


        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = gaussian_utils.build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = gaussian_utils.strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = gaussian_utils.inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        num_pts = self.config.initial_ptc_size
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        self.spatial_lr_scale = 5

        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
        fused_color = gaussian_utils.RGB2SH(torch.tensor(np.asarray(gaussian_utils.SH2RGB(shs))).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.ones((fused_point_cloud.shape[0], 3), device="cuda") * self.config.initial_gaussian_scales
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = gaussian_utils.inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.percent_dense = 0.01
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        # l = [
        #     {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
        #     {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        #     {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        #     {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        #     {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
        #     {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        # ]

        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self.xyz_scheduler_args = gaussian_utils.get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)

        self.device_indicator_param = nn.Parameter(torch.empty(0))

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        # TODO (jake-austin): remove and implement this
        return []
        
        callbacks = []

        def oneupSHdegree(self):
            if self.active_sh_degree < self.max_sh_degree:
                self.active_sh_degree += 1
        def wrapper_pruning_densification(step):
            if step <= self.config.prune_density_after and step > self.config.prune_density_until:
                return
            self.pruning_densification(
                training_callback_attributes.optimizers,
                training_callback_attributes.grad_scaler
            )
        def wrapper_reset_opacity(step):
            if step > self.config.prune_density_until:
                return
            self.reset_opacity(
                training_callback_attributes.optimizers,
                training_callback_attributes.grad_scaler
            )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=self.config.one_up_sh_every,
                func=oneupSHdegree,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=self.config.prune_density_every,
                func=wrapper_pruning_densification,
            )
        )

        return callbacks

    def pruning_densification(self, optimizers, grad_scaler):
        assert False

    def reset_opacity(self, optimizers, grad_scaler):
        opacities_new = gaussian_utils.inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.update_optimizer_tensors({"opacity": opacities_new}, "replace")
        self._opacity = optimizable_tensors["opacity"]

    def update_optimizer_tensors(self, data: dict, operation: Literal["cat", "prune", "replace"]):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in data:
                if operation == "cat":
                    # TODO (jake-austin): implement this, consider splitting this into 3 functions?
                    assert False
                elif operation == "prune":
                    assert False
                elif operation == "replace":
                    assert False
                else:
                    raise ValueError(f"Unknown operation {operation}")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        return {
            "xyz": [self._xyz],
            "f_dc": [self._features_dc],
            "f_rest": [self._features_rest],
            "opacity": [self._opacity],
            "scaling": [self._scaling],
            "rotation": [self._rotation],
        }

    def get_outputs(self, ray_bundle: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        TODO (jake-austin): use the new homebrew nerfstudio gaussian rasterization code instead

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        ray_bundle = ray_bundle.reshape(())
        assert ray_bundle.shape == ()
        R = ray_bundle.camera_to_worlds[..., :3, :3].squeeze()
        R[:,0] = -R[:,0]
        T = (R.T @ ray_bundle.camera_to_worlds[..., :3, 3:4]).squeeze()
        fovx = gaussian_utils.focal2fov(ray_bundle.fx, ray_bundle.width)
        fovy = gaussian_utils.focal2fov(ray_bundle.fy, ray_bundle.height)

        world_view_transform = gaussian_utils.getWorld2View2(R, T, torch.tensor([0.0, 0.0, 0.0]).to(self.device), 1.0).transpose(0, 1).to(self.device)
        projection_matrix = gaussian_utils.getProjectionMatrix(znear=.01, zfar=100, fovX=fovx, fovY=fovy).transpose(0,1).to(self.device)
        full_projection_matrix = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        camera_center = world_view_transform.inverse()[3, :3]
        data = {
            "xyz": self.get_xyz,
            "features": self.get_features,
            "opacity": self.get_opacity,
            "scaling": self.get_scaling,
            "rotation": self.get_rotation,
            "active_sh_degree": self.active_sh_degree,
            "max_sh_degree": self.max_sh_degree,
            "FoVx": fovx,
            "FoVy": fovy,
            "image_height": ray_bundle.width,
            "image_width": ray_bundle.width,
            "world_view_transform": world_view_transform,
            "full_proj_transform": full_projection_matrix,
            "camera_center": camera_center,
        }

        image = gaussian_utils.render_from_dict(
            data,
            bg_color=torch.ones(3).to(self.device),
        )["render"].permute(1, 2, 0) #HWC

        # TODO (jake-austin): need to clip the images since there are color overflow issues

        return {"rgb": image}

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        Ll1 = gaussian_utils.l1_loss(batch['image'], outputs['rgb'])
        loss = (1.0 - self.config.lambda_ssim) * Ll1 + self.config.lambda_ssim * (1.0 - gaussian_utils.ssim(batch['image'], outputs['rgb']))
        return {"main_loss": loss}

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: Cameras) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        return self.get_outputs(camera_ray_bundle)

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
