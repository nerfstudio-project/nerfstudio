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
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.engine.optimizers import Optimizers
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
    initial_ptc_size: int = 100000
    max_ptc_size: int = None
    render_accumulation: bool = False
    use_aabb: bool = True
    initial_gaussian_scales: int = -3
    prune_density_after: int = 500
    prune_density_every: int = 100
    prune_density_until: int = 15000
    opacity_reset_interval: int = 3000
    one_up_sh_every: int = 1000
    lambda_ssim: float = 0.2
    use_diff_rast: bool = False # TODO (jake-austin): remove

class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    TODO (jake-austin): Figure out how to print out on the training log in terminal the number of splats

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
        callbacks = []

        # def oneupSHdegree(step):
        #     if self.active_sh_degree < self.max_sh_degree:
        #         self.active_sh_degree += 1

        def wrapper_add_densification_stats(step):
            if step < self.config.prune_density_until:
                self.add_densification_stats(self.densification_stats_cache[0], self.densification_stats_cache[1])

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=wrapper_add_densification_stats,
            )
        )

        def wrapper_pruning_densification(step):
            if step >= self.config.prune_density_after and step < self.config.prune_density_until:
                return
            size_threshold = None#20 if iteration > opt.opacity_reset_interval else None
            self.optimizer = training_callback_attributes.optimizers
            self.densify_and_prune(0.0002, 0.005, 2.6, None)

        def wrapper_reset_opacity(step):
            if step < self.config.prune_density_until and step > 0:
                return
            self.reset_opacity()

        # callbacks.append(
        #     TrainingCallback(
        #         where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
        #         update_every_num_iters=self.config.one_up_sh_every,
        #         func=oneupSHdegree,
        #     )
        # )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=self.config.prune_density_every,
                func=wrapper_pruning_densification,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=self.config.opacity_reset_interval,
                func=wrapper_reset_opacity,
            )
        )

        return callbacks

    # def pruning_densification(self, optimizers: Optimizers, grad_scaler: GradScaler):

    #     prune_mask = (self.get_opacity < 0.005).squeeze()
    #     # if max_screen_size:
    #     #     big_points_vs = self.max_radii2D > max_screen_size
    #     #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    #     #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points(prune_mask)

    #     mask = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

    #     valid_points_mask = ~mask
    #     for group_name in self.optimizers.optimizers:
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #             stored_state["exp_avg"] = stored_state["exp_avg"][mask]
    #             stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]

    #     self._xyz = optimizable_tensors["xyz"]
    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]

    #     self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

    #     self.denom = self.denom[valid_points_mask]
    #     self.max_radii2D = self.max_radii2D[valid_points_mask]


    # def reset_opacity(self, optimizers, grad_scaler):
    #     opacities_new = gaussian_utils.inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
    #     optimizable_tensors = self.update_optimizer_tensors({"opacity": opacities_new}, "replace")
    #     self._opacity = optimizable_tensors["opacity"]

    # def update_optimizer_tensors(self, data: dict, operation: Literal["cat", "prune", "replace"]):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         if group["name"] in data:
    #             if operation == "cat":
    #                 # TODO (jake-austin): implement this, consider splitting this into 3 functions?

    #             elif operation == "prune":
                    
    #             elif operation == "replace":
    #                 assert False
    #             else:
    #                 raise ValueError(f"Unknown operation {operation}")

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

        print(f"Num Points: {self._xyz.shape[0]}")


        camera = ray_bundle.reshape(())
        assert camera.shape == ()
        R = camera.camera_to_worlds[..., :3, :3].squeeze()
        R[:,0] = -R[:,0]
        T = (R.T @ camera.camera_to_worlds[..., :3, 3:4]).squeeze()
        fovx = gaussian_utils.focal2fov(camera.fx, camera.width)
        fovy = gaussian_utils.focal2fov(camera.fy, camera.height)

        world_view_transform = gaussian_utils.getWorld2View2(R, T, torch.tensor([0.0, 0.0, 0.0]).to(self.device), 1.0).transpose(0, 1).to(self.device)
        projection_matrix = gaussian_utils.getProjectionMatrix(znear=.01, zfar=100, fovX=fovx, fovY=fovy).transpose(0,1).to(self.device)
        full_projection_matrix = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        if not self.config.use_diff_rast:

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
                "image_height": camera.height,
                "image_width": camera.width,
                "world_view_transform": world_view_transform,
                "full_proj_transform": full_projection_matrix,
                "camera_center": camera_center,
            }

            outs = gaussian_utils.render_from_dict(
                data,
                bg_color=torch.tensor([1,1,1]).to(torch.float32).to(self.device),
            ) #HWC
            image = outs["render"].permute(1, 2, 0)

            if self.train():
                self.densification_stats_cache = [outs["viewspace_points"], outs["visibility_filter"]]

            return {"rgb": image}
        
        else:

            c2w = camera.camera_to_worlds
            c2w = pose_utils.to4x4(c2w)
            w2c = torch.linalg.inv(c2w)
            proj = self._get_proj_matrix(
                w2c=w2c,
                fx=camera.fx.item(),
                fy=camera.fy.item(),
                width=camera.width.item(),
                height=camera.height.item(),
            )

            BLOCK_X = 16
            BLOCK_Y = 16
            W = camera.width.item()
            H = camera.height.item()
            tile_bounds = (W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1

            glob_scale = 1
            xys, depths, radii, conics, num_tiles_hit = diff_rast.project_gaussians(
                means3d=self.get_xyz,
                scales=self.get_scaling,
                glob_scale=glob_scale,
                quats=self.get_rotation,
                viewmat=w2c,
                projmat=proj,
                img_height=camera.width.item(),
                img_width=camera.height.item(),
                fx=camera.fx.item(),
                fy=camera.fy.item(),
                tile_bounds=tile_bounds,
            )

            rgb = diff_rast.rasterize(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                self.get_features,
                self.get_opacity.squeeze(),
                camera.height.item(),
                camera.width.item(),
            )

            return {"rgb": rgb}

    def _get_proj_matrix(
        self, w2c, fx: float, fy: float, width: int, height: int, znear=0.01, zfar=100
    ):
        top = 0.5 * height / fy * znear
        bottom = -top
        right = 0.5 * width / fx * znear
        left = -right

        P = torch.zeros(4, 4).to(self.device)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        return P @ w2c

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {"Num Gaussians": self._xyz.shape[0]}

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # plt.imsave("test.png", np.concatenate([outputs["rgb"].detach().cpu().numpy(), batch["image"].detach().cpu().numpy()], axis=1))

        Ll1 = gaussian_utils.l1_loss(batch['image'], outputs['rgb'])
        loss = (1.0 - self.config.lambda_ssim) * Ll1 + self.config.lambda_ssim * (1.0 - gaussian_utils.ssim(batch['image'], outputs['rgb']))
        return {"main_loss": loss}

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: Cameras) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        outs = self.get_outputs(camera_ray_bundle.to(self.device))
        outs["rgb"] = torch.clamp(outs["rgb"], 0.0, 1.0)
        return outs

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


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group_k, group_v in self.optimizer.optimizers.items():
            if group_k == name:
                stored_state = self.optimizer.optimizers[group_k].state.get(group_v.param_groups[0]['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.optimizers[group_k].state[group_v.param_groups[0]['params'][0]]
                group_v.param_groups[0]["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.optimizers[group_k].state[group_v.param_groups[0]['params'][0]] = stored_state

                optimizable_tensors[group_k] = group_v.param_groups[0]["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group_k, group_v in self.optimizer.optimizers.items():
            stored_state = self.optimizer.optimizers[group_k].state.get(group_v.param_groups[0]['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.optimizers[group_k].state[group_v.param_groups[0]['params'][0]]
                group_v.param_groups[0]["params"][0] = nn.Parameter((group_v.param_groups[0]["params"][0][mask].requires_grad_(True)))
                self.optimizer.optimizers[group_k].state[group_v.param_groups[0]['params'][0]] = stored_state

                optimizable_tensors[group_k] = group_v.param_groups[0]["params"][0]
            else:
                group_v.param_groups[0]["params"][0] = nn.Parameter(group_v.param_groups[0]["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group_k] = group_v.param_groups[0]["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group_k, group_v in self.optimizer.optimizers.items():
            assert len(group_v.param_groups[0]["params"]) == 1
            extension_tensor = tensors_dict[group_k]
            stored_state = self.optimizer.optimizers[group_k].state.get(group_v.param_groups[0]['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.optimizers[group_k].state[group_v.param_groups[0]['params'][0]]
                group_v.param_groups[0]["params"][0] = nn.Parameter(torch.cat((group_v.param_groups[0]["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.optimizers[group_k].state[group_v.param_groups[0]['params'][0]] = stored_state

                optimizable_tensors[group_k] = group_v.param_groups[0]["params"][0]
            else:
                group_v.param_groups[0]["params"][0] = nn.Parameter(torch.cat((group_v.param_groups[0]["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group_k] = group_v.param_groups[0]["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, num_selected=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if num_selected is not None and selected_pts_mask.sum() > 0:
            num_selected = int(min(num_selected, selected_pts_mask.sum() - 1))
            choices = np.arange(0, len(selected_pts_mask))[selected_pts_mask.cpu().numpy()]
            chosen = np.random.choice(choices, num_selected, replace=False)
            selected_pts_indices = torch.from_numpy(chosen).cuda()
            selected_pts_mask[...] = False
            selected_pts_mask[selected_pts_indices] = True

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = gaussian_utils.build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, num_selected=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if num_selected is not None and selected_pts_mask.sum() > 0:
            num_selected = int(min(num_selected, selected_pts_mask.sum() - 1))
            choices = np.arange(0, len(selected_pts_mask))[selected_pts_mask.cpu().numpy()]
            chosen = np.random.choice(choices, num_selected, replace=False)
            selected_pts_indices = torch.from_numpy(chosen).cuda()
            selected_pts_mask[...] = False
            selected_pts_mask[selected_pts_indices] = True
        
        # if selected_pts_mask.sum() > 0:
        #     breakpoint()
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        # if selected_pts_mask.sum() > 0:
        #     breakpoint()

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        num_added = self.max_points - self.get_xyz.shape[0] if self.max_points is not None else None
        if num_added is None or num_added > 0:
            self.densify_and_clone(grads, max_grad, extent, num_selected=num_added//2 if num_added is not None else None)
            self.densify_and_split(grads, max_grad, extent, num_selected=num_added//2 if num_added is not None else None)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def reset_opacity(self):
        opacities_new = gaussian_utils.inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
