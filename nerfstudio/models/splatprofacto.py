# ruff: noqa: E741
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

import json
import math
from dataclasses import dataclass, field
from random import randint
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.gaussianpro_utils import check_geometric_consistency, depth_propagation, load_pairs_relation, read_propagted_depth
from nerfstudio.models.splatfacto import RGB2SH, SplatfactoModel, SplatfactoModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from PIL import Image

from copy import deepcopy
from simple_knn._C import distCUDA2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


@dataclass
class SplatProfactoModelConfig(SplatfactoModelConfig):
    """SplatProfacto  Model Config, nerfstudio's implementation of Gaussian Splatting Pro Model"""

    _target: Type = field(default_factory=lambda: SplatProfactoModel)


    """
    Some of the setting needed for splatfacto pro
    """
    propagated_iteration_begin: int = 10
    propagated_iteration_after: int = 1000
    propagation_interval: int = 50
    flatten_loss: bool = False
    sparse_loss: bool  = False
    normal_loss: bool = False
    depth_loss: bool = False
    return_depth: bool = False
    return_normal: bool = False
    return_opacity: bool = False
    lambda_l1_normal: float = 0.01
    lambda_cos_normal: float = 0.01
    lambda_sparse: float = 0.001
    lambda_flatten: float = 100.0
    intervals: List[int] = field(default_factory=lambda: [-2, -1, 1, 2])
    depth_max: float = 20
    patch_size: int = 20
    depth_error_max_threshold: float = 1.0
    depth_error_min_threshold: float = 1.0
    pair_path: str = ""


class SplatProfactoModel(SplatfactoModel):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatProfactoModelConfig
    rescale_cameras: bool = True
    normals: Dict[int, torch.tensor] = {}
    
    def propagate_gaussians(self, optimizers: Optimizers, pipeline: Pipeline, step: int):
        assert self.step == step, "Step shall be the same"
        if self.config.depth_loss and self.step > self.config.propagated_iteration_begin and \
           self.step < self.config.propagated_iteration_after and (self.step % self.config.propagation_interval == 0):
            # reset rescale cameras
            self.rescale_cameras = False

            # get all cameras
            cameras: Cameras = pipeline.datamanager.train_dataset.cameras.to(self.device)

            # Pick a random Camera
            assert len(cameras.shape) == 1, "Assumes single batch dimension"
            randidx = pipeline.datamanager.get_next_train_idx()

            # begin with empty list
            src_idxs = []
            
            # set the neighboring frames
            if self.config.depth_loss:
                if len(self.config.intervals) <= 0:
                    src_idxs = load_pairs_relation(self.config.pair_path)[randidx]
                else:
                    src_idxs = [randidx+itv for itv in self.config.intervals if ((itv + randidx > 0) and (itv + randidx < cameras.shape[0]))]

            # create the all idx
            all_idxs = src_idxs.copy()
            all_idxs.append(randidx)

            # get the downscale and reset the camera 
            camera_downscale = self._get_downscale_factor()

            # get all the cameras
            propagation_cameras: Dict[int, Cameras] = {}

            # loop and get all cameras
            for idx in all_idxs:
                propagation_cameras[idx] = cameras[idx: idx + 1]
                propagation_cameras[idx].rescale_output_resolution(1/camera_downscale)

            render_pkg = self.get_outputs(propagation_cameras[randidx])
            projected_depth = render_pkg["render_depth"]
            
            # ground truth image
            data = deepcopy(pipeline.datamanager.cached_train[randidx])

            # get ground truth image
            gt_image = self.composite_with_background(self.get_gt_img(data["image"].to(self.device)), 
                                                        render_pkg["background"]) 

            sky_mask = None
            # check if sky mask is provided
            if "mask" in data:
                sky_mask = self._downscale_if_required(data["mask"]).to(self.device)           

            projected_depths = []
            gt_images = []
            # ground truth images
            for src_idx in src_idxs:
                # get current camera
                cur_camera = propagation_cameras[src_idx]

                # calculate outputs
                cur_render_pkg = self.get_outputs(cur_camera)
                projected_depths.append(cur_render_pkg["render_depth"])

                # get the data
                cur_data = deepcopy(pipeline.datamanager.cached_train[src_idx])
            
                # get ground truth image
                gt_images.append(self.composite_with_background(self.get_gt_img(cur_data["image"].to(self.device)), 
                                                                cur_render_pkg["background"]))

            # get the propagated depth
            depth_propagation(propagation_cameras=propagation_cameras,
                              projected_depth=projected_depth, 
                              ref_idx=randidx,
                              src_idxs=src_idxs, 
                              depth_max=self.config.depth_max, 
                              patch_size=self.config.patch_size, 
                              ref_img=gt_image, 
                              gt_images=gt_images,)
            propagated_depth, cost, normal = read_propagted_depth('./cache/propagated_depth')

            cost = torch.tensor(cost).to(projected_depth.device)
            normal = torch.tensor(normal).to(projected_depth.device)

            #transform normal to camera coordinate
            R_w2c = propagation_cameras[randidx].Rs.T.clone().detach().cuda().to(torch.float32).squeeze()

            # R_w2c[:, 1:] *= -1
            normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).permute(1,0).view(propagation_cameras[randidx].height, 
                                                                                  propagation_cameras[randidx].width, 3)             
            
            propagated_depth = torch.tensor(propagated_depth).to(projected_depth.device)
            valid_mask = propagated_depth != 300

            # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
            render_depth = render_pkg['render_depth']
            abs_rel_error = torch.abs(propagated_depth - render_depth) / propagated_depth
            abs_rel_error_threshold = self.config.depth_error_max_threshold - (self.config.depth_error_max_threshold - self.config.depth_error_min_threshold) * \
            (self.step - self.config.propagated_iteration_begin) / (self.config.propagated_iteration_after - self.config.propagated_iteration_begin)

            # color error
            render_color = render_pkg['rgb']
            color_error = torch.abs(render_color - gt_image)
            color_error = color_error.mean(dim=0).squeeze()

            #for waymo, quantile 0.6; for free dataset, quantile 0.4
            error_mask = (abs_rel_error > abs_rel_error_threshold)
            
            # calculate the geometric consistency
            ref_K = propagation_cameras[randidx].get_intrinsics_matrices().to(self.device)
            ref_pose = propagation_cameras[randidx].world_to_cameras.squeeze().transpose(0, 1).inverse()

            geometric_counts = None
            for idx, src_idx in enumerate(src_idxs):
                #c2w
                cur_camera = propagation_cameras[src_idx]

                src_pose = cur_camera.world_to_cameras.squeeze().transpose(0, 1).inverse()
                src_K = cur_camera.get_intrinsics_matrices().to(self.device)
                
                #get the src_depth first
                depth_propagation(propagation_cameras=propagation_cameras, 
                                  ref_idx=src_idx,
                                  src_idxs=src_idxs, 
                                  projected_depth=torch.zeros_like(projected_depths[idx]).to(self.device),  
                                  depth_max=self.config.depth_max, 
                                  patch_size=self.config.patch_size, 
                                  ref_img=gt_images[idx], 
                                  gt_images=gt_images,)
                src_depth, cost, src_normal = read_propagted_depth('./cache/propagated_depth')
                src_depth = torch.tensor(src_depth).cuda()
                mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = \
                    check_geometric_consistency(depth_ref=propagated_depth.unsqueeze(0).float(), 
                                                intrinsics_ref=ref_K.float(), 
                                                extrinsics_ref=ref_pose.unsqueeze(0).float(), 
                                                depth_src=src_depth.unsqueeze(0).float(), 
                                                intrinsics_src=src_K.float(), 
                                                extrinsics_src=src_pose.unsqueeze(0).float(), 
                                                thre1=2, thre2=0.01)
                if geometric_counts is None:
                    geometric_counts = mask.to(torch.uint8)
                else:
                    geometric_counts += mask.to(torch.uint8)
                    
            cost = geometric_counts.squeeze()
            cost_mask = cost >= 2

            #set -10 as nan              
            normal[~(cost_mask.unsqueeze(2).repeat(1, 1, 3))] = -10
            self.normals[randidx] = normal
                
            propagated_mask = valid_mask & error_mask & cost_mask
            if sky_mask is not None:
                propagated_mask = propagated_mask & sky_mask

            if propagated_mask.sum() > 100:
                self._densify_from_depth_propagation(propagation_cameras[randidx],
                                                     propagated_depth, 
                                                     propagated_mask.to(torch.bool), 
                                                     gt_image, 
                                                     optimizers) 

            # rescale back
            for idx in all_idxs:
                propagation_cameras[idx].rescale_output_resolution(camera_downscale)

            # reenable rescaling
            self.rescale_cameras = True
        
    def _densify_from_depth_propagation(self, 
                                        camera: Cameras, 
                                        propagated_depth: torch.Tensor, 
                                        filter_mask: torch.Tensor, 
                                        gt_image: torch.Tensor, 
                                        optimizers: Optimizers,):
        # inverse project pixels into 3D scenes
        K = camera.get_intrinsics_matrices().squeeze().to(self.device).float()
        cam2world = camera.world_to_cameras.squeeze().T.transpose(0, 1).inverse().float()

        # Get the shape of the depth image
        height, width = propagated_depth.shape
        # Create a grid of 2D pixel coordinates
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        # Stack the 2D and depth coordinates to create 3D homogeneous coordinates
        coordinates = torch.stack([x.to(propagated_depth.device), y.to(propagated_depth.device), torch.ones_like(propagated_depth)], dim=-1)
        # Reshape the coordinates to (height * width, 3)
        coordinates = coordinates.view(-1, 3).to(K.device).to(torch.float32)
        # Reproject the 2D coordinates to 3D coordinates
        coordinates_3D = (K.inverse() @ coordinates.T).T.squeeze()

        # Multiply by dim=0
        coordinates_3D *= propagated_depth.view(-1, 1)

        # convert to the world coordinate
        world_coordinates_3D = (cam2world[:3, :3] @ coordinates_3D.T).T + cam2world[:3, 3]

        #mask the points below the confidence threshold
        #downsample the pixels; 1/4
        world_coordinates_3D = world_coordinates_3D.view(height, width, 3)
        world_coordinates_3D_downsampled = world_coordinates_3D[::8, ::8]
        filter_mask_downsampled = filter_mask[::8, ::8]
        gt_image_downsampled = gt_image[::8, ::8]

        world_coordinates_3D_downsampled = world_coordinates_3D_downsampled[filter_mask_downsampled]
        color_downsampled = gt_image_downsampled[filter_mask_downsampled]

        # initialize gaussians
        fused_point_cloud = world_coordinates_3D_downsampled
        fused_color = RGB2SH(color_downsampled)
        features = torch.zeros((fused_color.shape[0], 3, (self.config.sh_degree + 1) ** 2)).to(fused_color.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        original_point_cloud = self.means
        # initialize the scale from the mode, if using the distance to calculate, there are outliers, if using the whole gaussians, it is memory consuming
        # quantile_scale = torch.quantile(self.get_scaling, 0.5, dim=0)
        # scales = self.scaling_inverse_activation(quantile_scale.unsqueeze(0).repeat(fused_point_cloud.shape[0], 1))
        fused_shape = fused_point_cloud.shape[0]
        all_point_cloud = torch.concat([fused_point_cloud, original_point_cloud], dim=0)
        all_dist2 = torch.clamp_min(distCUDA2(all_point_cloud), 0.0000001)
        dist2 = all_dist2[:fused_shape]        
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(1.0 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        new_xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = torch.nn.Parameter(features[:,:,0].contiguous().requires_grad_(True))
        new_features_rest = torch.nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = torch.nn.Parameter(scales.requires_grad_(True))
        new_rotation = torch.nn.Parameter(rots.requires_grad_(True))
        new_opacity = torch.nn.Parameter(opacities.requires_grad_(True))

        #update gaussians
        self._densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, optimizers)

    def _cat_tensors_to_optimizer(self, 
                                  optimizers: Optimizers, 
                                  tensors_dict: Dict[str, torch.nn.Parameter]):
        """adds the parameters to the optimizer"""
        param_groups = self.get_gaussian_param_groups()

        # add zeros to the start
        # for each param group
        for group, new_param in param_groups.items():
            # get the optimizer
            optimizer = optimizers.optimizers[group]

            # calculate the parameters
            param = optimizer.param_groups[0]["params"][0]

            # get the parameter state
            param_state = optimizer.state[param]

            # prepare the tensor
            extension_tensor = tensors_dict[group]

            # update exponential average
            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.cat(
                    [param_state["exp_avg"], 
                     torch.zeros_like(extension_tensor)], 
                     dim=0)
                param_state["exp_avg_sq"] = torch.cat(
                    [param_state["exp_avg_sq"], 
                     torch.zeros_like(extension_tensor)], 
                     dim=0)

            # delete old
            del optimizer.state[param]
            optimizer.state[new_param[0]] = param_state
            optimizer.param_groups[0]["params"] = new_param
            del param

    def _densification_postfix(self, 
                               new_xyz: torch.nn.Parameter, 
                               new_features_dc: torch.nn.Parameter, 
                               new_features_rest: torch.nn.Parameter, 
                               new_opacity: torch.nn.Parameter, 
                               new_scaling: torch.nn.Parameter, 
                               new_rotation: torch.nn.Parameter, 
                               optimizers: Optimizers,):
        # create the new gaussians
        tensor_dict = {"means": new_xyz,
                       "features_dc": new_features_dc,
                       "features_rest": new_features_rest,
                       "opacities": new_opacity,
                       "scales" : new_scaling,
                       "quats" : new_rotation}

        # update param groups first
        for name, param in self.gauss_params.items():
            # prepare the tensor
            extension_tensor = tensor_dict[name]

            # add zeros to the end for the newly introduced gaussians
            self.gauss_params[name] = torch.nn.Parameter(
                torch.cat([param.detach(), extension_tensor], dim=0)
            )

        # dup in all optim
        self._cat_tensors_to_optimizer(optimizers, tensor_dict,)

        # just append zeros for the new ones
        self.xys_grad_norm = torch.cat([self.xys_grad_norm, 
                                        torch.zeros(new_xyz.shape[0]).to(self.device)], 
                                        dim=0)
        self.vis_counts = torch.cat([self.vis_counts, 
                                     torch.zeros(new_xyz.shape[0]).to(self.device)],
                                     dim=0)
        self.max_2Dsize = torch.cat([self.max_2Dsize, 
                                     torch.zeros(new_xyz.shape[0]).to(self.device)], 
                                     dim=0)

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))

        # add propagation before train iteration
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], 
                                    self.propagate_gaussians,
                                    args=[training_callback_attributes.optimizers,
                                          training_callback_attributes.pipeline],))

        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # add masks and add check both
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # opacity mask
        if self.step < self.config.propagated_iteration_begin and self.config.depth_loss:
            opacity_mask = outputs['accumulation'] > 0.999
            opacity_mask = opacity_mask.repeat(1, 1, 3)
        else:
            opacity_mask = outputs['accumulation'] > 0.0
            opacity_mask = opacity_mask.repeat(1, 1, 3)

        Ll1 = torch.abs(gt_img[opacity_mask] - pred_img[opacity_mask]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss

        # flatten loss
        if self.config.flatten_loss:
            scales = self.scales
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss += self.config.lambda_flatten * flatten_loss

        # opacity loss
        if self.config.sparse_loss:
            opacity = self.opacities
            opacity = opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
            sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.radii > 0].mean()
            loss += self.config.lambda_sparse * sparse_loss

        # normal loss
        if self.config.normal_loss:
            rendered_normal = outputs['render_normal']
            if outputs['cam_idx'] in self.normals:
                normal_gt = self.normals[outputs['cam_idx']].cuda()
                filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
                loss += self.config.lambda_l1_normal * l1_normal + self.config.lambda_cos_normal * cos_normal

        return {
            "main_loss": loss,
            "scale_reg": scale_reg,
        }
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        return_dict = {}
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None
        
        # rescale cameras if enabled
        if self.rescale_cameras:
            camera_downscale = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_downscale)

        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        rgb, alpha = rasterize_gaussians(  # type: ignore
            xys=self.xys,
            depths=depths,
            radii=self.radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,  # type: ignore
            colors=rgbs,
            opacity=opacities,
            img_height=H,
            img_width=W,
            block_width=BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        return_dict.update({'rgb': rgb, 'accumulation': alpha, "background": background})

        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                xys=self.xys,
                depths=depths,
                radii=self.radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,  # type: ignore
                colors=depths[:, None].repeat(1, 3),
                opacity=opacities,
                img_height=H,
                img_width=W,
                block_width=BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
            return_dict.update({"depth": depth_im})

        if self.config.depth_loss:
            projvect1 = camera.world_to_cameras.squeeze().T[:,2][:3].detach()
            projvect2 = camera.world_to_cameras.squeeze().T[:,2][-1].detach()
            means3D_depth = (means_crop * projvect1.unsqueeze(0)).sum(dim=-1,keepdim=True) + projvect2
            means3D_depth = means3D_depth.repeat(1,3)
            render_depth = rasterize_gaussians(
                xys=self.xys,
                depths=depths,
                radii=self.radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,  # type: ignore
                colors=means3D_depth,
                opacity=opacities,
                img_height=H,
                img_width=W,
                block_width=BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),)
            render_depth = render_depth.mean(dim=2) 
            return_dict.update({'render_depth': render_depth})
    
        if self.config.return_normal:
            quats = self.quats / self.quats.norm(dim=-1, keepdim=True)  # normalize them first
            rotations_mat = quat_to_rotmat(quats)  # how these scales are rotated
            scales = self.scales
            min_scales = torch.argmin(scales, dim=1)
            indices = torch.arange(min_scales.shape[0])
            normal = rotations_mat[indices, :, min_scales]

            # get the camera center
            camera_center = camera.world_to_cameras.squeeze().T.inverse()[3, :3]

            # convert normal direction to the camera; calculate the normal in the camera coordinate
            view_dir = means_crop - camera_center
            normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]

            R_w2c = R.T.clone().detach().cuda().to(torch.float32)
            normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)

            render_normal = rasterize_gaussians(  # type: ignore
                xys=self.xys,
                depths=depths,
                radii=self.radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,  # type: ignore
                colors=normal,
                opacity=opacities,
                img_height=H,
                img_width=W,
                block_width=BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )  # type: ignore
    
            render_normal = torch.nn.functional.normalize(render_normal, dim = 2)                                                                                                                                                   
            return_dict.update({'render_normal': render_normal, 
                                'cam_idx': camera.metadata["cam_idx"] if camera.metadata is not None else -1})

        if self.rescale_cameras:
            # rescale the camera back to original dimensions before returning
            camera.rescale_output_resolution(camera_downscale)
        
        # add camera norm as well
        norm = None if camera.metadata is None else camera.metadata.get("directions_norm")
        return_dict.update({"normal": norm})
        return return_dict
    