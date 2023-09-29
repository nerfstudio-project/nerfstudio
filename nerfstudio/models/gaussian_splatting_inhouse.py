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
import torch
from copy import deepcopy
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import gc
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.viewer_beta.viewer_elements import ViewerControl
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
import math

from diff_rast.rasterize import RasterizeGaussians
from diff_rast.project_gaussians import ProjectGaussians
from diff_rast.sh import SphericalHarmonics, num_sh_bases

def get_references_to_object(obj):
    return [ref for ref in gc.get_referrers(obj) if isinstance(ref, (list, dict, tuple)) or hasattr(ref, '__dict__')]

def random_quat_tensor(N, **kwargs):
    u = torch.rand(N, **kwargs)
    v = torch.rand(N, **kwargs)
    w = torch.rand(N, **kwargs)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )


def identity_quat(N, **kwargs):
    quat = torch.zeros(N, 4, **kwargs)
    quat[:, 0] = 1
    return quat

def projection_matrix(znear, zfar, fovx, fovy, **kwargs):
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        **kwargs
    )


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""
    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    warmup_length:int = 1500
    """period of steps where refinement is turned off"""
    refine_every:int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 500
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh:float = .01
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh:float = .5
    """threshold of scale for culling gaussians"""
    reset_alpha_every:int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = .0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = .01
    """below this size, gaussians are *duplicated*, otherwise split"""
    #set to 3 to effeectively turn off, it seems to make things worse? need to investigate
    sh_degree_interval: int = 3
    """every n intervals turn on another sh degree"""

class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    TODO (jake-austin): Figure out how to print out on the training log in terminal the number of splats

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig
    def __init__(self,*args,**kwargs):
        self.seed_pts = kwargs['seed_points']
        super().__init__(*args,**kwargs)
        self.vc = ViewerControl()

    def populate_modules(self):
        # TODO (jake-austin): clean this up, this is transplanted code across all the implementation functions
        self.means = torch.nn.Parameter(self.seed_pts[0])
        self.means_grad_norm = None
        init_scale = torch.log(torch.tensor(.01)).item()
        self.scales = torch.nn.Parameter(torch.full((self.num_points,3),init_scale))
        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        self.degree = 3
        dim_sh = num_sh_bases(self.degree)
        self.colors = torch.nn.Parameter(torch.rand(self.num_points, dim_sh, 3))
        self.opacities = torch.nn.Parameter(torch.zeros(self.num_points, 1))
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step=0
        
    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"
        
        param = optimizer.param_groups[0]['params'][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]
        
        # Modify the state directly without deleting and reassigning.
        param_state['exp_avg'] = param_state['exp_avg'][~deleted_mask.squeeze()]
        param_state['exp_avg_sq'] = param_state['exp_avg_sq'][~deleted_mask.squeeze()]
        
        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]['params'][0]
        del optimizer.param_groups[0]['params']
        optimizer.param_groups[0]['params'] = new_params
        optimizer.state[new_params[0]] = param_state


    def dup_in_optim(self,optimizer, dup_mask, new_params):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]['params'][0]
        param_state = optimizer.state[param]
        param_state['exp_avg'] = torch.cat([param_state['exp_avg'], torch.zeros_like(param_state['exp_avg'][dup_mask.squeeze()])],dim=0)
        param_state['exp_avg_sq'] = torch.cat([param_state['exp_avg_sq'], torch.zeros_like(param_state['exp_avg_sq'][dup_mask.squeeze()])],dim=0)
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]['params'] = new_params
        del param

    def after_train(self,step):
        with torch.no_grad():
            #keep track of a moving average of grad norms
            weight = 1/self.config.refine_every
            if self.means_grad_norm is None:
                self.means_grad_norm = weight*self.means.grad.detach().norm(dim=-1,keepdim=True)
            else:
                self.means_grad_norm = weight*self.means.grad.detach().norm(dim=-1,keepdim=True) + self.means_grad_norm
    
    def refinement_before(self, optimizers:Optimizers, step):
        print("Inside refinement before")
        if self.step > self.config.warmup_length:
            with torch.no_grad():
                # do all the refinement stuff here
                #first we cull gaussians
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_param_groups()
                for group,param in param_groups.items(): 
                    self.remove_from_optim(optimizers.optimizers[group],deleted_mask,param)
            

    def refinement_after(self, optimizers:Optimizers, step):
        if self.step > self.config.warmup_length:
            with torch.no_grad():
                #then we densify
                high_grads = (self.means_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                splits &= high_grads
                split_means,split_colors,split_opacities,split_scales,split_quats = self.split_gaussians(splits)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_means,dup_colors,dup_opacities,dup_scales,dup_quats = self.dup_gaussians(dups)
                self.means = Parameter(torch.cat([self.means.detach(),split_means, dup_means],dim=0))
                self.colors = Parameter(torch.cat([self.colors.detach(),split_colors, dup_colors],dim=0))
                self.opacities = Parameter(torch.cat([self.opacities.detach(),split_opacities, dup_opacities],dim=0))
                self.scales = Parameter(torch.cat([self.scales.detach(),split_scales, dup_scales],dim=0))
                self.quats = Parameter(torch.cat([self.quats.detach(),split_quats, dup_quats],dim=0))
                
                split_idcs = torch.where(splits)[0]
                dup_idcs = torch.where(dups)[0]
                add_idcs = torch.cat([split_idcs, dup_idcs], dim=0)
                
                param_groups = self.get_param_groups()
                for group,param in param_groups.items():
                    print(f"adding {len(add_idcs)} params to {group} optimizer")
                    self.dup_in_optim(optimizers.optimizers[group],add_idcs,param)

                if self.step // self.config.refine_every % self.config.reset_alpha_every == 0:
                    print("Resetting alpha")
                    reset_value = .01
                    self.opacities.data = torch.full_like(self.opacities.data,torch.logit(torch.tensor(reset_value)).item())
                    #reset the exp of optimizer
                    optim = optimizers.optimizers['opacity']
                    param = optim.param_groups[0]['params'][0]
                    param_state = optim.state[param]
                    param_state['exp_avg'] = torch.zeros_like(param_state['exp_avg'])
                    param_state['exp_avg_sq'] = torch.zeros_like(param_state['exp_avg_sq'])
                self.means_grad_norm = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        #cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        #cull huge ones
        culls = culls | (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
        self.means_grad_norm = self.means_grad_norm[~culls].detach()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.colors = Parameter(self.colors[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())

        print(f"Culled {n_bef - self.num_points} gaussians")
        return culls

    def split_gaussians(self, split_mask):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        print(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        #step 1, sample new means
        cov_mats = torch.eye(3,device=self.device)[None,...].repeat(n_splits,1,1)
        cov3ds = self.cov3d[split_mask]
        cov_mats[:,0,0]=cov3ds[:,0]
        cov_mats[:,0,1]=cov3ds[:,1]
        cov_mats[:,1,0]=cov3ds[:,1]
        cov_mats[:,0,2]=cov3ds[:,2]
        cov_mats[:,2,0]=cov3ds[:,2]
        cov_mats[:,1,1]=cov3ds[:,3]
        cov_mats[:,1,2]=cov3ds[:,4]
        cov_mats[:,2,1]=cov3ds[:,4]
        cov_mats[:,2,2]=cov3ds[:,5]
        centered_samples = torch.randn((n_splits,3),device=self.device)
        new_means = torch.bmm(cov_mats,centered_samples[...,None]).squeeze() + self.means[split_mask]
        # step 2, sample new colors
        new_colors = self.colors[split_mask]
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask]
        # step 4, sample new scales
        new_scales = torch.logit(torch.exp(self.scales[split_mask])/1.6)
        self.scales[split_mask] = torch.logit(torch.exp(self.scales[split_mask])/1.6)
        # step 5, sample new quats
        new_quats = self.quats[split_mask]
        return new_means,new_colors,new_opacities,new_scales,new_quats

    def dup_gaussians(self,dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"Would duplicate {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]
        dup_colors = self.colors[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        return dup_means,dup_colors,dup_opacities,dup_scales,dup_quats
    
    @property
    def num_points(self):
        return self.means.shape[0]
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],self.step_cb))
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                                    self.refinement_before,
                                    update_every_num_iters=self.config.refine_every,
                                    args=[training_callback_attributes.optimizers]
                                    ))
        # The order of these matters
        cbs.append(TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                                    self.after_train,))
        cbs.append(TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                                    self.refinement_after,
                                    update_every_num_iters=self.config.refine_every,
                                    args=[training_callback_attributes.optimizers]
                                    ))
        return cbs
    
    def step_cb(self,step):
        self.step = step

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        return {
            "xyz": [self.means],
            "color": [self.colors],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
        }

    def _get_downscale_factor(self):
        return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule),0)
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        TODO (jake-austin): use the new homebrew nerfstudio gaussian rasterization code instead

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        import viser.transforms as vtf
        import numpy as np
        if not isinstance(camera,Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] ==1, "Only one camera at a time"
        #dont mutate the input
        camera = deepcopy(camera)
        if self.training:
            d = self._get_downscale_factor()
            camera.rescale_output_resolution(1/d)

        #shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[..., :3, :3] # 1 x 3 x 3
        T = camera.camera_to_worlds[..., :3, 3:4] # 1 x 3 x 1
        R = vtf.SO3.from_matrix(R.cpu().squeeze().numpy())
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.from_numpy(R.as_matrix()[None,...]).to(self.device,torch.float32)
        #vickie pops off here
        viewmat = torch.cat([R,T],dim=2)
        #add a row of zeros and a 1 to the bottom of the viewmat
        viewmat = torch.cat([viewmat,torch.tensor([[[0,0,0,1]]],device=self.device)],dim=1)
        #invert it
        viewmat = torch.inverse(viewmat)
        #calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        projmat = projection_matrix(.0001,1000,fovx,fovy).to(self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        if self.training:
            background = torch.rand(3, device=self.device)
        else:
            background = torch.zeros(3, device=self.device)
        xys, depths, radii, conics, num_tiles_hit,self.cov3d = ProjectGaussians.apply(
            self.means,
            torch.exp(self.scales),
            1,
            self.quats,
            viewmat.squeeze()[:3,:],
            projmat.squeeze()@viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            H,
            W,
            tile_bounds
        )
        if self.degree > 0:
            viewdirs = self.means - camera.camera_to_worlds[..., :3, 3]  # (N, 3)
            n = min(self.step // self.config.sh_degree_interval,self.degree)
            n_bases = num_sh_bases(n)
            rgbs = SphericalHarmonics.apply(n, viewdirs, self.colors[:,:n_bases])  # (N, 3)
        else:
            rgbs = self.colors.squeeze()  # (N, 3)
        cx_delta = cx - W / 2
        cy_delta = cy - H / 2
        xys = xys.view(-1, 2) + torch.tensor([cx_delta, cy_delta], device=self.device)
        rgb = RasterizeGaussians.apply(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(rgbs),
            torch.sigmoid(self.opacities),
            H,
            W,
            background,
        )
        return {"rgb": rgb}


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
        d = self._get_downscale_factor()
        if d > 1:
            #use torchvision to resize
            import torchvision.transforms.functional as TF
            newsize = (batch['image'].shape[0]//d,batch['image'].shape[1]//d)
            gt_img = TF.resize(batch['image'].permute(2,0,1),newsize).permute(1,2,0)
        else:
            gt_img = batch['image']
        Ll1 = torch.nn.functional.l1_loss(gt_img, outputs['rgb'])
        # This simloss makes the results look weird, removing for now
        simloss = self.ssim(gt_img.permute(2,0,1)[None,...], outputs['rgb'].permute(2,0,1)[None,...])
        return {"main_loss": Ll1}

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

