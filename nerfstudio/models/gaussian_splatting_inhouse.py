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
from nerfstudio.cameras.rays import RayBundle
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from diff_rast._torch_impl import compute_sh_color
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.viewer_beta.viewer_elements import ViewerControl
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors


from diff_rast.rasterize import RasterizeGaussians
from diff_rast.project_gaussians import ProjectGaussians
from diff_rast.sh import SphericalHarmonics, num_sh_bases


def get_references_to_object(obj):
    return [ref for ref in gc.get_referrers(obj) if isinstance(ref, (list, dict, tuple)) or hasattr(ref, "__dict__")]


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
        **kwargs,
    )


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    warmup_length: int = 900
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 300
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.005
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    # set to 3 to effeectively turn off, it seems to make things worse? need to investigate
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    max_screen_size: int = 700
    """maximum screen size of a gaussian, in pixels"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    ssim_lambda: float = 0.2


class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def __init__(self, *args, **kwargs):
        if "seed_points" in kwargs:
            self.seed_pts = kwargs["seed_points"]
        else:
            self.seed_pts = None
        super().__init__(*args, **kwargs)
        self.vc = ViewerControl()

    def populate_modules(self):
        # TODO (jake-austin): clean this up, this is transplanted code across all the implementation functions
        if self.seed_pts is not None and not self.config.random_init:
            # randmeans = torch.rand((30000,3))*8-4
            # self.means = torch.nn.Parameter(torch.cat([self.seed_pts[0],randmeans],dim=0)) # (Location, Color)
            self.means = torch.nn.Parameter(self.seed_pts[0])  # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((100000, 3)) - 0.5) * 2)
        self.proj_means = torch.nn.Parameter(torch.zeros(self.num_points, 2))
        self.xys_grad_norm = None
        # init_scale = torch.log(torch.tensor(.01)).item()
        # self.scales = torch.nn.Parameter(torch.full((self.num_points,3),init_scale))
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))

        # self.scales = torch.nn.Parameter(torch.full((self.num_points,3),init_scale))
        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        self.degree = 3
        dim_sh = num_sh_bases(self.degree)

        if self.seed_pts is not None and not self.config.random_init:
            # fused_color = self.RGB2SH(self.seed_pts[1]/255)
            # shs = torch.zeros((fused_color.shape[0], dim_sh, 3))
            # shs[:, 0, :] = fused_color
            # shs[:, 1:, :] = 0.0
            # randcolors = torch.rand((30000,dim_sh,3))
            # self.colors = torch.nn.Parameter(torch.cat([shs,randcolors]))
            fused_color = self.RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], 3, dim_sh)).float().cuda()
            shs[:, :3, 0] = fused_color
            shs[:, 3:, 1:] = 0.0
            self.colors = torch.nn.Parameter(shs[:, :, 0:1])
            self.shs_rest = torch.nn.Parameter(shs[:, :, 1:])
        else:
            self.colors = torch.nn.Parameter(torch.rand(self.num_points, 1, 3))
            self.shs_rest = torch.nn.Parameter(torch.zeros((self.num_points, dim_sh - 1, 3)))

        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    @property
    def get_colors(self):
        color = self.colors
        shs_rest = self.shs_rest
        return torch.cat((color, shs_rest), dim=2)

    def load_state_dict(self, dict, **kwargs):
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.proj_means = torch.nn.Parameter(torch.zeros(newp, 2, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.colors = torch.nn.Parameter(torch.zeros(self.num_points, 1, 3, device=self.device))
        self.shs_rest = torch.nn.Parameter(
            torch.zeros(self.num_points, num_sh_bases(self.degree) - 1, 3, device=self.device)
        )
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask.flatten()]
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask.flatten()]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def dup_in_optim(self, optimizer, dup_mask, new_params):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        param_state["exp_avg"] = torch.cat(
            [param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"][dup_mask.flatten()])], dim=0
        )
        param_state["exp_avg_sq"] = torch.cat(
            [param_state["exp_avg_sq"], torch.zeros_like(param_state["exp_avg_sq"][dup_mask.flatten()])], dim=0
        )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def after_train(self, step):
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.proj_means.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii)
            newradii = self.radii[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(self.max_2Dsize[visible_mask], newradii)

    def refinement_after(self, optimizers: Optimizers, step):
        if self.step > self.config.warmup_length:
            with torch.no_grad():
                # then we densify
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                splits &= high_grads
                (
                    split_means,
                    split_proj,
                    split_colors,
                    split_shs,
                    split_opacities,
                    split_scales,
                    split_quats,
                ) = self.split_gaussians(splits)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_means, dup_proj, dup_colors, dup_shs, dup_opacities, dup_scales, dup_quats = self.dup_gaussians(
                    dups
                )
                self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                self.proj_means = Parameter(torch.cat([self.proj_means.detach(), split_proj, dup_proj], dim=0))
                self.colors = Parameter(torch.cat([self.colors.detach(), split_colors, dup_colors], dim=0))
                self.shs_rest = Parameter(torch.cat([self.shs_rest.detach(), split_shs, dup_shs], dim=0))
                self.opacities = Parameter(torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0))
                self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])], dim=0
                )

                split_idcs = torch.where(splits)[0]
                dup_idcs = torch.where(dups)[0]
                add_idcs = torch.cat([split_idcs, dup_idcs], dim=0)

                param_groups = self.get_param_groups()
                for group, param in param_groups.items():
                    self.dup_in_optim(optimizers.optimizers[group], add_idcs, param)
                # then cull
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_param_groups()
                for group, param in param_groups.items():
                    self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)

                if self.step // self.config.refine_every % self.config.reset_alpha_every == 0:
                    print("Resetting alpha")
                    reset_value = 0.01
                    self.opacities.data = torch.full_like(
                        self.opacities.data, torch.logit(torch.tensor(reset_value)).item()
                    )
                    # reset the exp of optimizer
                    optim = optimizers.optimizers["opacity"]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        # cull huge ones
        culls = culls | (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
        # cull big screen space
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            culls = culls | (self.max_2Dsize > self.config.max_screen_size).squeeze()
        self.means = Parameter(self.means[~culls].detach())
        self.proj_means = Parameter(self.proj_means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.colors = Parameter(self.colors[~culls].detach())
        self.shs_rest = Parameter(self.shs_rest[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())

        print(f"Culled {n_bef - self.num_points} gaussians")
        return culls

    def split_gaussians(self, split_mask):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        print(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        # step 1, sample new means
        cov_mats = torch.eye(3, device=self.device)[None, ...].repeat(n_splits, 1, 1)
        cov3ds = self.cov3d[split_mask]
        cov_mats[:, 0, 0] = cov3ds[:, 0]
        cov_mats[:, 0, 1] = cov3ds[:, 1]
        cov_mats[:, 1, 0] = cov3ds[:, 1]
        cov_mats[:, 0, 2] = cov3ds[:, 2]
        cov_mats[:, 2, 0] = cov3ds[:, 2]
        cov_mats[:, 1, 1] = cov3ds[:, 3]
        cov_mats[:, 1, 2] = cov3ds[:, 4]
        cov_mats[:, 2, 1] = cov3ds[:, 4]
        cov_mats[:, 2, 2] = cov3ds[:, 5]
        centered_samples = torch.randn((n_splits, 3), device=self.device)
        new_means = torch.bmm(cov_mats, centered_samples[..., None]).squeeze() + self.means[split_mask]
        new_proj = self.proj_means[split_mask]
        # step 2, sample new colors
        new_colors = self.colors[split_mask]
        # step 3, sample new shs
        new_shs = self.shs_rest[split_mask]
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask]
        # step 4, sample new scales
        new_scales = torch.logit(torch.exp(self.scales[split_mask]) / 1.6)
        self.scales[split_mask] = torch.logit(torch.exp(self.scales[split_mask]) / 1.6)
        # step 5, sample new quats
        new_quats = self.quats[split_mask]
        return new_means, new_proj, new_colors, new_shs, new_opacities, new_scales, new_quats

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"Would duplicate {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask] + self.means.grad[dup_mask]
        dup_proj = self.proj_means[dup_mask]
        dup_colors = self.colors[dup_mask]
        dup_shs = self.shs_rest[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        return dup_means, dup_proj, dup_colors, dup_shs, dup_opacities, dup_scales, dup_quats

    @property
    def num_points(self):
        return self.means.shape[0]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
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

    def step_cb(self, step):
        self.step = step

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        return {
            "xyz": [self.means],
            "color": [self.colors],
            "shs": [self.shs_rest],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
        }

    def _get_downscale_factor(self):
        return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)

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

        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        # dont mutate the input
        camera = deepcopy(camera)
        if self.training:
            d = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / d)

        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[..., :3, :3]  # 1 x 3 x 3
        T = camera.camera_to_worlds[..., :3, 3:4]  # 1 x 3 x 1
        R = vtf.SO3.from_matrix(R.cpu().squeeze().numpy())
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.from_numpy(R.as_matrix()[None, ...]).to(self.device, torch.float32)
        # vickie pops off here
        viewmat = torch.cat([R, T], dim=2)
        # add a row of zeros and a 1 to the bottom of the viewmat
        viewmat = torch.cat([viewmat, torch.tensor([[[0, 0, 0, 1]]], device=self.device)], dim=1)
        # invert it
        viewmat = torch.inverse(viewmat)
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        projmat = projection_matrix(0.1, 1000, fovx, fovy).to(self.device)
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
        pix_fac = torch.tensor([0.5 * W, 0.5 * H], device=self.device)[None].repeat(self.num_points, 1)
        pix_means = self.proj_means * pix_fac
        xys, depths, self.radii, conics, num_tiles_hit, self.cov3d = ProjectGaussians.apply(
            self.means,
            pix_means,
            torch.exp(self.scales),
            1,
            self.quats,
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            camera.cx.item(),
            camera.cy.item(),
            H,
            W,
            tile_bounds,
        )
        if self.degree > 0:
            viewdirs = self.means - camera.camera_to_worlds[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.degree)
            n_bases = num_sh_bases(n)
            # rgbs = compute_sh_color(viewdirs, self.colors[:,:n_bases])
            rgbs = eval_sh(n, self.get_colors, viewdirs)
            # rgbs = torch.clamp_min(rgbs + 0.5, 0.0)
            # print("Min", rgbs.min().item(), "Max", rgbs.max().item())
            # rgbs = torch.clamp_max(rgbs, 1.0)
            # rgbs = SphericalHarmonics.apply(n, viewdirs, self.colors[:,:n_bases])  # (N, 3)
        else:
            rgbs = self.get_colors.squeeze()  # (N, 3)
        # cx_delta = cx - W / 2
        # cy_delta = cy - H / 2
        # xys = xys.view(-1, 2) + torch.tensor([cx_delta, cy_delta], device=self.device)
        rgb = RasterizeGaussians.apply(
            xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(rgbs),
            # rgbs,
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
            # use torchvision to resize
            import torchvision.transforms.functional as TF

            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        Ll1 = torch.nn.functional.l1_loss(gt_img, outputs["rgb"])
        simloss = (1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])) / 2.0
        return {"main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss}

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, camera: Optional[Cameras] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        assert camera is not None, "must provide camera to gaussian model"
        outs = self.get_outputs(camera.to(self.device))
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

    def RGB2SH(self, rgb):
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24]
                    )
    return result
