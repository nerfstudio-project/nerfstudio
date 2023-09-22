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
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.models.base_model import Model, ModelConfig
import math
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
    t = math.tan(0.5 * fovy)
    b = -t
    r = math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, -(f + n) / (f - n), -2.0 * f * n / (f - n)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        **kwargs
    )


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""
    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    num_points: int = 100000
    

class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    TODO (jake-austin): Figure out how to print out on the training log in terminal the number of splats

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def populate_modules(self):
        # TODO (jake-austin): clean this up, this is transplanted code across all the implementation functions
        self.means = 2 * (torch.nn.Parameter(torch.rand(self.config.num_points, 3, device=self.device) - 0.5))
        self.scales = torch.nn.Parameter(torch.rand(self.config.num_points, 3, device=self.device))
        # self.quats = random_quat_tensor(self.num_points, device=self.device)
        self.quats = torch.nn.Parameter(identity_quat(self.num_points, device=self.device))
        self.rgbs = torch.nn.Parameter(torch.rand(self.config.num_points, 3, device=self.device))
        self.opacities = torch.nn.Parameter(0.5 * torch.ones(self.config.num_points, 1, device=self.device))

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        return []

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        return {
            "xyz": [self.means],
            "color": [self.rgbs],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
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

        #print(f"Num Points: {self._xyz.shape[0]}")


        camera = ray_bundle.reshape(())
        assert camera.shape == ()
        R = camera.camera_to_worlds[..., :3, :3].squeeze()
        R[:,0] = -R[:,0]
        T = (R.T @ camera.camera_to_worlds[..., :3, 3:4]).squeeze()
        # return {"rgb": rgb}


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
        Ll1 = torch.nn.functional.l1_loss(batch['image'], outputs['rgb'])
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

