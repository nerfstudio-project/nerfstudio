# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Dreamfusion Pipeline and trainer"""

from dataclasses import dataclass, field
from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing_extensions import Literal

from nerfstudio.data.datamanagers.dreamfusion_datamanager import (
    DreamFusionDataManagerConfig,
)
from nerfstudio.generative.stable_diffusion import StableDiffusion
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class DreamfusionPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DreamfusionPipeline)
    """target class to instantiate"""
    datamanager: DreamFusionDataManagerConfig = DreamFusionDataManagerConfig()
    """specifies the datamanager config"""
    prompt: str = "A high quality photo of a pineapple"
    """prompt for stable dreamfusion"""

    location_based_prompting: bool = True
    """enables location based prompting"""
    side_prompt: str = ", side view"
    """appended to prompt for side view"""
    front_prompt: str = ", front view"
    """appended to prompt for front view"""
    back_prompt: str = ", back view"
    """appended to prompt for back view"""
    alphas_penalty: bool = True
    """enables penalty to encourage sparse weights (penalizing for uniform density along ray)"""
    opacity_penalty: bool = True
    """enables penalty to encourage transparent scenes, as in "dreamfields" paper"""
    target_transmittance: float = 0.8
    """target transmittance for opacity penalty"""

    guidance_scale: float = 100
    """guidance scale for sds loss"""

    stablediffusion_device: Optional[str] = None


class DreamfusionPipeline(VanillaPipeline):
    """Dreamfusion pipeline"""

    config: DreamfusionPipelineConfig

    def __init__(
        self,
        config: DreamfusionPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.generative = True
        self.sd_device = (
            torch.device(device)
            if self.config.stablediffusion_device is None
            else torch.device(self.config.stablediffusion_device)
        )
        self.sd = StableDiffusion(self.sd_device)
        if config.location_based_prompting:
            self.front_text_embedding = self.sd.get_text_embeds(f"{config.prompt}{config.front_prompt}", "")
            self.side_text_embedding = self.sd.get_text_embeds(f"{config.prompt}{config.side_prompt}", "")
            self.back_text_embedding = self.sd.get_text_embeds(f"{config.prompt}{config.back_prompt}", "")
        else:
            self.base_text_embedding = self.sd.get_text_embeds(config.prompt, "")

    def get_train_loss_dict(self, step: int):

        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        train_output = (
            model_outputs["train_output"]
            .view(1, self.config.datamanager.train_resolution, self.config.datamanager.train_resolution, 3)
            .permute(0, 3, 1, 2)
        )

        accumulation = (
            model_outputs["accumulation"]
            .view(self.config.datamanager.train_resolution, self.config.datamanager.train_resolution)
            .detach()
            .cpu()
            .numpy()
        )
        accumulation = np.clip(accumulation, 0.0, 1.0)
        plt.imsave("nerf_accumulation.jpg", accumulation)

        # background = model_outputs["background"].view(res, res, 3).detach().cpu().numpy()
        # background = np.clip(background, 0.0, 1.0)
        # plt.imsave("nerf_background.jpg", background)

        # shaded = model_outputs["shaded"].view(res, res, 3).detach().cpu().numpy()
        # shaded = np.clip(shaded, 0.0, 1.0)
        # plt.imsave("nerf_textureless.jpg", shaded)

        if self.config.location_based_prompting:
            if batch["central"] > 315 or batch["central"] <= 45:
                text_embedding = self.front_text_embedding
            elif batch["central"] > 45 and batch["central"] <= 135:
                text_embedding = self.side_text_embedding
            elif batch["central"] > 135 and batch["central"] <= 225:
                text_embedding = self.back_text_embedding
            else:  # batch["central"] > 225 and batch["central"] <= 315:
                text_embedding = self.side_text_embedding
        else:
            text_embedding = self.base_text_embedding

        sds_loss, latents, grad = self.sd.sds_loss(
            text_embedding.to(self.sd_device),
            train_output.to(self.sd_device),
            guidance_scale=int(self.config.guidance_scale),
        )
        loss_dict["sds_loss"] = sds_loss.to(self.device)
        # TODO: opacity penalty using transmittance, not accumultation
        if self.config.opacity_penalty:
            accum_mean = np.mean(1.0 - accumulation)
            sds_loss *= np.min((0.5, accum_mean))
            loss_dict["opacity_loss"] = -torch.minimum(
                (1 - model_outputs["accumulation"]).mean(), torch.tensor(self.config.target_transmittance)
            )

        model_outputs["latents"] = latents
        model_outputs["grad"] = grad

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
