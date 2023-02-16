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

import torch
from torch.cuda.amp.grad_scaler import GradScaler
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
    # prompt: str = "A high-quality photo of a tree frog on a stump"
    # """prompt for stable dreamfusion"""

    location_based_prompting: bool = True
    """enables location based prompting"""
    interpolated_prompting: bool = False
    """enables interpolated location prompting"""
    top_prompt: str = ", overhead view"
    """appended to prompt for overhead view"""
    side_prompt: str = ", side view"
    """appended to prompt for side view"""
    front_prompt: str = ", front view"
    """appended to prompt for front view"""
    back_prompt: str = ", back view"
    """appended to prompt for back view"""
    guidance_scale: float = 100
    """guidance scale for sds loss"""
    sd_version: str = "1-5"
    """Stable Diffusion version for dreamfusion"""

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
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        self.generative = True
        self.cur_prompt = self.model.prompt
        self.grad_scaler = grad_scaler
        self.sd_device = (
            torch.device(device)
            if self.config.stablediffusion_device is None
            else torch.device(self.config.stablediffusion_device)
        )
        self.location_based_prompting = config.location_based_prompting
        self.top_prompt = config.top_prompt
        self.side_prompt = config.side_prompt
        self.back_prompt = config.back_prompt 
        self.front_prompt = config.front_prompt
        self.sd = StableDiffusion(self.sd_device, version=config.sd_version)
        
        if config.location_based_prompting:
            self.top_text_embedding = self.sd.get_text_embeds(f"{self.cur_prompt}{self.top_prompt}", "")
            self.front_text_embedding = self.sd.get_text_embeds(f"{self.cur_prompt}{self.front_prompt}", "")
            self.side_text_embedding = self.sd.get_text_embeds(f"{self.cur_prompt}{self.side_prompt}", "")
            self.back_text_embedding = self.sd.get_text_embeds(f"{self.cur_prompt}{self.back_prompt}", "")
        else:
            self.base_text_embedding = self.sd.get_text_embeds(self.cur_prompt, "")

    def get_train_loss_dict(self, step: int):

        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)

        if self.model.prompt != self.cur_prompt:
            self.cur_prompt = self.model.prompt
            if self.location_based_prompting:
                self.top_text_embedding = self.sd.get_text_embeds(f"{self.cur_prompt}{self.top_prompt}", "")
                self.front_text_embedding = self.sd.get_text_embeds(f"{self.cur_prompt}{self.front_prompt}", "")
                self.side_text_embedding = self.sd.get_text_embeds(f"{self.cur_prompt}{self.side_prompt}", "")
                self.back_text_embedding = self.sd.get_text_embeds(f"{self.cur_prompt}{self.back_prompt}", "")
            else:
                self.base_text_embedding = self.sd.get_text_embeds(self.cur_prompt, "")

        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        train_output = (
            model_outputs["train_output"]
            .view(1, self.config.datamanager.train_resolution, self.config.datamanager.train_resolution, 3)
            .permute(0, 3, 1, 2)
        )

        if self.config.location_based_prompting:
            if self.config.interpolated_prompting:
                horiz = batch["central"].to(self.sd_device)
                vert = max(batch["vertical"].to(self.sd_device), 0)

                if batch["central"] > 0 or batch["central"] <= 90:
                    text_embedding = (
                        (horiz) * self.side_text_embedding + (90 - horiz) * self.front_text_embedding
                    ) / 90.0
                elif batch["central"] > 90 and batch["central"] <= 180:
                    text_embedding = (
                        (horiz - 90) * self.back_text_embedding + (180 - horiz) * self.side_text_embedding
                    ) / 90.0
                elif batch["central"] > 180 and batch["central"] <= 270:
                    text_embedding = (
                        (horiz - 180) * self.side_text_embedding + (270 - horiz) * self.back_text_embedding
                    ) / 90.0
                else:  # batch["central"] > 270 and batch["central"] <= 360:
                    text_embedding = (
                        (horiz - 270) * self.front_text_embedding + (360 - horiz) * self.side_text_embedding
                    ) / 90.0

                text_embedding = (vert * text_embedding + (90 - vert) * self.top_text_embedding) / 90.0

            else:
                if batch["vertical"] < 40:
                    text_embedding = self.top_text_embedding
                elif batch["central"] > 315 or batch["central"] <= 45:
                    text_embedding = self.front_text_embedding
                elif batch["central"] > 45 and batch["central"] <= 135:
                    text_embedding = self.side_text_embedding
                elif batch["central"] > 135 and batch["central"] <= 225:
                    text_embedding = self.back_text_embedding
                else:  # batch["central"] > 225 and batch["central"] <= 315:
                    text_embedding = self.side_text_embedding
        else:
            text_embedding = self.base_text_embedding

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            sds_loss = self.sd.sds_loss(
                text_embedding.to(self.sd_device),
                train_output.to(self.sd_device),
                guidance_scale=int(self.config.guidance_scale),
                grad_scaler=self.grad_scaler,
            )

        loss_dict["sds_loss"] = sds_loss.to(self.device)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
