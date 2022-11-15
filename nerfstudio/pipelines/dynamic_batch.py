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

"""
A pipeline that dynamically chooses the number of rays to sample.
"""

from dataclasses import dataclass, field
from typing import Type

import torch
from typing_extensions import Literal

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class DynamicBatchPipelineConfig(VanillaPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: DynamicBatchPipeline)
    target_num_samples: int = 262144  # 1 << 18
    """The target number of samples to use for an entire batch of rays."""
    max_num_samples_per_ray: int = 1024  # 1 << 10
    """The maximum number of samples to be placed along a ray."""


class DynamicBatchPipeline(VanillaPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    # pylint: disable=abstract-method

    config: DynamicBatchPipelineConfig
    datamanager: VanillaDataManager
    dynamic_num_rays_per_batch: int

    def __init__(
        self,
        config: DynamicBatchPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        assert isinstance(
            self.datamanager, VanillaDataManager
        ), "DynamicBatchPipeline only works with VanillaDataManager."

        self.dynamic_num_rays_per_batch = self.config.target_num_samples // self.config.max_num_samples_per_ray
        self._update_pixel_samplers()

    def _update_pixel_samplers(self):
        """Update the pixel samplers for train and eval with the dynamic number of rays per batch."""
        self.datamanager.train_pixel_sampler.set_num_rays_per_batch(self.dynamic_num_rays_per_batch)
        self.datamanager.eval_pixel_sampler.set_num_rays_per_batch(self.dynamic_num_rays_per_batch)

    def _update_dynamic_num_rays_per_batch(self, num_samples_per_batch: int):
        """Updates the dynamic number of rays per batch variable,
        based on the total number of samples in the last batch of rays."""
        self.dynamic_num_rays_per_batch = int(
            self.dynamic_num_rays_per_batch * (self.config.target_num_samples / num_samples_per_batch)
        )

    def get_train_loss_dict(self, step: int):
        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)

        # update the number of rays for the next step
        if "num_samples_per_batch" not in metrics_dict:
            raise ValueError(
                "'num_samples_per_batch' is not in metrics_dict."
                "Please return 'num_samples_per_batch' in the models get_metrics_dict function to use this method."
            )
        self._update_dynamic_num_rays_per_batch(metrics_dict["num_samples_per_batch"])
        self._update_pixel_samplers()

        # add the number of rays
        assert "num_rays_per_batch" not in metrics_dict
        metrics_dict["num_rays_per_batch"] = torch.tensor(self.datamanager.train_pixel_sampler.num_rays_per_batch)

        return model_outputs, loss_dict, metrics_dict

    def get_eval_loss_dict(self, step: int):
        model_outputs, loss_dict, metrics_dict = super().get_eval_loss_dict(step)

        # add the number of rays
        assert "num_rays_per_batch" not in metrics_dict
        metrics_dict["num_rays_per_batch"] = torch.tensor(self.datamanager.eval_pixel_sampler.num_rays_per_batch)

        return model_outputs, loss_dict, metrics_dict
