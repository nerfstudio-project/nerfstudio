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
Implementation of Neuralangelo model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils import colormaps


@dataclass
class NeuralangeloModelConfig(NeuSModelConfig):
    """Neuralangelo Model Config"""

    _target: Type = field(default_factory=lambda: NeuralangeloModel)
    #TODO move to base model config since it can be used in all models
    enable_progressive_hash_encoding: bool = True
    """whether to use progressive hash encoding"""
    enable_numerical_gradients_schedule: bool = True
    """whether to use numerical gradients delta schedule"""
    enable_curvature_loss_schedule: bool = True
    """whether to use curvature loss weight schedule"""
    curvature_loss_multi: float = 5e-4
    """curvature loss weight"""
    curvature_loss_warmup_steps: int = 5000
    """curvature loss warmup steps"""
    level_init: int = 4
    """initial level of multi-resolution hash encoding"""
    steps_per_level: int = 5000
    """steps per level of multi-resolution hash encoding"""

class NeuralangeloModel(NeuSModel):
    """Neuralangelo model

    Args:
        config: Neuralangelo configuration to instantiate model
    """

    config: NeuralangeloModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.curvature_loss_multi_factor = 1.0

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # read the hash encoding parameters from field
        level_init = self.config.level_init
        # schedule the delta in numerical gradients computation
        num_levels = self.field.num_levels
        max_res = self.field.max_res
        base_res = self.field.base_res
        growth_factor = self.field.growth_factor
        
        steps_per_level = self.config.steps_per_level
        
        
        init_delta = 1. / base_res
        end_delta = 1. / max_res
        
        # compute the delta based on level
        if self.config.enable_numerical_gradients_schedule:
            def set_delta(step):
                delta = 1. / (base_res * growth_factor ** ( step / steps_per_level))
                delta = max(1. / max_res, delta)
                self.field.set_numerical_gradients_delta(delta * 2.) # TODO because we divide 4 to normalize points to [0, 1]
                
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_delta,
                )
            )
        
        # schedule the current level of multi-resolution hash encoding
        if self.config.enable_progressive_hash_encoding:
            def set_mask(step):
                #TODO make this consistent with delta schedule
                level = int(step / steps_per_level) + 1
                level = max(level, level_init)
                self.field.update_mask(level)
    
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_mask,
                )
            )
        # schedule the curvature loss weight
        # linear warmup for 5000 steps to 5e-4 and then decay as delta
        if self.config.enable_curvature_loss_schedule:
            def set_curvature_loss_mult_factor(step):
                if step < self.config.curvature_loss_warmup_steps:
                    factor = step / self.config.curvature_loss_warmup_steps
                else:
                    delta = 1. / (base_res * growth_factor ** ( (step - self.config.curvature_loss_warmup_steps) / steps_per_level))
                    delta = max(1. / max_res, delta)
                    factor = delta / init_delta

                self.curvature_loss_multi_factor = factor
            
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_curvature_loss_mult_factor,
                )
            )
        
        
        #TODO switch to analytic gradients after delta is small enough?
        
        return callbacks
    
    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        
        if self.training:
            # training statics
            metrics_dict["activated_encoding"] = self.field.hash_encoding_mask.mean().item()
            metrics_dict["numerical_gradients_delta"] = self.field.numerical_gradients_delta
            metrics_dict["curvature_loss_multi"] = self.curvature_loss_multi_factor * self.config.curvature_loss_multi
            
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # curvature loss
        if self.training and self.config.curvature_loss_multi > 0.0:
            delta = self.field.numerical_gradients_delta
            centered_sdf = outputs['field_outputs'][FieldHeadNames.SDF]
            sourounding_sdf = outputs['field_outputs']["sampled_sdf"]
            
            sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))
            
            # (a - b)/d - (b -c)/d = (a + c - 2b)/d
            # ((a - b)/d - (b -c)/d)/d = (a + c - 2b)/(d*d)
            curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta * delta)
            loss_dict["curvature_loss"] = torch.abs(curvature).mean() * self.config.curvature_loss_multi * self.curvature_loss_multi_factor
            
        return loss_dict

