"""
Regional Nerfacto Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch

from regional_nerfacto.field import RNerfField

import nerfacc

from nerfstudio.field_components.spatial_distortions import SceneContraction

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.field_components.field_heads import FieldHeadNames


@dataclass
class RNerfModelConfig(NerfactoModelConfig):
    """RNerf Model Configuration.
    """

    _target: Type = field(default_factory=lambda: RNerfModel)
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class RNerfModel(NerfactoModel):
    """Regional NeRF Model."""

    config: RNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        
        # Fields
        self.field = RNerfField(
            grid_resolutions=self.config.hashgrid_resolutions,
            grid_layers=self.config.hashgrid_layers,
            grid_sizes=self.config.hashgrid_sizes,
            aabb=self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.tall_loss_factor = 0.01
        self.max_height = 1.0

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        
        outputs = super().get_outputs(ray_bundle)

        outputs["dino"] = torch.sum(weights * field_outputs["dino"], dim=-2)
        
        # # Extract height of samples and compute exponentially scaled density
        # # TODO: NAVLAB
        # positions = ray_samples.frustums.get_positions()
        # # height = torch.clip(positions[..., 2][..., None] + 10.0, 0.0, self.max_height + 10.0)
        # height = torch.exp(positions[..., 2][..., None] - self.max_height)
        
        # height_opacity = torch.sum(field_outputs[FieldHeadNames.DENSITY] * height, dim=-2)

        # outputs["height_opacity"] = height_opacity

        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = 0.1*unreduced_dino.sum(dim=-1).nanmean()

        # Add height opacity loss by its average
        # TODO: NAVLAB
        # loss_dict["height_opacity_loss"] = self.tall_loss_factor * torch.mean(
        #     outputs["height_opacity"]
        #     )
        return loss_dict