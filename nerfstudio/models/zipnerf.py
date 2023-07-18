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
ZipNeRF implementation.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.field_components.spatial_distortions import LinearizedSceneContraction
from nerfstudio.model_components.losses import (
    zipnerf_loss,
    CharbonnierLoss, 
)
from nerfstudio.fields.zipnerf_field import ZipNeRFField
from nerfstudio.model_components.ray_samplers import PowerSampler
from nerfstudio.fields.density_fields import HashMLPGaussianDensityField


@dataclass
class ZipNeRFModelConfig(NerfactoModelConfig):
    """ZipNeRF Model Config"""

    _target: Type = field(default_factory=lambda: ZipNeRFModel)

    proposal_weights_anneal_max_num_iters: int = 1
    """Max num iterations for the annealing function."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128)
    """Number of samples per ray for each proposal network."""
    interlevel_loss_mult: float = 1e-1
    """Proposal loss multiplier."""
    hash_decay_loss_mult: float = 1e-2
    """Hash decay loss multiplier."""
    compute_regularize_hash: bool = True
    """Whether to compute regularization on hash weights."""
    scale_featurization: bool = True
    """Scale featurization from appendix of ZipNeRF."""
    regularize_function: Literal["abs", "square"] = "square"
    """Type of regularization."""
    compute_hash_regularization: bool = True
    """Whether to compute regularization on hash weights."""
    proposal_initial_sampler: Literal["power"] = "power"
    """Initial sampler for the proposal network."""
    interlevel_loss_type: Literal["zipnerf"] = "zipnerf"
    """Type of interlevel loss."""
    implementation: Literal["tcnn", "torch"] = "torch"
    """Which implementation to use for the model."""


class ZipNeRFModel(NerfactoModel):
    """ZipNeRF model

    Args:
        config: ZipNeRF configuration to instantiate model
    """

    config: ZipNeRFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = LinearizedSceneContraction(order=float("inf"))
        regularize_function = getattr(torch, self.config.regularize_function, torch.square)

        # Fields
        self.field = ZipNeRFField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            scale_featurization=self.config.scale_featurization,
            regularize_function=regularize_function,
            compute_hash_regularization=self.config.compute_hash_regularization,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPGaussianDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                scale_featurization=self.config.scale_featurization,
                regularize_function=regularize_function,
                compute_hash_regularization=self.config.compute_hash_regularization,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPGaussianDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    scale_featurization=self.config.scale_featurization,
                    regularize_function=regularize_function,
                    compute_hash_regularization=self.config.compute_hash_regularization,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        self.proposal_sampler.initial_sampler = PowerSampler(single_jitter=self.config.use_single_jitter)

        # Losses
        self.rgb_loss = CharbonnierLoss()
        self.interlevel_loss = zipnerf_loss
