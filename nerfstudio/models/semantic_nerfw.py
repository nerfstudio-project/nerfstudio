# Copyright 2022 The nerfstudio Team. All rights reserved.
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
Semantic NeRF-W implementation which should be fast enough to view in the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    DepthLossType,
    MSELoss,
    depth_loss,
    distortion_loss,
    interlevel_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.utils import colormaps


@dataclass
class SemanticNerfWModelConfig(NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SemanticNerfWModel)
    use_transient_embedding: bool = False
    """Whether to use transient embedding."""
    semantic_loss_weight: float = 1.0
    """Lambda of the semantic loss"""
    pass_semantic_gradients: bool = False
    """Whether pass semantic gradients"""
    include_depth: bool = False
    """Whether include depth supervision"""
    depth_loss_mult: float = 1e-3
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    """Depth loss type."""


class SemanticNerfWModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: SemanticNerfWModelConfig

    def __init__(self, config: SemanticNerfWModelConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))

        if self.config.use_transient_embedding:
            raise ValueError("Transient embedding is not fully working for semantic nerf-w.")

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_transient_embedding=self.config.use_transient_embedding,
            use_semantics=True,
            num_semantic_classes=len(self.semantics.classes),
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )

        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
            self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for _ in range(self.config.num_proposal_iterations)]
        else:
            for _ in range(self.config.num_proposal_iterations):
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
                self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for network in self.proposal_networks]

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # Samplers
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # depth configurations
        if self.config.include_depth:
            if self.config.should_decay_sigma:
                self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
            else:
                self.depth_sigma = torch.tensor([self.config.depth_sigma])

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples)

        if self.training and self.config.use_transient_embedding:
            density = field_outputs[FieldHeadNames.DENSITY] + field_outputs[FieldHeadNames.TRANSIENT_DENSITY]
            weights = ray_samples.get_weights(density)
            weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            rgb_static_component = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            rgb_transient_component = self.renderer_rgb(
                rgb=field_outputs[FieldHeadNames.TRANSIENT_RGB], weights=weights
            )
            rgb = rgb_static_component + rgb_transient_component
        else:
            weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights = weights_static
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        weights_list.append(weights_static)
        ray_samples_list.append(ray_samples)

        depth = self.renderer_depth(weights=weights_static, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights_static)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # transients
        if self.training and self.config.use_transient_embedding:
            weights_transient = ray_samples.get_weights(field_outputs[FieldHeadNames.TRANSIENT_DENSITY])
            uncertainty = self.renderer_uncertainty(field_outputs[FieldHeadNames.UNCERTAINTY], weights_transient)
            outputs["uncertainty"] = uncertainty + 0.03  # NOTE(ethan): this is the uncertainty min
            outputs["density_transient"] = field_outputs[FieldHeadNames.TRANSIENT_DENSITY]

        # semantics
        semantic_weights = weights_static
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        # depth related outputs
        if self.config.include_depth:
            if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
                outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            if self.config.include_depth:
                metrics_dict["depth_loss"] = 0.0
                sigma = self._get_sigma().to(self.device)
                termination_depth = batch["depth_image"].to(self.device)
                for i in range(len(outputs["weights_list"])):
                    metrics_dict["depth_loss"] += depth_loss(
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                        termination_depth=termination_depth,
                        predicted_depth=outputs["depth"],
                        sigma=sigma,
                        directions_norm=outputs["directions_norm"],
                        is_euclidean=self.config.is_euclidean_depth,
                        depth_loss_type=self.config.depth_loss_type,
                    ) / len(outputs["weights_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

        # transient loss
        if self.training and self.config.use_transient_embedding:
            betas = outputs["uncertainty"]
            loss_dict["uncertainty_loss"] = 3 + torch.log(betas).mean()
            loss_dict["density_loss"] = 0.01 * outputs["density_transient"].mean()
            loss_dict["rgb_loss"] = (((image - outputs["rgb"]) ** 2).sum(-1) / (betas[..., 0] ** 2)).mean()
        else:
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        # semantic loss
        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.cross_entropy_loss(
            outputs["semantics"], batch["semantics"][..., 0].long()
        )

        # depth loss
        if self.training and self.config.include_depth:
            loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        rgb = torch.clamp(rgb, min=0, max=1)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        # semantics
        ground_truth_semantic_labels = torch.squeeze(batch["semantics"])
        predicted_semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        ground_truth_semantic_colormap = self.semantics.colors[ground_truth_semantic_labels]
        predicted_semantic_colormap = self.semantics.colors[predicted_semantic_labels]
        images_dict["semantics_colormap"] = torch.cat(
            [ground_truth_semantic_colormap, predicted_semantic_colormap], dim=1
        )

        # valid mask
        images_dict["mask"] = batch["mask"].repeat(1, 1, 3)

        # depth metrics
        if self.config.include_depth:
            ground_truth_depth = batch["depth_image"]
            if not self.config.is_euclidean_depth:
                ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

            ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
            predicted_depth_colormap = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
                near_plane=torch.min(ground_truth_depth),
                far_plane=torch.max(ground_truth_depth),
            )
            images_dict["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
            depth_mask = ground_truth_depth > 0
            metrics_dict["depth_mse"] = torch.nn.functional.mse_loss(
                outputs["depth"][depth_mask], ground_truth_depth[depth_mask]
            )

        return metrics_dict, images_dict

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(  # pylint: disable=attribute-defined-outside-init
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma
