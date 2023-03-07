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
NeRF implementation that combines many recent advancements.
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
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField, InputWavelengthStyle
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample"] = "white"
    """Whether to randomize the background color."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    num_output_color_channels: int = 3
    """For Hyperspectral, we may want to have a different number of output channels (RGB is 3)."""
    rgb_output_channels: Tuple[int, int, int] = (49, 36, 26)  # in nanometers: [620, 555, 503]
    """For Hyperspectral, we want to generate an RGB preview using these channels."""
    num_density_channels: int = 1
    """For wavelength-dependent transparency, we might want to have more density channels."""
    wavelength_style: InputWavelengthStyle = InputWavelengthStyle.NONE
    """Sets how to use the input wavelength."""
    num_wavelength_encoding_freqs: int = 2
    """Number of frequencies to use for the wavelength encoding."""
    proposal_wavelength_use: bool = False
    proposal_wavelength_encoding_freqs: int = 2
    proposal_wavelength_latent_embedding_dim: int = 7
    proposal_wavelength_num_layers: int = 2
    num_wavelength_samples_per_batch: int = -1
    """When wavelength_style is not NONE, this determines how many wavelengths to sample per batch"""
    train_wavelengths_every_nth: int = 1


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = None

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            num_output_color_channels=self.config.num_output_color_channels if not self.config.proposal_wavelength_use else 1,
            num_output_density_channels=self.config.num_density_channels,
            wavelength_style=self.config.wavelength_style,
            num_wavelength_encoding_freqs=self.config.num_wavelength_encoding_freqs,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        wavelength_proposal_kwargs = {
            "latent_embedding_dim":
                self.config.proposal_wavelength_latent_embedding_dim,
            "wavelength_layers":
                self.config.proposal_wavelength_num_layers,
            "wavelength_embedding":
                None if not self.config.proposal_wavelength_use else tcnn.Encoding(
                    n_input_dims=1,
                    encoding_config={
                        "otype": "Frequency",
                        "n_frequencies": self.config.proposal_wavelength_encoding_freqs,
                    })
        }
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb,
                                          spatial_distortion=scene_contraction,
                                          **wavelength_proposal_kwargs,
                                          **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **wavelength_proposal_kwargs,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        # Collider
        # self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        # self.collider = NearFarCollider(near_plane=4, far_plane=7)
        self.collider = AABBBoxCollider(self.scene_box)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.config.rgb_output_channels = tuple(wavelength *
                                                self.config.num_output_color_channels // 128
                                                for wavelength in self.config.rgb_output_channels)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

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
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_wavelengths(self, override_wavelengths=None):
        n_wavelengths = self.config.num_output_color_channels
        if override_wavelengths is not None:
            wavelengths = torch.tensor(override_wavelengths, dtype=torch.float32, device=self.device, requires_grad=False) / n_wavelengths
        elif not self.training:
            wavelengths = torch.arange(
                n_wavelengths,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            ) / n_wavelengths
        elif self.config.num_wavelength_samples_per_batch < 0:
            wavelengths = torch.arange(
                start=0,
                end=n_wavelengths,
                step=self.config.train_wavelengths_every_nth * 1.0,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            ) / n_wavelengths
        else:
            # TODO(gerry): use different wavelengths for every sample?
            wavelengths = (
                torch.randperm(n_wavelengths // self.config.train_wavelengths_every_nth,
                               dtype=torch.float32,
                               device=self.device,
                               requires_grad=False)[:self.config.num_wavelength_samples_per_batch] *
                self.config.train_wavelengths_every_nth) / n_wavelengths
        return len(wavelengths), wavelengths

    def run_network_for_every_wavelength_batch(self, ray_samples, override_wavelengths=None):
        # First create the batched "wavelengths" metadata
        n_ch, wavelengths = self.get_wavelengths(override_wavelengths)
        ray_samples.metadata['wavelengths'] = torch.ones(
            (*ray_samples.frustums.shape, 1), dtype=torch.float32,
            device=self.device, requires_grad=False) * wavelengths.view(1, 1, -1)
        # broadcast the frustums and camera indices to repeat the wavelength batch dimension
        ray_samples.frustums = ray_samples.frustums[:, :, None].broadcast_to((-1, -1, n_ch))
        ray_samples.camera_indices = ray_samples.camera_indices.broadcast_to((-1, -1, n_ch))
        # execute forward pass and squeeze outputs
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(ray_samples.metadata['wavelengths'].shape)
        field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(ray_samples.metadata['wavelengths'].shape)
        # un-broadcast the frustums and camera indices
        ray_samples.frustums = ray_samples.frustums[:, :, 0]
        ray_samples.camera_indices = ray_samples.camera_indices[:, :, 0]
        ray_samples.wavelengths = wavelengths
        return field_outputs

    def run_network_for_every_wavelength_batch_partial(self, ray_samples, override_wavelengths=None):
        # First create the batched "wavelengths" metadata
        n_ch, ray_samples.wavelengths = self.get_wavelengths(override_wavelengths)
        # execute forward pass and squeeze outputs
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(*ray_samples.frustums.shape, n_ch)
        field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(*ray_samples.frustums.shape, n_ch)
        return field_outputs

    def run_proposal_for_every_wavelength(self, ray_bundle: RayBundle):
        if not self.config.proposal_wavelength_use:
            return self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        else:
            n_ch, wavelengths = self.get_wavelengths(ray_bundle.wavelengths)
            # broadcast ray_bundle repeat the wavelength batch dimension
            ray_bundle = ray_bundle[:, None].broadcast_to((-1, n_ch))
            # add the wavelengths metadata
            ray_bundle.metadata["wavelengths"] = (torch.ones(
                ray_bundle.shape, dtype=torch.float32, device=self.device,
                requires_grad=False) * wavelengths.view(1, -1)).view(-1, 1)
            # execute forward pass and squeeze outputs
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle.reshape(-1), density_fns=self.density_fns)
            # (n_rays*n_ch, n_samples_per_ray)
            # shape = (ray_bundle.shape[0], n_ch, self.proposal_sampler.num_nerf_samples_per_ray)
            shape = (ray_bundle.shape[0], n_ch, -1)
            # ray_samples = ray_samples.reshape(shape)
            # # ray_samples = ray_samples.permute(0, 2, 1, 3).reshape((ray_bundle.shape[0] * self.proposal_sampler.num_nerf_samples_per_ray, n_ch))
            # ray_samples = ray_samples.permute(0, 2, 1, 3).contiguous()
            ray_samples = ray_samples.reshape(shape).permute(0, 2, 1, 3).contiguous()
            weights_list = [weight.reshape(shape).permute(0, 2, 1).contiguous()[..., 0:1] for weight in weights_list]
            ray_samples_list = [ray_sample.reshape(shape).permute(0, 2, 1, 3).contiguous()[..., 0] for ray_sample in ray_samples_list]
            ray_samples.wavelengths = wavelengths
            ray_samples.n_ch = n_ch
            return ray_samples, weights_list, ray_samples_list

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.run_proposal_for_every_wavelength(ray_bundle)
        if not self.config.proposal_wavelength_use:
            if self.config.wavelength_style == InputWavelengthStyle.BEFORE_BASE:
                field_outputs = self.run_network_for_every_wavelength_batch(ray_samples)
                # ray_samples.metadata["set_of_wavelengths"] = torch.arange(self.config.num_output_color_channels, dtype=torch.float32).to(self.device)
                # ray_samples.metadata["set_of_wavelengths"].requires_grad = False
                # print(ray_samples.metadata["set_of_wavelengths"].shape, ray_samples.metadata["set_of_wavelengths"].dtype)
                # field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
            elif self.config.wavelength_style == InputWavelengthStyle.AFTER_BASE:
                field_outputs = self.run_network_for_every_wavelength_batch_partial(ray_samples, override_wavelengths=ray_bundle.wavelengths)
            else:
                field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        else:
            field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
            shape = (ray_bundle.shape[0], self.proposal_sampler.num_nerf_samples_per_ray, ray_samples.n_ch)
            field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY].view(shape)
            field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB].view(shape)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        if not self.config.proposal_wavelength_use:
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)
        else:
            weights_list.append(weights)
            ray_samples_list.append(ray_samples[..., 0])

        if not self.config.wavelength_style != InputWavelengthStyle.NONE and not self.config.proposal_wavelength_use:
            assert weights.shape[-1] == self.config.num_density_channels, f"weights should have num_density_channels channels: {weights.shape = }"
        else:
            assert weights.shape[-1] == len(ray_samples.wavelengths), f"weights should have {len(ray_samples.wavelengths)=} channels: {weights.shape = }"

        # when num_output_color_channels is not 3, then image will also not be 3.
        image = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples if not self.config.proposal_wavelength_use else ray_samples[..., 0])
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": image,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.num_output_color_channels != 3:
            if self.config.wavelength_style == InputWavelengthStyle.NONE and not self.config.proposal_wavelength_use:
                rgb_indices = self.config.rgb_output_channels
                outputs["wavelengths"] = torch.arange(self.config.num_output_color_channels,
                                                      device=self.device, requires_grad=False).long()
            else:
                outputs["wavelengths"] = (ray_samples.wavelengths *
                                        self.config.num_output_color_channels).long()
                if self.config.num_wavelength_samples_per_batch < 0:
                    # if all wavelengths are used, then we can just use the rgb_output_channels
                    rgb_indices = self.config.rgb_output_channels
                else:
                    # find the indices in wavelengths that are closest to rgb_output_channels
                    rgb_indices = torch.argmin(torch.abs(outputs["wavelengths"][:, None] -
                                                        torch.tensor(self.config.rgb_output_channels,
                                                                    dtype=torch.float32,
                                                                    device=self.device,
                                                                    requires_grad=False)[None, :]),
                                            dim=0)
            outputs["rgb"] = image[:, rgb_indices]  # [rays, 3]
            outputs["image"] = image  # [rays, num_output_color_channels]

        if self.config.predict_normals:
            outputs["normals"] = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            outputs["pred_normals"] = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        if self.config.num_output_color_channels == 3:
            image = batch["image"].to(self.device)
            metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        else:
            image = batch["hs_image"][..., outputs["wavelengths"]].to(self.device)
            metrics_dict["psnr"] = self.psnr(outputs["image"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        if self.config.num_output_color_channels == 3:
            image = batch["image"].to(self.device)
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        else:
            image = batch["hs_image"][..., outputs["wavelengths"]].to(self.device)
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["image"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        rgb_gt = batch["image"].to(self.device)
        rgb = outputs["rgb"] * 1.0
        if self.config.num_output_color_channels != 3:
            image_gt = batch["hs_image"][..., outputs["wavelengths"]].to(self.device)
            image = outputs["image"]
        else:
            image_gt = rgb_gt
            image = rgb
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([rgb_gt, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        image_gt = torch.moveaxis(image_gt, -1, 0)[None, ...]
        image = torch.moveaxis(image, -1, 0)[None, ...]


        n_wavelengths = self.config.num_output_color_channels
        eval_ch_inds = torch.arange(
            n_wavelengths,
            device=self.device,
            requires_grad=False,
        ).long()
        eval_ch_inds = eval_ch_inds[eval_ch_inds % self.config.train_wavelengths_every_nth != 0]

        psnr = self.psnr(image_gt, image)
        ssim = self.ssim(image_gt, image)
        lpips = self.lpips(rgb_gt, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.train_wavelengths_every_nth > 1:
            psnr_wavelengths = self.psnr(image_gt[:, eval_ch_inds, :, :], image[:, eval_ch_inds, :, :])
            ssim_wavelengths = self.ssim(image_gt[:, eval_ch_inds, :, :], image[:, eval_ch_inds, :, :])
            metrics_dict["psnr_eval_wavelengths"] = float(psnr_wavelengths.item())
            metrics_dict["ssim_eval_wavelengths"] = float(ssim_wavelengths)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        # normals to RGB for visualization. TODO: use a colormap
        if "normals" in outputs:
            images_dict["normals"] = (outputs["normals"] + 1.0) / 2.0
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = (outputs["pred_normals"] + 1.0) / 2.0

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, override_wavelengths=None) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        if self.config.num_output_color_channels != 3:
            if override_wavelengths is None:
                # camera_ray_bundle.wavelengths = torch.arange(128, device=self.device, requires_grad=False).long()
                camera_ray_bundle.wavelengths = list(range(128))
            elif override_wavelengths == 'rgb':
                camera_ray_bundle.wavelengths = self.config.rgb_output_channels
            else:
                camera_ray_bundle.wavelengths = override_wavelengths
        with torch.no_grad():
            return super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
