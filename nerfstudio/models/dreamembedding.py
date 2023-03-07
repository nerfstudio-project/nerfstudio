from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type


import torch

from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.dreamembedding_field import DreamEmbeddingField
from nerfstudio.generative.stable_diffusion import StableDiffusion
from nerfstudio.generative.stable_diffusion_utils import PositionalTextEmbeddings
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
)

from nerfstudio.model_components.renderers import FeatureRenderer
from nerfstudio.models.dreamfusion import DreamFusionModel, DreamFusionModelConfig
from nerfstudio.utils import colormaps, colors, math, misc


@dataclass
class DreamEmbeddingModelConfig(DreamFusionModelConfig):
    """DreamFusion model config"""

    _target: Type = field(default_factory=lambda: DreamEmbeddingModel)
    """target class to instantiate"""
    prompt: str = "A high quality photo of a tiger wearing a lab coat"
    """prompt for stable dreamfusion"""

    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    random_light_source: bool = True
    """Randomizes light source per output."""
    initialize_density: bool = True
    """Initialize density in center of scene."""
    taper_range: Tuple[int, int] = (0, 1000)
    """Range of step values for the density tapering"""
    taper_strength: Tuple[float, float] = (1.0, 0.0)
    """Strength schedule of center density"""
    sphere_collider: bool = True
    """Use spherical collider instead of box"""
    random_background: bool = True
    """Randomly choose between using background mlp and random color for background"""
    target_transmittance_start: float = 0.4
    """target transmittance for opacity penalty. This is the percent of the scene that is
    background when rendered at the start of training"""
    target_transmittance_end: float = 0.7
    """target transmittance for opacity penalty. This is the percent of the scene that is
    background when rendered at the end of training"""
    transmittance_end_schedule: int = 1500
    """number of iterations to reach target_transmittance_end"""
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
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 128,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 256,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 500
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 1.0
    """Distortion loss multiplier."""
    start_normals_training: int = 1000
    """Start training normals after this many iterations"""
    start_lambertian_training: int = 1000
    """start training with lambertian shading after this many iterations"""
    opacity_penalty: bool = True
    """enables penalty to encourage sparse weights (penalizing for uniform density along ray)"""
    opacity_loss_mult: float = 1
    """scale for opacity penalty"""
    max_res: int = 256
    """Maximum resolution of the density field."""

    location_based_prompting: bool = True
    """enables location based prompting"""
    interpolated_prompting: bool = False
    """enables interpolated location prompting"""
    positional_prompting: Literal["discrete", "interpolated", "off"] = "discrete"
    """ how to incorporate position into prompt"""
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
    stablediffusion_device: Optional[str] = None
    """device for stable diffusion"""
    sd_version: str = "1-5"


class DreamEmbeddingModel(DreamFusionModel):
    """DreamEmbeddingModel Model

    Args:
        config: DreamFusion configuration to instantiate model
    """

    config: DreamEmbeddingModelConfig

    def __init__(
        self,
        config: DreamEmbeddingModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        self.field = DreamEmbeddingField(
            self.scene_box.aabb, max_res=self.config.max_res
        )

        self.sd = StableDiffusion(self.sd_device, version=self.sd_version)
        self.text_embeddings = PositionalTextEmbeddings(
            base_prompt=self.cur_prompt,
            top_prompt=self.cur_prompt + self.top_prompt,
            side_prompt=self.cur_prompt + self.side_prompt,
            back_prompt=self.cur_prompt + self.back_prompt,
            front_prompt=self.cur_prompt + self.front_prompt,
            stable_diffusion=self.sd,
            positional_prompting=self.positional_prompting,
        )

        self.renderer_feature = FeatureRenderer()

    def get_outputs(self, ray_bundle: RayBundle, ray_dims: Tuple):
        # uniform sampling
        background_feature = self.field.get_background_feature(ray_bundle)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        field_outputs = self.field(ray_samples, compute_normals=True)
        density = field_outputs[FieldHeadNames.DENSITY]

        if self.initialize_density:
            pos = ray_samples.frustums.get_positions()
            density_blob = (
                self.density_strength
                * torch.exp(-torch.norm(pos, dim=-1) / (2 * 0.04))[..., None]
            )
            density = density + density_blob

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples)
        features = self.renderer_feature(
            features=field_outputs[FieldHeadNames.FEATURE], weights=weights
        )

        accum_mask = torch.clamp(
            (torch.nan_to_num(accumulation, nan=0.0)), min=0.0, max=1.0
        )
        accum_mask_inv = 1.0 - accum_mask

        background = accum_mask_inv * background_feature

        outputs = {
            "feature_only": features,
            "background_feature": background_feature,
            "background": background,
            "accumulation": accum_mask,
            "depth": depth,
        }

        latents = accum_mask * features + background
        outputs["latents"] = latents

        if ray_dims:
            height = ray_dims[0]
            width = ray_dims[1]
            # print(height, width)

            latents_input = latents.view(1, height, width, 4).half()
            # latents_input = features.view(1, height, width, 4).half()
            latents_input = latents_input.permute(0, 3, 1, 2)
            # latents_input = torch.nn.functional.interpolate(latents_input, size=(latents_input.shape[2] // 8, latents_input.shape[3] // 8), mode='bilinear')
            rgb = self.sd.latents_to_img(latents_input)  # .permute(0, 2, 3, 1)[0]
            # rgb = torch.nn.functional.interpolate(
            #     rgb, size=(height, width), mode="bilinear"
            # )

            # ADD WHITE BG TO RGB
            white_bg = torch.tensor([1.0, 1.0, 1.0], device=self.device)

            a_mask = accum_mask.reshape(1, 1, height, width)

            a_mask = torch.nn.functional.interpolate(
                a_mask, size=(height * 8, width * 8), mode="bilinear"
            )

            a_mask = torch.flatten(a_mask.permute(0, 2, 3, 1), end_dim=-2)
            a_mask_inv = 1 - a_mask

            rgb = rgb.permute(0, 2, 3, 1)[0]
            rgb = torch.flatten(rgb, end_dim=-2)
            rgb = a_mask * rgb + (a_mask_inv * white_bg)
            rgb = rgb.to(torch.float32)
            # print("rgb", rgb)
            # print(rgb.shape)

            outputs["rgb"] = rgb

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        normals = self.renderer_normals(
            normals=field_outputs[FieldHeadNames.NORMALS], weights=weights
        )
        outputs["normals"] = self.shader_normals(normals, weights=accum_mask)

        outputs["rendered_orientation_loss"] = orientation_loss(
            weights.detach(),
            field_outputs[FieldHeadNames.NORMALS],
            ray_bundle.directions,
        )

        assert weights.shape[-1] == 1
        if self.config.opacity_penalty:
            outputs["opacity_loss"] = (
                torch.sqrt(torch.sum(weights, dim=-2) ** 2 + 0.01)
                * self.config.opacity_loss_mult
            )

        return outputs

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.

        loss_dict = {}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        if self.train_normals:
            # orientation loss for computed normals
            loss_dict[
                "orientation_loss"
            ] = self.config.orientation_loss_mult * torch.mean(
                outputs["rendered_orientation_loss"]
            )
        else:
            loss_dict["orientation_loss"] = 0

        if self.config.opacity_penalty:
            loss_dict["opacity_loss"] = (
                self.config.opacity_loss_mult * outputs["opacity_loss"].mean()
            )

        if self.prompt != self.cur_prompt:
            self.cur_prompt = self.prompt
            self.text_embeddings.update_prompt(
                base_prompt=self.cur_prompt,
                top_prompt=self.cur_prompt + self.top_prompt,
                side_prompt=self.cur_prompt + self.side_prompt,
                back_prompt=self.cur_prompt + self.back_prompt,
                front_prompt=self.cur_prompt + self.front_prompt,
            )

        text_embedding = self.text_embeddings.get_text_embedding(
            vertical_angle=batch["vertical"], horizontal_angle=batch["central"]
        )

        latents = (
            outputs["latents"]
            .view(
                1,
                int(outputs["latents"].shape[0] ** 0.5),
                int(outputs["latents"].shape[0] ** 0.5),
                4,
            )
            .permute(0, 3, 1, 2)
        )

        sds_loss = self.sd.sds_loss_latent(
            text_embedding.to(self.sd_device),
            latents.to(self.sd_device),
            guidance_scale=int(self.guidance_scale),
            grad_scaler=self.grad_scaler,
        )

        loss_dict["sds_loss"] = sds_loss.to(self.device)

        if self.training:
            loss_dict[
                "distortion_loss"
            ] = self.config.distortion_loss_mult * distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            loss_dict[
                "interlevel_loss"
            ] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        return loss_dict

    def forward(
        self, ray_bundle: RayBundle, ray_dims: Tuple = ()
    ) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, ray_dims)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(
                start_idx, end_idx
            )
            outputs = self.forward(
                ray_bundle=ray_bundle, ray_dims=(image_height, image_width)
            )
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            if output_name == "rgb":
                outputs[output_name] = torch.cat(outputs_list).view(image_height * 8, image_width * 8, -1)  # type: ignore
            else:
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            # outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            # print(output_name, outputs[output_name].shape)

        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        prop_depth_0 = colormaps.apply_depth_colormap(
            outputs["prop_depth_0"],
            accumulation=outputs["accumulation"],
        )
        prop_depth_1 = colormaps.apply_depth_colormap(
            outputs["prop_depth_1"],
            accumulation=outputs["accumulation"],
        )

        size = outputs["latents"].shape[0]

        latents_input = outputs["latents"].view(1, size, size, 4).half()
        latents_input = latents_input.permute(0, 3, 1, 2)
        rgb = self.sd.latents_to_img(latents_input)
        rgb = rgb.permute(0, 2, 3, 1)[0].reshape(512, 512, 3).cpu().to(torch.float32)

        metrics_dict = {}
        images_dict = {
            "img": rgb,
            "accumulation": acc,
            "depth": depth,
            "prop_depth_0": prop_depth_0,
            "prop_depth_1": prop_depth_1,
            "normals": outputs["normals"],
        }
        return metrics_dict, images_dict
