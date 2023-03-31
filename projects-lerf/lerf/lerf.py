from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import open_clip
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

from lerf.encoders.image_encoder import BaseImageEncoder
from lerf.lerf_field import LERFField
from lerf.lerf_renderers import CLIPRenderer, MeanRenderer
from lerf.lerf_fieldheadnames import LERFFieldHeadNames


@dataclass
class LERFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LERFModel)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    specify_scale: bool = False
    num_lerf_samples: int = 12


class LERFModel(NerfactoModel):
    config: LERFModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.lerf_field = LERFField(clip_n_dims=self.image_encoder.embedding_dim)

    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if self.config.specify_scale:
            assert preset_scales is not None
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        for s, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            for i in range(n_phrases):
                probs = self.image_encoder.get_relevancy(clip_output, i)
                pos_prob = probs[..., 0:1]
                if n_phrases_maxs[i] is None or pos_prob.max() > n_phrases_sims[i].max():
                    n_phrases_maxs[i] = s
                    n_phrases_sims[i] = pos_prob
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        if self.training:
            clip_scales = ray_bundle.metadata["clip_scales"]
            clip_scales = clip_scales[..., None]
            dist = lerf_samples.spacing_to_euclidean_fn(lerf_samples.spacing_starts.squeeze(-1)).unsqueeze(-1)
            clip_scales = clip_scales * ray_bundle.metadata["width"] * (1 / ray_bundle.metadata["fx"]) * dist
        else:
            clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples, clip_scales)

        if self.training:
            outputs["clip"] = self.renderer_clip(
                embeds=lerf_field_outputs[LERFFieldHeadNames.CLIP], weights=lerf_weights.detach()
            )
            outputs["dino"] = self.renderer_mean(
                embeds=lerf_field_outputs[LERFFieldHeadNames.DINO], weights=lerf_weights.detach()
            )

        if not self.training:
            max_across, best_scales = self.get_max_across(
                lerf_samples, lerf_weights, lerf_field_outputs[LERFFieldHeadNames.HASHGRID], clip_scales.shape
            )
            multiphrase = max_across[0]
            # normalization for sanity check TODO(cmk) remove
            multiphrase[multiphrase < 0.5] = 0.5
            multiphrase -= 0.5
            multiphrase = multiphrase / multiphrase.max()
            outputs[f"multiphrase"] = multiphrase

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        LERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
        which are not independent since they need to use the same scale
        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        # TODO(justin) implement max across behavior
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        loss_dict["clip_loss"] = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
            outputs["clip"], batch["clip"], delta=1.25
        )
        loss_dict["dino_loss"] = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"])
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["lerf"] = list(self.lerf_field.parameters())
        return param_groups
