import torch
from typing import Dict, List, Tuple, Type
from torch.nn import Parameter

import numpy as np
import open_clip
import torch
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction

from nerfstudio.fields.lerf_field import LERFField

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.model_components.renderers import CLIPRenderer, MeanRenderer, DepthRenderer
from nerfstudio.cameras.rays import RayBundle

from typing import Type
from nerfstudio.data.scene_box import SceneBox
from dataclasses import dataclass, field

@dataclass
class LERFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LERFModel)
    clip_loss_weight: float = 1
    n_scales: int = 30
    specify_scale: bool = False

class LERFModel(NerfactoModel):
    config: LERFModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.network = self.kwargs['network']
        self.lerf_field = LERFField(clip_n_dims=self.network.clip_n_dims)

    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales = []):
        # TODO smoothen this out
        if self.config.specify_scale:
            assert len(preset_scales) != 0
            assert len(preset_scales) == len(self.network.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, 1.0, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.network.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        scale_tensor = torch.ones(scales_shape, device=weights.device)
        for s, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(ray_samples, hashgrid_field, scale*scale_tensor)
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            for i in range(n_phrases):
                probs = self.network.get_relevancy(clip_output, i)
                pos_prob = probs[..., 0:1]
                print(scale, pos_prob.min(), pos_prob.max())
                if n_phrases_maxs[i] is None or pos_prob.max() > n_phrases_sims[i].max():
                    n_phrases_maxs[i] = s
                    n_phrases_sims[i] = pos_prob
        
        print(n_phrases_maxs, scales_list, n_phrases_sims[0].min(), n_phrases_sims[0].max())
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        if "clip_scales" in ray_bundle.metadata:
            clip_scales = ray_bundle.metadata["clip_scales"]
            clip_scales = clip_scales[..., None]
            dist = ray_samples.spacing_to_euclidean_fn(ray_samples.spacing_starts.squeeze(-1)).unsqueeze(-1)
            clip_scales = (
                clip_scales * ray_bundle.metadata['width'] * (1/ray_bundle.metadata['fx']) * dist
            )
        else:
            clip_scales = torch.ones_like(ray_samples.spacing_starts, device=self.device)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        lerf_field_outputs = self.lerf_field.get_outputs(ray_samples, clip_scales)
        outputs['clip'] = self.renderer_clip(embeds=lerf_field_outputs[FieldHeadNames.CLIP], weights=weights.detach())
        outputs['dino'] = self.renderer_mean(embeds=lerf_field_outputs[FieldHeadNames.DINO], weights=weights.detach())

        if not "clip_scales" in ray_bundle.metadata:
            max_across, best_scales = self.get_max_across(
                ray_samples, weights, lerf_field_outputs[FieldHeadNames.HASHGRID], clip_scales.shape, [0.111]
            )
            multiphrase = max_across[0]
            print(max_across[0].min(), max_across[0].max())
            # normalization for sanity check TODO(cmk) remove
            multiphrase[multiphrase < 0.5] = 0.5; multiphrase -= 0.5
            multiphrase = multiphrase / multiphrase.max()
            outputs[f"multiphrase"] = multiphrase

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
        param_groups["fields"] = list(self.lerf_field.parameters())+list(self.field.parameters())
        return param_groups
    
