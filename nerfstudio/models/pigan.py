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
Implementation of pi-gan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.pigan_field import PiganField
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.latent import FilmLatent
from nerfstudio.model_components.losses import GANLoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.discriminator import Discriminator
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc


# FIXME - config
@dataclass
class GenerativeModelConfig(ModelConfig):
    """3D-Aware GAN Config"""

    _target: Type = field(default_factory=lambda: PiganModel)
    # enable_collider: bool = True
    """Whether to create a scene collider to filter rays.""" # module with near and far setting for rays

    #FIXME - need to be pigan loss ( genreator loss & discriminator loss )
    # loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """parameters to instantiate density field with"""

    #FIXME - delete this config and need to be per image eval option
    # eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""

    #NOTE - If do hierarchical sampling, need to set below options 
    num_coarse_samples: int = 128
    """Number of samples in coarse field evaluation"""
    # num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""

    z_dim: int = 100
    """Latent Code Z dimension"""
    num_mapping_network_layers: int = 4
    """NUmber of mapping network layers"""
    mapping_network_hiddem_dim: int = 256
    """Hidden dimension of mapping network"""

    # discriminator model config



class PiganModel(Model):
    """Pi-gan model

    Args:
        config : Basic Pi-gan configuration to instantiate mdoel
    """

    def __init__(
        self,
        config : GenerativeModelConfig, #ANCHOR
        **kwargs,
    ) -> None:


        super().__init__(
            config=config,
            **kwargs,
        )
    
    def populate_modules(self):
        """Set the generator(field), discriminator and modules """
        super().populate_modules()

        # Generator ( filed )
        # not use positional encoding but use siren activation
        self.field_coarse = PiganField()

        self.field_fine = PiganField()

        #FIXME -  Discriminator F
        self.discriminator = Discriminator(128)

        # mapping network - parameter same with https://github.com/marcoamonteiro/pi-GAN
        #NOTE - 여러개의 frequency와 shifts가 필요한 듯 - pigan github 참고.
        self.mapping_network = MLP(
            in_dim=self.config.z_dim,
            num_layers=self.config.num_mapping_network_layers, 
            layer_width=self.config.mapping_network_hiddem_dim, 
            out_dim=(getattr(self.field_coarse,'base_mlp_num_layers')+1)*self.config.mapping_network_hiddem_dim*2,
        ) # output의 절반은 frqencies, 뒤 절반은 shifts여야함.

        # latent type
        self.latent = FilmLatent

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        # self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.generator_loss = GANLoss(m_type='G')
        self.discriminator_loss = GANLoss(m_type='D')

        # metrics
        self.fid = FrechetInceptionDistance()
        self.inception = InceptionScore()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
    
        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)
        

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        # if self.temporal_distortion is not None:
        #     param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())

        param_groups["discriminator"] = list(self.discriminator.parameters())
        param_groups["mapping_network"] = list(self.mapping_network.parameters())
        return param_groups


    def get_outputs(self, ray_bundle: RayBundle, latent :  TensorType['batch','z_dim']): 

        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # setting freq & phase_shift
        freq_offsets = self.mapping_network(latent)
        freq = freq_offsets[..., :freq_offsets.shape[-1]//2] * (self.field_coarse.siren_omega/2) + self.field_coarse.siren_omega
        phase_shifts = freq_offsets[..., freq_offsets.shape[-1]//2:]

        latents = self.latent(freq = freq,phase_shift = phase_shift)
    
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        # if self.temporal_distortion is not None:
        #     offsets = self.temporal_distortion(ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times)
        #     ray_samples_uniform.frustums.set_offsets(offsets)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform, latents)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # # pdf sampling
        # ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        # if self.temporal_distortion is not None:
        #     offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
        #     ray_samples_pdf.frustums.set_offsets(offsets)

        # # fine field:
        # field_outputs_fine = self.field_fine.forward(ray_samples_pdf, latents)
        # weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        # rgb_fine = self.renderer_rgb(
        #     rgb=field_outputs_fine[FieldHeadNames.RGB],
        #     weights=weights_fine,
        # )
        # accumulation_fine = self.renderer_accumulation(weights_fine)
        # depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        # FIXME - pigan concat coarse and fine for all rgb output
        # rgb_all = self.renderer_rgb(
        #     rgb=torch.cat([field_outputs_coarse[FieldHeadNames.RGB], field_outputs_fine[FieldHeadNames.RGB]], dim = -2)
        # )

        outputs = {
            "rgb_coarse": rgb_coarse, 
            "accumulation_coarse": accumulation_coarse,
            "depth_coarse": depth_coarse,
        }
        return outputs

    def forward(self, ray_bundle: RayBundle, latent :  TensorType['batch','z_dim']) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
 
        num_rays_per_chunk = self.config.train_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)

        ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(0, num_rays)
        outputs = self.get_outputs(ray_bundle=ray_bundle) # .view(image_height, image_width, -1) 
        # for i in range(0, num_rays, num_rays_per_chunk):
        #     start_idx = i
        #     end_idx = i + num_rays_per_chunk
            # ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # outputs = self.get_outputs(ray_bundle=ray_bundle)
        #     for output_name, output in outputs.items():  # type: ignore
        #         outputs_lists[output_name].append(output)
        # outputs = {}
        # for output_name, outputs_list in outputs_lists.items():
        #     if not torch.is_tensor(outputs_list[0]):
        #         # TODO: handle lists of tensors as well
        #         continue
        #     outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

        # return self.get_outputs(ray_bundle, latent)

    #REVIEW 
    def get_discriminator(self, model_outputs, batch=None):
        fake_pred = self.discriminator(model_outputs)
        if batch is not None:
            real_pred = self.discriminator(batch['image'])
            return fake_pred, real_pred
        else:
            return fake_pred

    def get_loss_dict(self, preds, batch, step : int, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = preds[0].device
        image = batch["image"].to(device)

        # type g
        if len(preds)==1:
            g_loss = self.generator_loss(preds, step)
            loss_dict = {"generator_loss": g_loss}
        elif len(preds)==2:
            d_loss = self.discriminator_loss(preds, step)
            loss_dict = {"discriminator_loss": d_loss}
        return loss_dict
    
    def freeze_discriminator(self):   
        field_parameters = list(self.discriminator.parameters())
        self.requires_grad(field_parameters, False)

    def freeze_field(self):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        
        field_parameters = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        self.requires_grad(field_parameters, False)

    def unlock_discriminator(self):   
        field_parameters = list(self.discriminator.parameters())
        self.requires_grad(field_parameters, True)

    def unlock_field(self):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        
        field_parameters = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        self.requires_grad(field_parameters, True)

    def requires_grad(self, paramters, flag):
        for paramter in parameters:
            parameter.requires_grad = flag
