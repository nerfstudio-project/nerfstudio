"""
NeRF-W (NeRF in the wild) implementation.
TODO:
"""

from typing import Tuple

import torch
from torch import nn

from mattport.nerf.field_modules.embedding import Embedding
from mattport.nerf.field_modules.field_heads import (
    FieldHeadNames,
    TransientDensityHead,
    TransientRGBHead,
    UncertaintyFieldHead,
)
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.graph.vanilla_nerf import NeRFField, NeRFGraph
from mattport.nerf.renderers import UncertaintyRenderer
from mattport.structures.rays import RayBundle, RaySamples
from mattport.utils import visualization, writer


class NerfWField(NeRFField):
    """The NeRF-W field which has appearance and transient conditioning."""

    def __init__(
        self,
        num_layers=8,
        layer_width=256,
        skip_connections: Tuple = (4,),
        num_images: int = 0,
        appearance_embedding_dim: int = 48,
        transient_embedding_dim: int = 16,
    ) -> None:
        assert num_images > 0
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.transient_embedding_dim = transient_embedding_dim
        super().__init__(num_layers, layer_width, skip_connections)

    def build_mlp_base(self):
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
        super().build_mlp_base()
        self.mlp_transient = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.embedding_transient.get_out_dim(),
            out_dim=self.layer_width // 2,
            num_layers=4,
            layer_width=self.layer_width,
            activation=nn.ReLU(),
        )

    def build_mlp_rgb(self):
        self.mlp_rgb = MLP(
            in_dim=self.mlp_base.get_out_dim()
            + self.encoding_dir.get_out_dim()
            + self.embedding_appearance.get_out_dim(),
            out_dim=self.layer_width // 2,
            num_layers=1,
            activation=nn.ReLU(),
        )

    def build_heads(self):
        super().build_heads()
        self.field_output_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.get_out_dim())
        self.field_output_transient_rgb = TransientRGBHead(in_dim=self.mlp_transient.get_out_dim())
        self.field_output_transient_density = TransientDensityHead(in_dim=self.mlp_transient.get_out_dim())

    def forward(self, ray_samples: RaySamples):
        positions = ray_samples.positions
        directions = ray_samples.directions
        camera_indices = ray_samples.get_camera_indices()
        encoded_xyz = self.encoding_xyz(positions)
        encoded_dir = self.encoding_dir(directions)
        embedded_appearance = self.embedding_appearance(camera_indices)
        embedded_transient = self.embedding_transient(camera_indices)

        base_mlp_out = self.mlp_base(encoded_xyz)
        rgb_mlp_out = self.mlp_rgb(torch.cat([base_mlp_out, encoded_dir, embedded_appearance], dim=-1))
        transient_mlp_out = self.mlp_transient(torch.cat([base_mlp_out, embedded_transient], dim=-1))

        field_rgb_output = self.field_output_rgb(rgb_mlp_out)
        field_density_out = self.field_output_density(base_mlp_out)
        field_transient_uncertainty_out = self.field_output_transient_uncertainty(transient_mlp_out)
        field_transient_rgb_out = self.field_output_transient_rgb(transient_mlp_out)
        field_transient_density_out = self.field_output_transient_density(transient_mlp_out)

        field_outputs = {}
        field_outputs.update(field_rgb_output)
        field_outputs.update(field_density_out)
        field_outputs.update(field_transient_uncertainty_out)
        field_outputs.update(field_transient_rgb_out)
        field_outputs.update(field_transient_density_out)
        return field_outputs


class NerfWGraph(NeRFGraph):
    """NeRF-W graph"""

    def __init__(self, intrinsics=None, camera_to_world=None, **kwargs) -> None:
        self.num_images = len(intrinsics)
        self.appearance_embedding_dim = 48
        self.transient_embedding_dim = 16
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)
        self.renderer_uncertainty = UncertaintyRenderer()

    def populate_fields(self):
        """Set the fields."""
        self.field_coarse = NeRFField()
        self.field_fine = NerfWField(
            num_images=self.num_images,
            appearance_embedding_dim=self.appearance_embedding_dim,
            transient_embedding_dim=self.transient_embedding_dim,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        # coarse network
        uniform_ray_samples = self.sampler_uniform(ray_bundle)
        coarse_field_outputs = self.field_coarse(uniform_ray_samples)
        coarse_weights = uniform_ray_samples.get_weights(coarse_field_outputs[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=coarse_field_outputs[FieldHeadNames.RGB],
            weights=coarse_weights,
        ).rgb
        depth_coarse = self.renderer_depth(coarse_weights, uniform_ray_samples.ts).depth

        # fine network
        pdf_ray_samples = self.sampler_pdf(uniform_ray_samples, coarse_weights)
        fine_field_outputs = self.field_fine(pdf_ray_samples)
        fine_weights = pdf_ray_samples.get_weights(
            fine_field_outputs[FieldHeadNames.DENSITY] + fine_field_outputs[FieldHeadNames.TRANSIENT_DENSITY]
        )
        fine_weights_static = pdf_ray_samples.get_weights(fine_field_outputs[FieldHeadNames.DENSITY])
        fine_weights_transient = pdf_ray_samples.get_weights(fine_field_outputs[FieldHeadNames.TRANSIENT_DENSITY])

        rgb_fine_static_component = self.renderer_rgb(
            rgb=fine_field_outputs[FieldHeadNames.RGB],
            weights=fine_weights,
        ).rgb
        rgb_fine_transient_component = self.renderer_rgb(
            rgb=fine_field_outputs[FieldHeadNames.TRANSIENT_RGB],
            weights=fine_weights,
        ).rgb
        rgb_fine = rgb_fine_static_component + rgb_fine_transient_component

        rgb_fine_static = self.renderer_rgb(
            rgb=fine_field_outputs[FieldHeadNames.RGB],
            weights=fine_weights_static,
        ).rgb

        # fine_renderer_accumulation_static = self.renderer_accumulation(fine_weights_static)
        depth_fine = self.renderer_depth(fine_weights, pdf_ray_samples.ts).depth
        depth_fine_static = self.renderer_depth(fine_weights_static, pdf_ray_samples.ts).depth
        uncertainty = self.renderer_uncertainty(
            fine_field_outputs[FieldHeadNames.UNCERTAINTY], fine_weights_transient
        ).uncertainty

        density_transient = fine_field_outputs[FieldHeadNames.TRANSIENT_DENSITY]

        # outputs:
        outputs = {
            "rgb_coarse": rgb_coarse,  # (num_rays, 3)
            "rgb_fine": rgb_fine,
            "rgb_fine_static": rgb_fine_static,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "depth_fine_static": depth_fine_static,
            "density_transient": density_transient,  # (num_rays, num_samples, 1)
            "uncertainty": uncertainty,  # (num_rays, 1)
        }
        return outputs

    def get_loss_dict(self, outputs, batch):
        device = outputs["rgb_coarse"].device
        pixels = batch["pixels"].to(device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        density_transient = outputs["density_transient"]
        betas = outputs["uncertainty"]
        rgb_loss_coarse = 0.5 * ((pixels - rgb_coarse) ** 2).sum(-1).mean()
        rgb_loss_fine = 0.5 * (((pixels - rgb_fine) ** 2).sum(-1) / (betas[..., 0] ** 2)).mean()
        uncertainty_loss = 0.5 * (3 + torch.log(betas)).mean()
        density_loss = density_transient.mean()

        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine,
            "uncertainty_loss": uncertainty_loss,
            "density_loss": density_loss,
        }
        loss_dict["aggregated_loss"] = self.get_aggregated_loss_from_loss_dict(loss_dict)
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, image, outputs):
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        rgb_fine_static = outputs["rgb_fine_static"]
        depth_coarse = outputs["depth_coarse"]
        depth_fine = outputs["depth_fine"]
        depth_fine_static = outputs["depth_fine_static"]
        uncertainty = outputs["uncertainty"]

        depth_coarse = visualization.apply_depth_colormap(depth_coarse)
        depth_fine = visualization.apply_depth_colormap(depth_fine)
        depth_fine_static = visualization.apply_depth_colormap(depth_fine_static)
        uncertainty = visualization.apply_depth_colormap(uncertainty)

        row0 = torch.cat([image, uncertainty, torch.ones_like(rgb_fine)], dim=-2)
        row1 = torch.cat([rgb_fine, rgb_fine_static, rgb_coarse], dim=-2)
        row2 = torch.cat([depth_fine, depth_fine_static, depth_coarse], dim=-2)
        combined_image = torch.cat([row0, row1, row2], dim=-3)

        writer.write_event(
            {"name": f"image_idx_{image_idx}-nerfw", "x": combined_image, "step": step, "group": "val_img"}
        )
