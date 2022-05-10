"""
Implementation of vanilla nerf.
"""


from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import Parameter

from mattport.nerf import metrics
from mattport.nerf.field_modules.encoding import NeRFEncoding
from mattport.nerf.field_modules.field_heads import DensityFieldHead, FieldHeadNames, RGBFieldHead
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.graph.base import Graph
from mattport.nerf.loss import MSELoss
from mattport.nerf.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from mattport.nerf.sampler import PDFSampler, UniformSampler
from mattport.structures import colors
from mattport.structures.rays import RayBundle, RaySamples
from mattport.utils import stats_tracker, visualization, writer


class NeRFField(nn.Module):
    """NeRF module"""

    def __init__(self, num_layers=8, layer_width=256, skip_connections: Tuple = (4,)) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections

        self.encoding_xyz = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.encoding_dir = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )
        self.mlp_base = MLP(
            in_dim=self.encoding_xyz.get_out_dim(),
            out_dim=self.layer_width,
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            skip_connections=self.skip_connections,
            activation=nn.ReLU(),
        )
        self.mlp_rgb = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.encoding_dir.get_out_dim(),
            out_dim=self.layer_width // 2,
            num_layers=2,
            layer_width=self.layer_width // 2,
            activation=nn.ReLU(),
        )
        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_rgb.get_out_dim())
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at points along the ray
        Args:
            xyz: ()
        # TODO(ethan): change the input to be something more abstracted
        e.g., a FieldInput structure
        """
        positions = ray_samples.positions
        directions = ray_samples.directions
        encoded_xyz = self.encoding_xyz(positions)
        encoded_dir = self.encoding_dir(directions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        rgb_mlp_out = self.mlp_rgb(torch.cat([encoded_dir, base_mlp_out], dim=-1))

        field_rgb_output = self.field_output_rgb(rgb_mlp_out)
        field_density_out = self.field_output_density(base_mlp_out)

        field_outputs = {}
        field_outputs.update(field_rgb_output)
        field_outputs.update(field_density_out)
        return field_outputs


class NeRFGraph(Graph):
    """Vanilla NeRF graph"""

    def __init__(
        self,
        intrinsics=None,
        camera_to_world=None,
        near_plane=2.0,
        far_plane=6.0,
        num_coarse_samples=64,
        num_importance_samples=128,
        **kwargs,
    ) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.num_coarse_samples = num_coarse_samples
        self.num_importance_samples = num_importance_samples
        self.field_coarse = None
        self.field_fine = None
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_fields(self):
        """Set the fields."""
        self.field_coarse = NeRFField()
        self.field_fine = NeRFField()

    def populate_modules(self):

        # samplers
        self.sampler_uniform = UniformSampler(
            near_plane=self.near_plane, far_plane=self.far_plane, num_samples=self.num_coarse_samples
        )
        self.sampler_pdf = PDFSampler(num_samples=self.num_importance_samples)

        # field
        self.populate_fields()

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Dict[str, List[Parameter]]: Mapping of different parameter groups
        """
        param_groups = {}
        # param_groups["cameras"] = list(self.ray_generator.parameters())
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # coarse network:
        uniform_ray_samples = self.sampler_uniform(ray_bundle)  # RaySamples

        # time coarse network # xyz -> asd
        coarse_field_outputs = self.field_coarse(uniform_ray_samples)  # FieldOutputs
        # time end coarse network

        coarse_weights = uniform_ray_samples.get_weights(coarse_field_outputs[FieldHeadNames.DENSITY])

        coarse_renderer_outputs = self.renderer_rgb(
            rgb=coarse_field_outputs[FieldHeadNames.RGB],
            weights=coarse_weights,
        )  # RendererOutputs
        coarse_renderer_accumulation = self.renderer_accumulation(coarse_weights)  # RendererOutputs
        coarse_renderer_depth = self.renderer_depth(coarse_weights, uniform_ray_samples.ts)

        # fine network:
        pdf_ray_samples = self.sampler_pdf(uniform_ray_samples, coarse_weights)  # RaySamples
        fine_field_outputs = self.field_fine(pdf_ray_samples)  # FieldOutputs

        fine_weights = pdf_ray_samples.get_weights(fine_field_outputs[FieldHeadNames.DENSITY])

        fine_renderer_outputs = self.renderer_rgb(
            rgb=fine_field_outputs[FieldHeadNames.RGB],
            weights=fine_weights,
        )  # RendererOutputs
        fine_renderer_accumulation = self.renderer_accumulation(fine_weights)  # RendererOutputs
        fine_renderer_depth = self.renderer_depth(fine_weights, pdf_ray_samples.ts)

        # TODO refactor this into "vis" section. Doesn't need to be run during training.
        coarse_renderer_depth = visualization.apply_depth_colormap(
            coarse_renderer_depth.depth,
            accumulation=coarse_renderer_accumulation.accumulation,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )
        fine_renderer_depth = visualization.apply_depth_colormap(
            fine_renderer_depth.depth,
            accumulation=fine_renderer_accumulation.accumulation,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )
        coarse_renderer_accumulation = visualization.apply_colormap(coarse_renderer_accumulation.accumulation)
        fine_renderer_accumulation = visualization.apply_colormap(fine_renderer_accumulation.accumulation)

        # outputs:
        outputs = {
            "rgb_coarse": coarse_renderer_outputs.rgb,
            "rgb_fine": fine_renderer_outputs.rgb,
            "accumulation_coarse": coarse_renderer_accumulation,
            "accumulation_fine": fine_renderer_accumulation,
            "depth_coarse": coarse_renderer_depth,
            "depth_fine": fine_renderer_depth,
        }
        return outputs

    def get_loss_dict(self, outputs, batch):
        device = outputs["rgb_coarse"].device
        pixels = batch["pixels"].to(device)
        rgb_loss_coarse = self.rgb_loss(pixels, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(pixels, outputs["rgb_fine"])
        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict["aggregated_loss"] = self.get_aggregated_loss_from_loss_dict(loss_dict)
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, image, outputs):
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        accumulation_coarse = outputs["accumulation_coarse"]
        accumulation_fine = outputs["accumulation_fine"]
        depth_coarse = outputs["depth_coarse"]
        depth_fine = outputs["depth_fine"]

        combined_image = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        writer.write_event(
            {"name": f"image_idx_{image_idx}-rgb_coarse_fine", "x": combined_image, "step": step, "group": "val_img"}
        )

        combined_image = torch.cat([accumulation_coarse, accumulation_fine], dim=1)
        writer.write_event(
            {"name": f"image_idx_{image_idx}", "x": combined_image, "step": step, "group": "val_accumulation"}
        )

        combined_image = torch.cat([depth_coarse, depth_fine], dim=1)
        writer.write_event({"name": f"image_idx_{image_idx}", "x": combined_image, "step": step, "group": "val_depth"})

        coarse_psnr = metrics.get_psnr(image, rgb_coarse)
        writer.write_event(
            {"name": f"image_idx_{image_idx}", "scalar": float(coarse_psnr), "step": step, "group": "val"}
        )

        coarse_ssim = metrics.get_ssim(image, rgb_coarse)
        writer.write_event(
            {"name": f"image_idx_{image_idx}", "scalar": float(coarse_ssim), "step": step, "group": "ssim"}
        )

        fine_psnr = metrics.get_psnr(image, rgb_fine)
        stats_tracker.update_stats(
            {"name": stats_tracker.Stats.CURR_TEST_PSNR, "value": float(fine_psnr), "step": step}
        )
        writer.write_event(
            {"name": f"image_idx_{image_idx}-fine_psnr", "scalar": float(fine_psnr), "step": step, "group": "val"}
        )

        fine_ssim = metrics.get_ssim(image, rgb_fine)
        writer.write_event(
            {"name": f"image_idx_{image_idx}-fine_ssim", "scalar": float(fine_ssim), "step": step, "group": "ssim"}
        )
