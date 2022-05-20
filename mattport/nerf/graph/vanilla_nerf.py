"""
Implementation of vanilla nerf.
"""


from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mattport.nerf.field_modules.encoding import NeRFEncoding
from mattport.nerf.field_modules.field_heads import DensityFieldHead, RGBFieldHead
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.fields.base import Field
from mattport.nerf.graph.base import Graph
from mattport.nerf.loss import MSELoss
from mattport.nerf.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from mattport.nerf.sampler import PDFSampler, UniformSampler
from mattport.structures import colors
from mattport.structures.rays import PointSamples, RayBundle
from mattport.nerf.colliders import NearFarCollider
from mattport.utils import visualization, writer
from mattport.utils.misc import is_not_none


class NeRFField(Field):
    """NeRF module"""

    OUTPUT_RGB = "rgb"

    def __init__(self, num_layers=8, layer_width=256, skip_connections: Tuple = (4,)) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections

        self.build_encodings()
        self.build_mlp_base()
        self.build_mlp_rgb()
        self.build_heads()

    def build_encodings(self):
        """Build the encodings."""
        self.encoding_xyz = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.encoding_dir = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

    def build_mlp_base(self):
        """Build the MLP base."""
        self.mlp_base = MLP(
            in_dim=self.encoding_xyz.get_out_dim(),
            out_dim=self.layer_width,
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            skip_connections=self.skip_connections,
            activation=nn.ReLU(),
        )

    def build_mlp_rgb(self):
        """Build the MLP for RGB."""
        self.mlp_rgb = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.encoding_dir.get_out_dim(),
            out_dim=self.layer_width // 2,
            num_layers=2,
            layer_width=self.layer_width // 2,
            activation=nn.ReLU(),
        )

    def build_heads(self):
        """Build the heads."""
        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_rgb.get_out_dim())
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())

    def get_density(self, point_samples: PointSamples):
        """Computes and returns the densities."""
        positions = point_samples.positions
        valid_mask = point_samples.valid_mask
        if not is_not_none(valid_mask):
            valid_mask = torch.ones_like(positions[..., 0]).bool()
        # placeholders for values to return
        density = torch.zeros(*valid_mask.shape, 1, dtype=torch.float32, device=positions.device)
        base_mlp_out = torch.zeros(
            *valid_mask.shape, self.mlp_base.out_dim, dtype=torch.float32, device=positions.device
        )
        if not valid_mask.any():  # empty mask
            return density, base_mlp_out

        encoded_xyz = self.encoding_xyz(positions[valid_mask])
        base_mlp_out[valid_mask] = self.mlp_base(encoded_xyz)
        density[valid_mask] = self.field_output_density(base_mlp_out[valid_mask])
        return density, base_mlp_out

    def get_outputs(self, point_samples: PointSamples, density_embedding=None, valid_mask=None):
        directions = point_samples.directions
        if not is_not_none(valid_mask):
            valid_mask = torch.ones_like(directions[..., 0]).bool()
        # placeholders for values to return
        rgb = torch.zeros(*valid_mask.shape, 3, dtype=torch.float32, device=directions.device)
        if not valid_mask.any():  # empty mask
            return {"rgb": rgb}
        encoded_dir = self.encoding_dir(directions[valid_mask])
        rgb_mlp_out = self.mlp_rgb(torch.cat([encoded_dir, density_embedding[valid_mask]], dim=-1))
        rgb[valid_mask] = self.field_output_rgb(rgb_mlp_out)
        return {"rgb": rgb}


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

    def populate_collider(self):
        self.collider = NearFarCollider(self.near_plane, self.far_plane)

    def populate_fields(self):
        """Set the fields."""
        self.field_coarse = NeRFField()
        self.field_fine = NeRFField()

    def populate_misc_modules(self):
        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform.to_point_samples())
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse["density"])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse["rgb"],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform.ts)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf.to_point_samples())
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine["density"])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine["rgb"],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf.ts)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
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
        acc_coarse = visualization.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = visualization.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = visualization.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )
        depth_fine = visualization.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        writer.put_image(name=f"image_idx_{image_idx}", image=combined_rgb, step=step, group="img")
        writer.put_image(name=f"image_idx_{image_idx}", image=combined_acc, step=step, group="accumulation")
        writer.put_image(name=f"image_idx_{image_idx}", image=combined_depth, step=step, group="depth")

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        writer.put_scalar(name=f"val_{image_idx}-coarse", scalar=float(coarse_psnr), step=step, group="psnr")
        writer.put_scalar(name=f"val_{image_idx}-fine", scalar=float(fine_psnr), step=step, group="psnr")
        writer.put_scalar(name=f"val_{image_idx}", scalar=float(fine_ssim), step=step, group="ssim")
        writer.put_scalar(name=f"val_{image_idx}", scalar=float(fine_lpips), step=step, group="lpips")

        writer.put_scalar(name="Test PSNR", scalar=float(fine_psnr), step=step)

        return fine_psnr.item()
