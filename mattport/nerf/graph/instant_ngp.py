"""
Implementation of Instant NGP.
"""

from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mattport.nerf.colliders import AABBBoxCollider
from mattport.nerf.field_modules.field_heads import FieldHeadNames
from mattport.nerf.fields.instant_ngp_field import field_implementation_to_class
from mattport.nerf.graph.base import Graph
from mattport.nerf.loss import MSELoss
from mattport.nerf.occupancy_grid import OccupancyGrid
from mattport.nerf.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from mattport.nerf.sampler import PDFSampler, UniformSampler
from mattport.structures import colors
from mattport.structures.rays import RayBundle
from mattport.utils import visualization, writer


class NGPGraph(Graph):

    """NeRF-W graph"""

    def __init__(self, field_implementation="torch", intrinsics=None, camera_to_world=None, **kwargs) -> None:
        assert field_implementation in field_implementation_to_class
        self.field_implementation = field_implementation
        self.field = None
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_collider(self):
        self.collider = AABBBoxCollider(self.scene_bounds)

    def populate_fields(self):
        """Set the fields."""
        # torch or tiny-cuda-nn version
        self.field = field_implementation_to_class[self.field_implementation](self.scene_bounds.aabb)

    def populate_misc_modules(self):
        # occupancy grid
        self.occupancy_grid = OccupancyGrid(aabb=self.scene_bounds.aabb)

        # samplers
        self.sampler_occupancy_grid = UniformSampler(num_samples=128, occupancy_field=self.occupancy_grid)
        # NOTE(ethan): are we sure we want the include_original flag used like this?
        # it could be easily forgotten that it's by default True...?
        self.sampler_pdf = PDFSampler(num_samples=128, include_original=False)

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
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        # uniform sampling
        ray_samples_uniform = self.sampler_occupancy_grid(ray_bundle)
        field_outputs_uniform = self.field.forward(ray_samples_uniform.to_point_samples())
        weights_uniform = ray_samples_uniform.get_weights(field_outputs_uniform[FieldHeadNames.DENSITY])

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_uniform)
        field_outputs_pdf = self.field.forward(ray_samples_pdf.to_point_samples())

        ts, indices = torch.sort(torch.cat([ray_samples_uniform.ts, ray_samples_pdf.ts], -1), -1)
        ray_samples = ray_bundle.get_ray_samples(ts)
        field_outputs = {}
        for fo_name, _ in field_outputs_pdf.items():
            fo_uniform = field_outputs_uniform[fo_name]
            fo_pdf = field_outputs_pdf[fo_name]
            fo_uniform_pdf = torch.cat([fo_uniform, fo_pdf], 1)
            index = indices.view(fo_uniform_pdf[..., :1].shape)
            index = index.expand(-1, -1, fo_uniform_pdf.shape[-1])  # TODO: don't hardcode this
            field_outputs[fo_name] = torch.gather(fo_uniform_pdf, dim=1, index=index)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
        )
        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples.ts)

        densities_occupancy_grid = self.occupancy_grid.get_densities(ray_samples.positions)
        weights_occupancy_grid = ray_samples.get_weights(densities_occupancy_grid)
        depth_occupancy_grid = self.renderer_depth(weights_occupancy_grid, ray_samples.ts)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "depth_occupancy_grid": depth_occupancy_grid,
        }
        return outputs

    def get_loss_dict(self, outputs, batch):
        device = self.get_device()
        pixels = batch["pixels"].to(device)
        rgb_loss = self.rgb_loss(pixels, outputs["rgb"])
        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict["aggregated_loss"] = self.get_aggregated_loss_from_loss_dict(loss_dict)
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, image, mask, outputs):
        rgb = outputs["rgb"]
        acc = visualization.apply_colormap(outputs["accumulation"])
        depth = visualization.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        writer.put_image(name=f"image_idx_{image_idx}", image=combined_rgb, step=step, group="img")
        writer.put_image(name=f"image_idx_{image_idx}", image=combined_acc, step=step, group="accumulation")
        writer.put_image(name=f"image_idx_{image_idx}", image=combined_depth, step=step, group="depth")

        depth = visualization.apply_depth_colormap(outputs["depth_occupancy_grid"])
        writer.put_image(name=f"image_idx_{image_idx}", image=combined_depth, step=step, group="depth_occupancy_grid")

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        writer.put_scalar(name=f"val_{image_idx}-fine", scalar=float(psnr), step=step, group="psnr")
        writer.put_scalar(name=f"val_{image_idx}", scalar=float(ssim), step=step, group="ssim")
        writer.put_scalar(name=f"val_{image_idx}", scalar=float(lpips), step=step, group="lpips")

        return psnr.item()
