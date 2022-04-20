"""
Implementation of vanilla nerf.
"""


from torchtyping import TensorType

from mattport.nerf.field.nerf import NeRFField
from mattport.nerf.field_modules.ray_generator import RayGenerator
from mattport.nerf.graph.base import Graph
from mattport.nerf.loss import MSELoss
from mattport.nerf.renderers import RGBRenderer
from mattport.nerf.sampler import PDFSampler, UniformSampler


class NeRFGraph(Graph):
    """_summary_

    Args:
        Graph (_type_): _description_
    """

    def __init__(self, intrinsics=None, camera_to_world=None) -> None:
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world)

    def populate_modules(self):
        # ray generator
        self.ray_generator = RayGenerator(self.intrinsics, self.camera_to_world)

        # samplers
        self.sampler_uniform = UniformSampler(near_plane=0.1, far_plane=4.0, num_samples=64)
        self.sampler_pdf = PDFSampler(num_samples=64)

        # field
        self.field_coarse = NeRFField()
        self.field_fine = NeRFField()

        # renderers
        self.renderer_rgb = RGBRenderer()

        # losses
        self.rgb_loss = MSELoss()

    def forward(self, ray_indices: TensorType["num_rays", 3]):
        """Takes in the ray indices and renders out values."""
        # get the rays:
        ray_bundle = self.ray_generator.forward(ray_indices)  # RayBundle
        # coarse network:
        uniform_ray_samples = self.sampler_uniform(ray_bundle)  # RaySamples
        coarse_field_outputs = self.field_coarse(uniform_ray_samples)  # FieldOutputs
        coarse_renderer_outputs = self.renderer_rgb(
            rgb=coarse_field_outputs.rgb, density=coarse_field_outputs.density, deltas=uniform_ray_samples.deltas
        )  # RendererOutputs
        # fine network:
        pdf_ray_samples = self.sampler_pdf(ray_bundle, uniform_ray_samples, coarse_field_outputs.density)  # RaySamples
        fine_field_outputs = self.field_fine(pdf_ray_samples)  # FieldOutputs
        fine_renderer_outputs = self.renderer_rgb(
            rgb=fine_field_outputs.rgb, density=fine_field_outputs.density, deltas=pdf_ray_samples.deltas
        )  # RendererOutputs
        # outputs:
        outputs = {"rgb_coarse": coarse_renderer_outputs.rgb, "rgb_fine": fine_renderer_outputs.rgb}
        return outputs

    def get_losses(self, batch, graph_outputs):
        # batch.pixels # (num_rays, 3)
        losses = {}
        rgb_loss_coarse = self.rgb_loss(batch.pixels, graph_outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(batch.pixels, graph_outputs["rgb_fine"])
        losses = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        return losses
