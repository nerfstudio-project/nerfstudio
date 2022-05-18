"""
Semantic NeRF implementation.
"""

from typing import Tuple

import torch
from torch import nn

from mattport.nerf.field_modules.field_heads import FieldHeadNames, SemanticStuffHead
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.graph.vanilla_nerf import NeRFField, NeRFGraph
from mattport.nerf.renderers import SemanticRenderer
from mattport.structures.rays import RayBundle, RaySamples
from mattport.utils import visualization, writer


class SemanticNerfField(NeRFField):
    """Semantic-NeRF field"""

    def __init__(
        self, num_layers=8, layer_width=256, skip_connections: Tuple = (4,), num_stuff_classes: int = None
    ) -> None:
        assert num_stuff_classes is not None
        self.num_stuff_classes = num_stuff_classes
        super().__init__(num_layers=num_layers, layer_width=layer_width, skip_connections=skip_connections)
        self.semantic_mlp = MLP(
            in_dim=self.mlp_base.get_out_dim(),
            out_dim=self.layer_width // 2,
            num_layers=1,
            activation=nn.ReLU(),
        )
        self.field_output_semantic = SemanticStuffHead(
            in_dim=self.semantic_mlp.get_out_dim(), num_classes=num_stuff_classes
        )
        self.semantic_sequential = torch.nn.Sequential(self.semantic_mlp, self.field_output_semantic)

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at points along the ray."""
        positions = ray_samples.positions
        directions = ray_samples.directions
        encoded_xyz = self.encoding_xyz(positions)
        encoded_dir = self.encoding_dir(directions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        rgb_mlp_out = self.mlp_rgb(torch.cat([encoded_dir, base_mlp_out], dim=-1))

        field_rgb_output = self.field_output_rgb(rgb_mlp_out)
        field_density_out = self.field_output_density(base_mlp_out)
        field_semantic_output = self.semantic_sequential(base_mlp_out)

        field_outputs = {}
        field_outputs.update(field_rgb_output)
        field_outputs.update(field_density_out)
        field_outputs.update(field_semantic_output)
        return field_outputs


class SemanticNerfGraph(NeRFGraph):
    """Semantic-NeRF graph"""

    def __init__(self, intrinsics=None, camera_to_world=None, stuff_classes: int = None, **kwargs) -> None:
        assert stuff_classes is not None
        self.stuff_classes = stuff_classes
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)
        self.renderer_semantic = SemanticRenderer()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")

    def populate_fields(self):
        """Set the fields."""
        self.field_coarse = NeRFField()
        self.field_fine = SemanticNerfField(num_stuff_classes=len(self.stuff_classes))

    def get_outputs(self, ray_bundle: RayBundle):
        # coarse network:
        uniform_ray_samples = self.sampler_uniform(ray_bundle)  # RaySamples

        coarse_field_outputs = self.field_coarse(uniform_ray_samples)  # FieldOutputs

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

        semantic_logits = self.renderer_semantic(
            fine_field_outputs[FieldHeadNames.SEMANTICS_STUFF], weights=fine_weights
        ).semantics

        # outputs:
        outputs = {
            "rgb_coarse": coarse_renderer_outputs.rgb,
            "rgb_fine": fine_renderer_outputs.rgb,
            "accumulation_coarse": coarse_renderer_accumulation,
            "accumulation_fine": fine_renderer_accumulation,
            "depth_coarse": coarse_renderer_depth,
            "depth_fine": fine_renderer_depth,
            "semantic_logits": semantic_logits,
        }
        return outputs

    def get_loss_dict(self, outputs, batch):
        # TODO(ethan): batch has "stuff_image". use it
        device = outputs["rgb_coarse"].device
        pixels = batch["pixels"].to(device)
        rgb_loss_coarse = self.rgb_loss(pixels, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(pixels, outputs["rgb_fine"])

        semantic_logits = outputs["semantic_logits"]
        semantic_classes = batch["stuff_image"][..., 0].to(device).long()
        semantic_loss_fine = self.cross_entropy_loss(semantic_logits, semantic_classes)
        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine,
            "semantic_loss_fine": semantic_loss_fine,
        }

        loss_dict["aggregated_loss"] = self.get_aggregated_loss_from_loss_dict(loss_dict)
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, image, outputs):
        super().log_test_image_outputs(image_idx, step, image, outputs)
        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantic_logits"], dim=-1), dim=-1) / len(
            self.stuff_classes
        )
        semantic_labels_image = visualization.apply_colormap(semantic_labels[..., None])
        writer.write_image(name="image_idx_{image_idx}", image=semantic_labels_image, step=step, group="semantics")
