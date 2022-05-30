"""
The Graph module contains all trainable parameters.
"""
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import Parameter
from torchtyping import TensorType

from radiance.nerf.dataset.structs import SceneBounds
from radiance.nerf.ray_generator import RayGenerator
from radiance.nerf.colliders import AABBBoxCollider
from radiance.structures.cameras import get_camera_model
from radiance.structures.rays import RayBundle
from radiance.utils.misc import get_masked_dict
from radiance.utils.misc import is_not_none


class AbstractGraph(nn.Module):
    """Highest level graph class. Somewhat useful to lift code up and out of the way."""

    def __init__(self) -> None:
        super().__init__()
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    def get_device(self):
        """Returns the device that the torch parameters are on."""
        return self.device_indicator_param.device

    @abstractmethod
    def forward(
        self,
        ray_indices: TensorType["num_rays", 3],
        batch: Union[str, Dict[str, torch.tensor]] = None,
        step: int = None,
    ):
        """Process starting with ray indices. Turns them into rays, then performs volume rendering."""


class Graph(AbstractGraph):
    """_summary_"""

    def __init__(
        self,
        intrinsics=None,
        camera_to_world=None,
        loss_coefficients: DictConfig = None,
        steps_per_occupancy_grid_update=16,
        scene_bounds: SceneBounds = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert is_not_none(scene_bounds), "scene_bounds is needed to use the occupancy grid"
        self.intrinsics = intrinsics
        self.camera_to_world = camera_to_world
        self.scene_bounds = scene_bounds
        self.loss_coefficients = loss_coefficients
        self.steps_per_occupancy_grid_update = steps_per_occupancy_grid_update
        self.kwargs = kwargs
        self.collider = None
        self.ray_generator = RayGenerator(self.intrinsics, self.camera_to_world)
        self.populate_collider()
        self.populate_fields()
        self.populate_misc_modules()  # populate the modules

    def populate_collider(self):
        """Set the scene bounds collider to use."""
        self.collider = AABBBoxCollider(self.scene_bounds)

    @abstractmethod
    def populate_misc_modules(self):
        """Initializes any additional modules that are part of the network."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Dict[str, List[Parameter]]: Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle):
        """Takes in a Ray Bundle and returns a dictionary of outputs."""

    def forward(
        self,
        ray_indices: TensorType["num_rays", 3],
        batch: Union[str, Dict[str, torch.tensor]] = None,
        step: int = None,
    ):

        # get the rays
        original_ray_bundle = self.ray_generator.forward(ray_indices)  # RayBundle
        intersected_ray_bundle = self.collider(original_ray_bundle)

        if isinstance(batch, type(None)):
            # during inference, keep all rays
            outputs = self.get_outputs(intersected_ray_bundle)
            return outputs

        # during training, keep only the rays that intersect the scene. discard the rest
        valid_mask = intersected_ray_bundle.valid_mask
        masked_intersected_ray_bundle = intersected_ray_bundle.get_masked_ray_bundle(valid_mask)
        masked_batch = get_masked_dict(batch, valid_mask)  # NOTE(ethan): this is really slow if on CPU!
        outputs = self.get_outputs(masked_intersected_ray_bundle)
        loss_dict = self.get_loss_dict(outputs=outputs, batch=masked_batch)
        return outputs, loss_dict

    @abstractmethod
    def get_loss_dict(self, outputs, batch) -> Dict[str, torch.tensor]:
        """Computes and returns the losses."""

    def get_aggregated_loss_from_loss_dict(self, loss_dict):
        """Computes the aggregated loss from the loss_dict and the coefficients specified."""
        aggregated_loss = 0.0
        for loss_name, loss_value in loss_dict.items():
            assert loss_name in self.loss_coefficients, f"{loss_name} no in self.loss_coefficients"
            loss_coefficient = self.loss_coefficients[loss_name]
            aggregated_loss += loss_coefficient * loss_value
        return aggregated_loss

    @torch.no_grad()
    def get_outputs_for_camera(self, intrinsics, camera_to_world, chunk_size=1024, training_camera_index=0):
        """Takes in camera parameters and computes the output of the graph."""
        # NOTE(ethan): this function has a spike in CPU usage

        device = self.get_device()
        assert len(intrinsics.shape) == 1
        num_intrinsics_params = len(intrinsics)
        camera_class = get_camera_model(num_intrinsics_params)
        camera = camera_class(*intrinsics.tolist(), camera_to_world=camera_to_world)
        camera_ray_bundle = camera.generate_camera_rays()
        # TODO(ethan): decide how to properly handle the image indices for validation images
        camera_ray_bundle.set_camera_indices(camera_index=training_camera_index)
        camera_ray_bundle.move_to_device(device)
        image_height, image_width = camera_ray_bundle.origins.shape[:2]

        num_rays = len(camera_ray_bundle)

        outputs = {}
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, chunk_size):
            start_idx = i
            end_idx = i + chunk_size
            original_ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            intersected_ray_bundle = self.collider(original_ray_bundle)
            outputs = self.get_outputs(intersected_ray_bundle)
            for output_name, output in outputs.items():
                outputs_lists[output_name].append(output)
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)
        return outputs

    @abstractmethod
    def log_test_image_outputs(self, image_idx, step, image, mask, outputs):
        """Writes the test image outputs."""
