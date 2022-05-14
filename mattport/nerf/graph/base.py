"""
The Graph module contains all trainable parameters.
"""
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import Parameter
from torchtyping import TensorType

from mattport.nerf.field_modules.ray_generator import RayGenerator
from mattport.structures.cameras import get_camera_model
from mattport.structures.rays import RayBundle


@dataclass
class GraphInputs:
    """Datastucture to encode the inputs to the graph."""

    points: float


@dataclass
class Node:
    """Node datastructure for graph composition."""

    name: str
    children: Dict[str, "Node"]
    parents: Dict[str, "Node"]
    visited_order: Optional[bool] = False
    visited_in_dim: Optional[bool] = False

    def __hash__(self):
        return hash(self.name)


class Graph(nn.Module):
    """_summary_"""

    def __init__(self, intrinsics=None, camera_to_world=None, loss_coefficients: DictConfig = None, **kwargs) -> None:
        super().__init__()
        self.intrinsics = intrinsics
        self.camera_to_world = camera_to_world
        self.loss_coefficients = loss_coefficients
        self.kwargs = kwargs
        self.ray_generator = RayGenerator(self.intrinsics, self.camera_to_world)
        self.populate_modules()  # populate the modules

        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    def get_device(self):
        """Returns the device that the torch parameters are on."""
        return self.device_indicator_param.device

    @abstractmethod
    def populate_modules(self):
        """Initializes the modules that are part of the network."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups of the network in a dictionary format"""

    def get_in_dim(self, curr_node: Node) -> None:
        """Dynamically calculates and sets the input dimensions of the modules based on dependency graph

        Args:
            curr_node (Node): pointer to current node in process
        """
        curr_node.visited_in_dim = True
        if len(curr_node.parents) > 0:
            in_dim = 0
            for parent_name in curr_node.parents.keys():
                in_dim += self[parent_name].get_out_dim()
            self[curr_node.name].set_in_dim(in_dim)
            self.modules[curr_node.name].meta_data.in_dim = in_dim

        for child_node in curr_node.children.values():
            if not child_node.visited_in_dim:
                self.get_in_dim(child_node)

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle):
        """Takes in a Ray Bundle and returns a dictionary of outputs."""

    def forward(self, ray_indices: TensorType["num_rays", 3], batch: Union[str, Dict[str, torch.tensor]] = None):
        """Forward function that needs to be overridden."""
        # get the rays:
        ray_bundle = self.ray_generator.forward(ray_indices)  # RayBundle
        outputs = self.get_outputs(ray_bundle)
        if not isinstance(batch, type(None)):
            loss_dict = self.get_loss_dict(outputs=outputs, batch=batch)
            return outputs, loss_dict
        return outputs

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
        assert len(intrinsics.shape) == 1
        num_intrinsics_params = len(intrinsics)
        camera_class = get_camera_model(num_intrinsics_params)
        camera = camera_class(*intrinsics.tolist(), camera_to_world=camera_to_world)
        camera_ray_bundle = camera.generate_camera_rays()
        # TODO(ethan): decide how to properly handle the image indices for validation images
        camera_ray_bundle.set_camera_indices(camera_index=training_camera_index)
        image_height, image_width = camera_ray_bundle.origins.shape[:2]

        device = self.get_device()
        num_rays = camera_ray_bundle.get_num_rays()

        outputs = {}
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, chunk_size):
            start_idx = i
            end_idx = i + chunk_size
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            ray_bundle.move_to_device(device)
            outputs = self.get_outputs(ray_bundle)
            for output_name, output in outputs.items():
                outputs_lists[output_name].append(output)
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)
        return outputs

    @abstractmethod
    def log_test_image_outputs(self, image_idx, step, image, outputs):
        """Writes the test image outputs."""
