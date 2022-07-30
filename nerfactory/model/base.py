# Copyright 2022 The Plenoptix Team. All rights reserved.
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

from abc import abstractmethod
from typing import Any, Dict, List
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import Parameter

from nerfactory.data.structs import DataloaderOutputs, ModelOutputs, SceneBounds
from nerfactory.graphs.modules.ray_generator import RayGenerator
from nerfactory.utils.misc import instantiate_from_dict_config


class Model(nn.Module):
    """This contains the rendering, and the loss calculation, taking in a GenericDataContainer.

    This is effectivley the graph object, just without any of the data and streamlining
    the process of forward propagation. It removes the get_outputs and forward_after_ray_generation
    functions, since we now pass in containers that have already generated the rays. Also removes
    some of the functions like get_outputs_for_camera since we can't assume what information is
    required to render a specific camera angle. The same goes for get_outputs_for_camera_ray_bundle.
    TODO: This viewer functionality will be moved into the Pipeline class.


    Args:
        steps_per_occupancy_grid_update: The number of steps per occupancy grid update.
        scene_bounds: The scene bounds.
        collider_config: The collider config.
        kwargs: The extra kwargs.

    Attributes:
        device_indicator_param: A dummy parameter that is used to indicate the device the model is on.
        scene_bounds: The scene bounds.
        collider_config: The collider config.
        steps_per_occupancy_grid_update: The number of steps per occupancy grid update.
        kwargs: The extra kwargs passed to the model.
        collider: The collider.
        ray_generator: The ray generator.
        callbacks: The callbacks.
    """

    def __init__(
        self,
        steps_per_occupancy_grid_update: int = 16,
        scene_bounds: SceneBounds = None,
        collider_config: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device_indicator_param = nn.Parameter(torch.empty(0))
        self.scene_bounds = scene_bounds
        self.collider_config = collider_config
        self.steps_per_occupancy_grid_update = steps_per_occupancy_grid_update
        self.kwargs = kwargs
        self.collider = None
        self.ray_generator = RayGenerator(self.intrinsics, self.camera_to_world)
        self.populate_collider()
        self.populate_fields()
        self.populate_misc_modules()  # populate the modules
        self.callbacks = None
        # variable for visualizer to fetch TODO(figure out if there is cleaner way to do this)
        self.vis_outputs = None
        self.default_output_name = None

    def register_callbacks(self):  # pylint:disable=no-self-use
        """Option to register callback for training functions"""
        self.callbacks = []

    def populate_collider(self):
        """Set the scene bounds collider to use."""
        # TODO: FIX ME
        # TODO: Collider class with an initializer from the config
        self.collider = instantiate_from_dict_config(self.collider_config, scene_bounds=self.scene_bounds)

    @abstractmethod
    def populate_fields(self):
        """Initializes the field for this model"""

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
    def forward(self, data_batch: DataloaderOutputs) -> ModelOutputs:
        """Run the forward starting with ray indices."""

    @abstractmethod
    def get_loss_dict(self, data_batch: DataloaderOutputs, outputs: ModelOutputs) -> Dict[str, torch.tensor]:
        """Computes and returns the losses."""

    @abstractmethod
    def log_test_image_outputs(self, data_batch: DataloaderOutputs) -> None:
        """Log the test image outputs"""

    def load_graph(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path"""
        self.load_state_dict({key.replace("module.", ""): value for key, value in loaded_state["model"].items()})
