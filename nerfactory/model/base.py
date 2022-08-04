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

"""
Abstract Model classes
"""

from abc import abstractmethod
from typing import Any, Dict, List
import torch
from torch import nn
from torch.nn import Parameter
from nerfactory.cameras.rays import RayBundle

from nerfactory.data.structs import SceneBounds
from nerfactory.graphs.modules.scene_colliders import SceneBoundsCollider


class Model(nn.Module):
    """This contains the rendering, and the loss calculation, taking in a GenericDataContainer.

    This is effectivley the graph object, just without any of the data and streamlining
    the process of forward propagation. It removes the get_outputs and forward_after_ray_generation
    functions, since we now pass in containers that have already generated the rays. Also removes
    some of the functions like get_outputs_for_camera since we can't assume what information is
    required to render a specific camera angle. The same goes for get_outputs_for_camera_ray_bundle.
    TODO: This viewer functionality will be moved into the Pipeline class.


    Args:
        collider (SceneBoundsCollider): Collider for our points
        scene_bounds (SceneBounds): Scene bounds

    Attributes:
        device_indicator_param: A dummy parameter that is used to indicate the device the model is on.
        scene_bounds: The scene bounds.
        steps_per_occupancy_grid_update: The number of steps per occupancy grid update.
        kwargs: The extra kwargs passed to the model.
        collider: The collider.
        callbacks: The callbacks.

    """

    def __init__(
        self,
        collider: SceneBoundsCollider,
        scene_bounds: SceneBounds = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device_indicator_param = nn.Parameter(torch.empty(0))
        self.scene_bounds = scene_bounds
        self.kwargs = kwargs
        self.collider: SceneBoundsCollider = collider
        self.populate_fields()
        self.populate_misc_modules()  # populate the modules
        self.callbacks = None

    def register_callbacks(self):  # pylint:disable=no-self-use
        """Option to register callback for training functions"""
        self.callbacks = []

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
    def forward(self, rays: RayBundle) -> torch.TensorType["rays_shape":..., "color_channels":3]:
        """Run the forward starting with ray indices.

        For subclassed versions of the model, we intend for additional arguments to be
        added as needed for different models. This will generally return pixel values, but
        may also return other information as well for different subclasses. These subclassed
        differences are why we have the pipeline class alongside this to provide a standard
        higher level interface.
        """

    @abstractmethod
    def get_loss_dict(self) -> Dict[str, torch.tensor]:
        """Computes and returns the losses from our data and the outputs of the forward function."""

    def load_graph(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path"""
        self.load_state_dict({key.replace("module.", ""): value for key, value in loaded_state["model"].items()})
