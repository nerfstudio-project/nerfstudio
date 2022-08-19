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
Base Model implementation which takes in RayBundles
"""
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import Parameter

from nerfactory.cameras.cameras import Camera
from nerfactory.cameras.rays import RayBundle
from nerfactory.dataloaders.structs import SceneBounds
from nerfactory.utils import profiler
from nerfactory.utils.callbacks import Callback
from nerfactory.utils.config import ModelConfig
from nerfactory.utils.misc import instantiate_from_dict_config, is_not_none


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    TODO:

    Args:
        scene_bounds: Bounds of target scene.
        loss_coefficients: Loss specific weights.
        enable_collider: Whether to create a scene collider to filter rays.
        collider_config: Configuration of scene collider.
        enable_density_field: Whether to create a density field to filter samples.
        density_field_config: Configuration of density field.
    """

    def __init__(
        self,
        scene_bounds: SceneBounds,
        enable_collider: bool = True,
        collider_config: Optional[DictConfig] = None,
        enable_density_field: bool = False,
        density_field_config: Optional[DictConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.scene_bounds = scene_bounds
        self.enable_collider = enable_collider
        self.collider_config = collider_config
        self.enable_density_field = enable_density_field
        self.density_field_config = density_field_config
        self.density_field = None
        self.kwargs = kwargs
        self.collider = None
        self.populate_density_field()
        self.populate_collider()
        self.populate_fields()
        self.populate_misc_modules()  # populate the modules
        self.callbacks = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(self) -> List[Callback]:  # pylint:disable=no-self-use
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_density_field(self):
        """Set the scene density field to use."""
        if self.enable_density_field:
            self.density_field = instantiate_from_dict_config(self.density_field_config)

    def populate_collider(self):
        """Set the scene bounds collider to use."""
        if self.enable_collider:
            self.collider = instantiate_from_dict_config(self.collider_config, scene_bounds=self.scene_bounds)

    @abstractmethod
    def populate_fields(self):
        """Set the fields."""

    @abstractmethod
    def populate_misc_modules(self):
        """Initializes any additional modules that are part of the network."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def forward(self, ray_bundle: RayBundle) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Run forward starting with a ray bundle.

        This takes in raybundles (containing all the information needed to render that ray...
        latents included), and returns the outputs of the model, as well as the mask for the
        rays if there is a collider / scene bounds used."""
        if self.collider is not None:
            intersected_ray_bundle = self.collider(ray_bundle)
            valid_mask = intersected_ray_bundle.valid_mask[..., 0]
        else:
            # NOTE(ruilongli): we don't need collider for ngp
            intersected_ray_bundle = ray_bundle
            valid_mask = None

        if valid_mask is not None:
            # during training, keep only the rays that intersect the scene. discard the rest
            intersected_ray_bundle = intersected_ray_bundle[valid_mask]

        outputs = self.get_outputs(intersected_ray_bundle)

        return outputs, valid_mask

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in ray bundle that needs to be batched and returns the outputs."""
        assert is_not_none(camera_ray_bundle.num_rays_per_chunk)
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs = {}
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, camera_ray_bundle.num_rays_per_chunk):
            start_idx = i
            end_idx = i + camera_ray_bundle.num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs, _ = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path"""
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore


class VanillaModel(Model):
    """Model class for the vanilla NeRF paradigm"""

    def get_outputs_for_camera(self, camera: Camera):
        """Get the model outputs for a Camera."""
        camera_ray_bundle = camera.get_camera_ray_bundle(device=self.device)
        return self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    @abstractmethod
    def populate_fields(self):
        """Set the fields."""

    @abstractmethod
    def populate_misc_modules(self):
        """Initializes any additional modules that are part of the network."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def log_test_image_outputs(self, image_idx, step, batch, outputs) -> float:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            The psnr.
        """


@profiler.time_function
def setup_model(config: ModelConfig, scene_bounds: SceneBounds, device: str) -> Model:
    """Setup the model. The dataset inputs should be set with the training data.

    Args:
        dataset_inputs: The inputs which will be used to define the camera parameters.
    """
    model = instantiate_from_dict_config(DictConfig(config), scene_bounds=scene_bounds, device=device)
    model.to(device)
    return model
