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

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import torch
from torch import nn
from torch.nn import Parameter

from nerfactory.cameras.rays import RayBundle
from nerfactory.configs import base as cfg
from nerfactory.datamanagers.structs import SceneBounds
from nerfactory.fields.density_fields.density_grid import DensityGrid
from nerfactory.models.modules.scene_colliders import NearFarCollider
from nerfactory.utils.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfactory.utils.misc import get_masked_dict, is_not_none


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_bounds: dataset scene bounds
    """

    config: cfg.ModelConfig

    def __init__(
        self,
        config: cfg.ModelConfig,
        scene_bounds: SceneBounds,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.scene_bounds = scene_bounds
        self.num_train_data = num_train_data
        self.density_field = None
        self.kwargs = kwargs
        self.collider = None
        self.populate_modules()  # populate the modules
        self.callbacks = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # default instantiates optional modules that are common among many networks
        # NOTE: call `super().populate_modules()` in subclasses

        if self.config.enable_density_field:
            self.density_field = DensityGrid(
                center=self.config.density_field_params["center"],
                base_scale=self.config.density_field_params["base_scale"],
                num_cascades=self.config.density_field_params["num_cascades"],
                resolution=self.config.density_field_params["resolution"],
                update_every_num_iters=self.config.density_field_params["update_every_num_iters"],
            )
        if self.config.enable_collider:
            self.collider = NearFarCollider(
                near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
            )

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

    @overload
    def forward(self, ray_bundle: RayBundle, batch: None = None) -> Dict[str, torch.Tensor]:
        ...

    @overload
    def forward(
        self, ray_bundle: RayBundle, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ...

    def forward(
        self, ray_bundle: RayBundle, batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Union[
        Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
    ]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            batch: containing all the auxilury things needed to train like masks and ground truth pixels.
        """

        if self.collider is not None:
            intersected_ray_bundle = self.collider(ray_bundle)  # pylint: disable=not-callable
            valid_mask = intersected_ray_bundle.valid_mask[..., 0]
        else:
            # NOTE(ruilongli): we don't need collider for ngp
            intersected_ray_bundle = ray_bundle
            valid_mask = None

        if batch is None:
            # during inference, keep all rays
            outputs = self.get_outputs(intersected_ray_bundle)
            return outputs

        if valid_mask is not None:
            intersected_ray_bundle = intersected_ray_bundle[valid_mask]
            # during training, keep only the rays that intersect the scene. discard the rest
            batch = get_masked_dict(batch, valid_mask)  # NOTE(ethan): this is really slow if on CPU!

        outputs = self.get_outputs(intersected_ray_bundle)
        metrics_dict = self.get_metrics_dict(outputs=outputs, batch=batch)
        loss_dict = self.get_loss_dict(
            outputs=outputs, batch=batch, metrics_dict=metrics_dict, loss_coefficients=self.config.loss_coefficients
        )
        return outputs, loss_dict, metrics_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict, loss_coefficients) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: collection of metrics to compute
            loss_coefficients: list of loss coefficients/weightings to apply
        """

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        assert is_not_none(camera_ray_bundle.num_rays_per_chunk)
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs = {}
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, camera_ray_bundle.num_rays_per_chunk):
            start_idx = i
            end_idx = i + camera_ray_bundle.num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    @abstractmethod
    def log_test_image_outputs(self, image_idx, step, batch, outputs) -> Dict[str, float]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore
