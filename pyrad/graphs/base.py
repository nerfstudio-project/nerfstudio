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
The Graph module contains all trainable parameters.
"""
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtyping import TensorType

from pyrad.cameras.cameras import Camera
from pyrad.cameras.rays import RayBundle
from pyrad.data.structs import DatasetInputs, SceneBounds
from pyrad.graphs.modules.ray_generator import RayGenerator
from pyrad.optimizers.optimizers import Optimizers
from pyrad.utils import profiler
from pyrad.utils.decorators import check_main_thread
from pyrad.utils.misc import get_masked_dict, instantiate_from_dict_config, is_not_none


def load_checkpoint(load_config: DictConfig, graph: "Graph", optimizers: Optimizers) -> int:
    """Load the checkpoint from the given path

    Args:
        load_config (DictConfig): Configuration for loading a model.

    Returns:
        int: step iteration associated with the loaded checkpoint
    """
    load_path = os.path.join(load_config.load_dir, f"step-{load_config.load_step:09d}.ckpt")
    assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    graph.load_state_dict({key.replace("module.", ""): value for key, value in loaded_state["model"].items()})
    for k, v in loaded_state["optimizers"].items():
        optimizers.optimizers[k].load_state_dict(v)
    start_step = loaded_state["step"] + 1
    logging.info("done loading checkpoint from %s", load_path)
    return start_step


@check_main_thread
def save_checkpoint(graph: "Graph", optimizers: Optimizers, output_dir: str, step: int) -> None:
    """Save the model and optimizers

    Args:
        graph (Graph): graph to checkpoint
        optimizers (Optimizers): optimizer to checkpoint
        output_dir (str): directory to save the checkpoint
        step (int): number of steps in training for given checkpoint
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ckpt_path = os.path.join(output_dir, f"step-{step:09d}.ckpt")
    torch.save(
        {
            "step": step,
            "model": graph.module.state_dict() if hasattr(graph, "module") else graph.state_dict(),
            "optimizers": {k: v.state_dict() for (k, v) in optimizers.optimizers.items()},
        },
        ckpt_path,
    )


@profiler.time_function
def setup_graph(
    config: DictConfig, dataset_inputs: DatasetInputs, local_rank: int = 0, world_size: int = 1
) -> Tuple["Graph", Optimizers, int]:
    """Setup the graph. The dataset inputs should be set with the training data.

    Args:
        dataset_inputs (DatasetInputs): The inputs which will be used to define the camera parameters.
    """
    device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
    graph = instantiate_from_dict_config(config.network, **dataset_inputs.as_dict())
    graph.to(device)

    optimizers = Optimizers(config.param_groups, graph.get_param_groups())  # NOTE(ethan): can this be before DDP?

    start_step = 0
    if config.resume_train.load_dir:
        start_step = load_checkpoint(config.resume_train, graph, optimizers)

    if world_size > 1:
        graph = DDP(graph, device_ids=[local_rank])
        dist.barrier(device_ids=[local_rank])
    return graph, optimizers, start_step


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
    def forward(self, ray_indices: TensorType["num_rays", 3], batch: Union[str, Dict[str, torch.tensor]] = None):
        """Process starting with ray indices. Turns them into rays, then performs volume rendering."""


class Graph(AbstractGraph):
    """Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        intrinsics (torch.Tensor): Camera intrinsics.
        camera_to_world (torch.Tensor): Camera to world transformation.
        loss_coefficients (DictConfig): Loss specific weights.
        steps_per_occupancy_grid_update (int): How often to update occupancy grid.
        scene_bounds (SceneBounds): Bounds of target scene.
        collider_config (DictConfig): Configuration of scene collider.
    """

    def __init__(
        self,
        intrinsics: torch.Tensor = None,
        camera_to_world: torch.Tensor = None,
        loss_coefficients: DictConfig = None,
        steps_per_occupancy_grid_update: int = 16,
        scene_bounds: SceneBounds = None,
        collider_config: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert is_not_none(scene_bounds), "scene_bounds is needed to use the occupancy grid"
        self.intrinsics = intrinsics
        self.camera_to_world = camera_to_world
        self.scene_bounds = scene_bounds
        self.collider_config = collider_config
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
        self.collider = instantiate_from_dict_config(self.collider_config, scene_bounds=self.scene_bounds)

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
    def get_outputs(self, ray_bundle: RayBundle) -> dict:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle (RayBundle): Input bundle of rays.

        Returns:
            dict: Outputs of graph. (ie. rendered colors)
        """

    def forward_after_ray_generator(self, ray_bundle: RayBundle, batch: Union[str, Dict[str, torch.tensor]] = None):
        """Run forward starting with a ray bundle."""
        intersected_ray_bundle = self.collider(ray_bundle)

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
        aggregated_loss_dict = self.get_aggregated_loss_dict(loss_dict)
        return outputs, aggregated_loss_dict

    def forward(self, ray_indices: TensorType["num_rays", 3], batch: Union[str, Dict[str, torch.tensor]] = None):
        """Run the forward starting with ray indices."""
        ray_bundle = self.ray_generator.forward(ray_indices)
        return self.forward_after_ray_generator(ray_bundle, batch=batch)

    @abstractmethod
    def get_loss_dict(self, outputs, batch) -> Dict[str, torch.tensor]:
        """Computes and returns the losses."""

    def get_aggregated_loss_dict(self, loss_dict):
        """Computes the aggregated loss from the loss_dict and the coefficients specified."""
        aggregated_loss_dict = {}
        for loss_name, loss_value in loss_dict.items():
            assert loss_name in self.loss_coefficients, f"{loss_name} no in self.loss_coefficients"
            loss_coefficient = self.loss_coefficients[loss_name]
            aggregated_loss_dict[loss_name] = loss_coefficient * loss_value
        aggregated_loss_dict["aggregated_loss"] = sum(loss_dict.values())
        return aggregated_loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle):
        """Takes in camera parameters and computes the output of the graph."""
        assert is_not_none(camera_ray_bundle.num_rays_per_chunk)
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs = {}
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, camera_ray_bundle.num_rays_per_chunk):
            start_idx = i
            end_idx = i + camera_ray_bundle.num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward_after_ray_generator(ray_bundle)
            for output_name, output in outputs.items():
                outputs_lists[output_name].append(output)
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)
        return outputs

    def get_outputs_for_camera(self, camera: Camera):
        """Get the graph outputs for a Camera."""
        camera_ray_bundle = camera.get_camera_ray_bundle(device=self.get_device())
        return self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    @abstractmethod
    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        """Writes the test image outputs."""
