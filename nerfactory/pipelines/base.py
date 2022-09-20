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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfactory.configs import base as cfg
from nerfactory.datamanagers.base import DataManager
from nerfactory.models.base import Model
from nerfactory.utils import profiler
from nerfactory.utils.callbacks import TrainingCallback, TrainingCallbackAttributes


def module_wrapper(module: nn.Module) -> nn.Module:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(module, DDP):
        return module.module
    return module


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.


    TODO: For viewer functionality to be added down the line, we should make sure
    that we are still abstracting away the Model from the end user. The viewer function
    should probably be done by adding a new iterator on the base data manager that will
    talk to the actual viewer. This is probably ideal to have the viewer be
    primarily located in the data manager (first because it makes sense as the
    viewers main job in this context is to feed in data for the model to load)
    so that we can have an easier time ensuring that the viewer is always
    returning the same formatted data as for in train / eval. All this is pending changes to
    be done in the future... but just bear in mind that if learned parameters are in the data manager,
    the viewer may have to use those parameters as well.


    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode: if True, loads test datset. if False, loads train/eval datasets
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        self.datamanager: The data manager that will be used
        self.model: The model that will be used
    """

    def __init__(
        self, config: cfg.PipelineConfig, device: str, test_mode: bool = False, world_size: int = 1, local_rank: int = 0
    ):
        super().__init__()
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_input_dataset is not None, "Missing input dataset"
        self.model: Model = config.model.setup(
            scene_bounds=self.datamanager.train_input_dataset.dataset_inputs.scene_bounds,
            num_train_data=len(self.datamanager.train_input_dataset),
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self.model = typing.cast(
                Model, typing.cast(Model, DDP(self.model, device_ids=[local_rank], find_unused_parameters=True))
            )
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return module_wrapper(self.model).device

    @abstractmethod
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = module_wrapper(self.model).get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = module_wrapper(self.model).get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average."""
        self.eval()
        metrics_dict_list = []
        # TODO: add something like tqdm but so that it doesn't interfere with logging
        for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
            outputs = module_wrapper(self.model).get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, _ = module_wrapper(self.model).get_image_metrics_and_images(outputs, batch)
            metrics_dict_list.append(metrics_dict)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict

    @abstractmethod
    def log_test_image_outputs(self) -> None:
        """Log the test image outputs"""

    def load_pipeline(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        self.load_state_dict(state)  # type: ignore

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = module_wrapper(self.model).get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = module_wrapper(self.model).get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
