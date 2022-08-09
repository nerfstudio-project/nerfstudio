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

from abc import abstractmethod
import os
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler

from nerfactory.data.dataloader import AbstractDataloader
from nerfactory.model.base import Model
from nerfactory.optimizers.optimizers import Optimizers
from nerfactory.utils.decorators import check_main_thread


class Pipeline:
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the dataloader and model classes from the trainer,
    worrying about:
    1) Fetching data with the dataloader
    2) Feeding the model the data and fetching the loss
    3) (TODO) Visualization
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.


    TODO: For visualizer functionality to be added down the line, we should make sure
    that we are still abstracting away the Model from the end user. The visualizer function
    should probably be done by adding a new iterator on the base dataloader that will
    talk to the actual visualizer. This is probably ideal to have the visualizer be
    primarily located in the dataloader (first because it makes sense as the
    visualizers main job in this context is to feed in data for the model to load)
    so that we can have an easier time ensuring that the visualizer is always
    spitting out the same formatted data as for in train / eval.


    Args:
        model: The model to be used in the pipeline.
        dataloader: The dataloader to be used in the pipeline.
        loss_coefficients: A dictionary of loss coefficients that will be used

    Attributes:
        self.dataloader (AbstractDataloader): The dataloader that will be used
        self.dataloader_train_iter (Iterator): The iterator for the training dataloader
        self.dataloader_eval_iter (Iterator): The iterator for the evaluation dataloader
        self.model (Model): The model that will be used
        self.mixed_precision (bool): Whether or not to use mixed precision when fetching
            the loss dicts
        self.loss_coefficients (Dict): The loss coefficients for the model
    """

    def __init__(
        self,
        model: Model,
        dataloader: AbstractDataloader,
        optimizers: Optimizers,
        grad_scaler: GradScaler = None,
    ):
        self.dataloader: AbstractDataloader = dataloader
        self.dataloader_train_iter = self.dataloader.iter_train()
        self.dataloader_eval_iter = self.dataloader.iter_eval()
        self.model: Model = model
        self.mixed_precision: bool = False
        self.optimizers: Optimizers = optimizers
        self.grad_scaler: GradScaler = grad_scaler

    @abstractmethod
    def get_train_loss_dict(self):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the dataloader and interfacing with the
        Model class, feeding the data to the model's forward function."""

    @abstractmethod
    def get_eval_loss_dict(self):
        """This function gets your evaluation loss dict. It needs to get the data
        from the dataloader and feed it to the model's forward function"""

    @abstractmethod
    def log_test_image_outputs(self) -> None:
        """Log the test image outputs"""

    @check_main_thread
    def save_checkpoint(self, output_dir: str, step: int) -> None:
        """Save the model and optimizers

        Args:
            output_dir: directory to save the checkpoint
            step: number of steps in training for given checkpoint
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ckpt_path = os.path.join(output_dir, f"step-{step:09d}.ckpt")
        torch.save(
            {
                "step": step,
                "dataloader": self.dataloader.state_dict(),
                "model": self.model.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict() if self.grad_scaler else None,
            },
            ckpt_path,
        )

    def load_checkpoint(self, load_config) -> int:
        """Helper function to load graph and optimizer from prespecified checkpoint.

        Returns the next step number to start from."""
        load_path = os.path.join(load_config.load_dir, f"step-{load_config.load_step:09d}.ckpt")
        assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        # load the checkpoints for graph and optimizer
        self.dataloader.load_state_dict(loaded_state["dataloader"])
        self.model.load_state_dict(loaded_state["model"])
        self.optimizers.load_optimizers(loaded_state)
        self.grad_scaler.load_state_dict(loaded_state["scaler"])

        return loaded_state["step"] + 1

    def get_param_groups(self) -> List[Dict]:
        """Get the param groups for the optimizers.

        Returns:
            A list of dictionaries containing the optimizer's param groups.
        """
        return {**self.dataloader.get_param_groups(), **self.model.get_param_groups()}
