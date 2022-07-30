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

from typing import Dict
import torch
from nerfactory.data.dataloader import AbstractDataloader
from nerfactory.data.structs import DataloaderOutputs, ModelOutputs
from nerfactory.model.base import Model


class Pipeline:
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization.

    This class's function is to hide the dataloader and model classes from the trainer,
    worrying about:
    1) Fetching data with the dataloader
    2) Feeding the model the data and fetching the loss
    3) (TODO) Visualization
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes.


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
        loss_coefficients: Dict,
    ):
        self.dataloader: AbstractDataloader = dataloader
        self.dataloader_train_iter = self.dataloader.iter_train()
        self.dataloader_eval_iter = self.dataloader.iter_eval()
        self.model: Model = model
        self.mixed_precision: bool = False
        self.loss_coefficients: Dict = loss_coefficients

    def get_train_loss(self):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the dataloader and interfacign with the
        Model class."""
        data: DataloaderOutputs = next(self.dataloader_train_iter)
        with torch.autocast(device_type=data.rays.origins.device, enabled=self.mixed_precision):
            outputs: ModelOutputs = self.model(data)
            loss_dict: Dict = self.model.get_loss_dict(data, outputs)
        return loss_dict

    def get_eval_loss_dict(self):
        """This function gets your evaluation loss dict. It needs to get the data
        from the dataloader and feed it to the model,"""
        data: DataloaderOutputs = next(self.dataloader_eval_iter)
        with torch.autocast(device_type=data.rays.origins.device, enabled=self.mixed_precision):
            outputs: ModelOutputs = self.model(data)
            loss_dict: Dict = self.model.get_loss_dict(data, outputs)
        return loss_dict

    def get_aggregated_loss_dict(self, loss_dict: Dict):
        """Computes the aggregated loss from the loss_dict and the coefficients specified."""
        aggregated_loss_dict = {}
        for loss_name, loss_value in loss_dict.items():
            assert loss_name in self.loss_coefficients, f"{loss_name} no in self.loss_coefficients"
            loss_coefficient = self.loss_coefficients[loss_name]
            aggregated_loss_dict[loss_name] = loss_coefficient * loss_value
        aggregated_loss_dict["aggregated_loss"] = sum(loss_dict.values())
        return aggregated_loss_dict
