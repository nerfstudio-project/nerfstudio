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


import torch
from nerfactory.data.dataloader import AbstractDataloader
from nerfactory.data.structs import BaseDataContainer, BaseModelOutputs
from nerfactory.model.base import Model


class Pipeline:
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization."""

    def __init__(
        self,
        model_cls,
        model_config,
        dataloader_cls,
        dataloader_config,
        loss_coefficients,
    ):
        self.dataloader: AbstractDataloader = dataloader_cls(**dataloader_config)
        self.dataloader_train_iter = self.dataloader.iter_train()
        self.dataloader_eval_iter = self.dataloader.iter_eval()
        self.model: Model = model_cls(**model_config)
        self.mixed_precision = False
        self.loss_coefficients = loss_coefficients

    def get_train_loss(self):
        """This function gets your training loss dict"""
        data: BaseDataContainer = next(self.dataloader_train_iter)
        with torch.autocast(device_type=data.rays.origins.device, enabled=self.mixed_precision):
            outputs: BaseModelOutputs = self.model(data)
            loss_dict: dict = self.model.get_loss_dict(data, outputs)
        return loss_dict

    def get_eval_loss_dict(self):
        """This function gets your evaluation loss dict"""
        data: BaseDataContainer = next(self.dataloader_eval_iter)
        with torch.autocast(device_type=data.rays.origins.device, enabled=self.mixed_precision):
            outputs: BaseModelOutputs = self.model(data)
            loss_dict: dict = self.model.get_loss_dict(data, outputs)
        return loss_dict

    def get_aggregated_loss_dict(self, loss_dict):
        """Computes the aggregated loss from the loss_dict and the coefficients specified."""
        aggregated_loss_dict = {}
        for loss_name, loss_value in loss_dict.items():
            assert loss_name in self.loss_coefficients, f"{loss_name} no in self.loss_coefficients"
            loss_coefficient = self.loss_coefficients[loss_name]
            aggregated_loss_dict[loss_name] = loss_coefficient * loss_value
        aggregated_loss_dict["aggregated_loss"] = sum(loss_dict.values())
        return aggregated_loss_dict

    def process_outputs_as_images(self, outputs: BaseModelOutputs, shape: tuple):  # pylint: disable=no-self-use
        """Process outputs into visualizable colored images

        This function will assume that your output was formed from a raybundle that was
        created from whole images."""
        return outputs.rendered_pixels.reshape(shape)
