# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
3D Gaussian Splatting data manager.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union

import torch
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datamanagers.base_datamanager import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.gaussian_splatting_dataparser import GaussianSplattingDataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
)

from torch import Tensor


@dataclass
class RasterizerDataManagerConfig(DataManagerConfig):
    """Configuration for data manager that does not load from a dataset. Instead, it generates random poses."""

    _target: Type = field(default_factory=lambda: RasterizerDataManager)
    """Target class to initiate."""
    dataparser: AnnotatedDataParserUnion = GaussianSplattingDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics.
    """


class RasterizerDataManager(DataManager):  # pylint: disable=abstract-method
    """Rasterization based data manager.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RasterizerDataManagerConfig

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        config: RasterizerDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.gaussians = self.dataparser.get_initial_gaussians()
        self.train_dataset = self.create_train_dataset()

        self.train_image_index = 0

        super().__init__()
        # DataManager.__init__(self)

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def next_train(self, step: int) -> Tuple[int, Tensor]:
        """Returns next training image index and image"""

        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        idx = image_batch["image_idx"][self.train_image_index]
        image = image_batch["image"][self.train_image_index]
        self.train_image_index += 1

        return idx, image

    def get_param_groups(
        self,
    ) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups
