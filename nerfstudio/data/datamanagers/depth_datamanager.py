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
Depth datamanager.
"""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.data.datasets.depth_dataset import DepthDataset


@dataclass
class DepthDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A depth datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: DepthDataManager)


class DepthDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing depth data.
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def create_train_dataset(self) -> DepthDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return DepthDataset(
            dataparser_outputs=self.train_dataparser_outputs,
        )

    def create_eval_dataset(self) -> DepthDataset:
        return DepthDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
        )
