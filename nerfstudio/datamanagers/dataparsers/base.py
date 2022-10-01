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

"""A set of standard datasets."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type

import nerfstudio.configs.base as cfg
from nerfstudio.datamanagers.structs import DatasetInputs


@dataclass
class DataParserConfig(cfg.InstantiateConfig):
    """Basic dataset config"""

    _target: Type = field(default_factory=lambda: DataParser)
    """_target: target class to instantiate"""


@dataclass
class DataParser:
    """A dataset.

    Args:
        config: datasetparser config containing all information needed to instantiate dataset
    """

    config: DataParserConfig

    def __init__(self, config: DataParserConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def _generate_dataset_inputs(self, split: str = "train") -> DatasetInputs:
        """Abstract method that returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DatasetInputs containing data for the specified dataset and split
        """

    def get_dataset_inputs(self, split: str = "train") -> DatasetInputs:
        """Returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DatasetInputs containing data for the specified dataset and split
        """
        dataset_inputs = self._generate_dataset_inputs(split)
        return dataset_inputs
