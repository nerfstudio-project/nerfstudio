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
from dataclasses import dataclass

import nerfactory.configs.base as cfg
from nerfactory.datamanagers.structs import DatasetInputs


@dataclass
class DataParser:
    """A dataset."""

    def __init__(self, config: cfg.DataParserConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def _generate_dataset_inputs(self, split: str = "train") -> DatasetInputs:
        """Returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate.

        Returns:
            DatasetInputs
        """

    def get_dataset_inputs(self, split: str = "train") -> DatasetInputs:
        """Returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate.

        Returns:
            DatasetInputs
        """
        dataset_inputs = self._generate_dataset_inputs(split)
        return dataset_inputs
