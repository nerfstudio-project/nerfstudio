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
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
from torchtyping import TensorType

import nerfstudio.configs.base_config as cfg
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBounds


@dataclass
class Semantics:
    """Dataclass for semantic labels."""

    stuff_filenames: List[Path]
    """filenames to load "stuff"/background data"""
    stuff_classes: List[str]
    """class labels for "stuff" data"""
    stuff_colors: torch.Tensor
    """color mapping for "stuff" classes"""
    thing_filenames: List[Path]
    """filenames to load "thing"/foreground data"""
    thing_classes: List[str]
    """class labels for "thing" data"""
    thing_colors: torch.Tensor
    """color mapping for "thing" classes"""


@dataclass
class DataparserOutputs:
    """Dataset inputs for the image dataset and the ray generator."""

    image_filenames: List[Path]
    """Filenames for the images."""
    cameras: Cameras
    """Camera object storing collection of camera information in dataset"""
    alpha_color: Optional[TensorType[3]] = None
    """color of dataset background"""
    scene_bounds: SceneBounds = SceneBounds()
    """scene bounds of dataset"""
    # we support additional input information/formats including mask/depth/semantics.
    additional_inputs: Dict[str, Any] = to_immutable_dict({})
    """Dictionary of additional dataset information (e.g. semantics/point clouds/masks).
    {input_name:
    ... {"func": function to process additional dataset inputs,
    ... "kwargs": dictionary of data to pass into "func"}
    }
    """

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)


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
    def _generate_dataset_inputs(self, split: str = "train") -> DataparserOutputs:
        """Abstract method that returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DatasetInputs containing data for the specified dataset and split
        """

    def get_dataset_inputs(self, split: str = "train") -> DataparserOutputs:
        """Returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DatasetInputs containing data for the specified dataset and split
        """
        dataset_inputs = self._generate_dataset_inputs(split)
        return dataset_inputs
