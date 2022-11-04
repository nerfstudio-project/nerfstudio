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
from nerfstudio.data.scene_box import SceneBox


@dataclass
class Semantics:
    """Dataclass for semantic labels."""

    filenames: List[Path]
    """filenames to load semantic data"""
    classes: List[str]
    """class labels for data"""
    colors: torch.Tensor
    """color mapping for classes"""
    mask_classes: List[str] = field(default_factory=lambda: [])
    """classes to mask out from training for all modalities"""


@dataclass
class DataparserOutputs:
    """Dataparser outputs for the which will be used by the DataManager
    for creating RayBundle and RayGT objects."""

    image_filenames: List[Path]
    """Filenames for the images."""
    cameras: Cameras
    """Camera object storing collection of camera information in dataset."""
    alpha_color: Optional[TensorType[3]] = None
    """Color of dataset background."""
    scene_box: SceneBox = SceneBox()
    """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
    mask_filenames: Optional[List[Path]] = None
    """Filenames for any masks that are required"""
    metadata: Dict[str, Any] = to_immutable_dict({})
    """Dictionary of any metadata that be required for the given experiment.
    Will be processed by the InputDataset to create any additional tensors that may be required.
    """

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)


@dataclass
class DataParserConfig(cfg.InstantiateConfig):
    """Basic dataset config"""

    _target: Type = field(default_factory=lambda: DataParser)
    """_target: target class to instantiate"""
    data: Path = Path()
    """Directory specifying location of data."""


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
    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """Abstract method that returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """

    def get_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """Returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        dataparser_outputs = self._generate_dataparser_outputs(split)
        return dataparser_outputs
