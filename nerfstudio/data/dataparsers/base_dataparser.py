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

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
from torchtyping import TensorType
from typing_extensions import Literal

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
    dataparser_transform: TensorType[3, 4] = torch.eye(4)[:3, :]
    """Transform applied by the dataparser."""
    dataparser_scale: float = 1.0
    """Scale applied by the dataparser."""

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)

    def save_dataparser_transform(self, path: Path):
        """Save dataparser transform to json file. Some dataparsers will apply a transform to the poses,
        this method allows the transform to be saved so that it can be used in other applications.

        Args:
            path: path to save transform to
        """
        data = {
            "transform": self.dataparser_transform.tolist(),
            "scale": float(self.dataparser_scale),
        }
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w", encoding="UTF-8") as file:
            json.dump(data, file, indent=4)

    def transform_poses_to_original_space(
        self,
        poses: TensorType["num_poses", 3, 4],
        camera_convention: Literal["opengl", "opencv"] = "opencv",
    ) -> TensorType["num_poses", 3, 4]:
        """
        Transforms the poses in the transformed space back to the original world coordinate system.
        Args:
            poses: Poses in the transformed space
            camera_convention: Camera system convention used for the transformed poses
        Returns:
            Original poses
        """
        return transform_poses_to_original_space(
            poses,
            self.dataparser_transform,
            self.dataparser_scale,
            camera_convention=camera_convention,
        )


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


def transform_poses_to_original_space(
    poses: TensorType["num_poses", 3, 4],
    applied_transform: TensorType[3, 4],
    applied_scale: float,
    camera_convention: Literal["opengl", "opencv"] = "opencv",
) -> TensorType["num_poses", 3, 4]:
    """
    Transforms the poses in the transformed space back to the original world coordinate system.
    Args:
        poses: Poses in the transformed space
        applied_transform: Transform matrix applied in the data processing step
        applied_scale: Scale used in the data processing step
        camera_convention: Camera system convention used for the transformed poses
    Returns:
        Original poses
    """
    output_poses = torch.cat(
        (
            poses,
            torch.tensor([[[0, 0, 0, 1]]], dtype=poses.dtype, device=poses.device).repeat_interleave(len(poses), 0),
        ),
        1,
    )
    output_poses[..., :3, 3] /= applied_scale
    inv_transform = torch.linalg.inv(
        torch.cat(
            (
                applied_transform,
                torch.tensor([[0, 0, 0, 1]], dtype=applied_transform.dtype, device=applied_transform.device),
            ),
            0,
        )
    )
    output_poses = torch.einsum("ij,bjk->bik", inv_transform, output_poses)
    if camera_convention == "opencv":
        output_poses[..., 0:3, 1:3] *= -1
    elif camera_convention == "opengl":
        pass
    else:
        raise ValueError(f"Camera convention {camera_convention} is not supported.")
    return output_poses[:, :3]
