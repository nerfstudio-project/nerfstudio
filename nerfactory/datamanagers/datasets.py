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
Some dataset code.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Union

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchtyping import TensorType

from nerfactory.configs import base as cfg
from nerfactory.datamanagers.structs import DatasetInputs
from nerfactory.utils.misc import is_not_none


class InputDataset(Dataset):
    """Dataset that returns images."""

    def __init__(self, config: Union[cfg.DataParserConfig, cfg.InstantiateConfig], split: str):
        """_summary_

        Args:
            image_filenames (List[str]): List of image filenames
            alpha_color (TensorType[3], optional): Sets transparent regions to specified color, otherwise black.
        """
        super().__init__()
        self.inputs: DatasetInputs = config.setup().get_dataset_inputs(split=split)

    def __len__(self):
        return len(self.inputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image.

        Args:
            image_idx (int): The image index in the dataset.

        Returns:
            np.uint8: an image of shape (H, W, 3 or 4)
        """
        image_filename = self.inputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a 3 channel image."""
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self.inputs.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self.inputs.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    @abstractmethod
    def get_mask(self, image_idx: int) -> Union[TensorType["image_height", "image_width", 1], None]:
        """Returns a mask, which indicates which pixels are valid to use with nerf."""
        if self.inputs.semantics:
            person_index = self.inputs.semantics.thing_classes.index("person")
            thing_image_filename = self.inputs.semantics.thing_filenames[image_idx]
            pil_image = Image.open(thing_image_filename)
            thing_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
            mask = (thing_semantics != person_index).to(torch.float32)  # 1 where valid
            return mask
        return None

    @abstractmethod
    def get_semantics(self, image_idx: int) -> Union[TensorType["image_height", "image_width", "num_classes"], None]:
        """Returns an image with semantic class values."""
        if self.inputs.semantics:
            stuff_image_filename = self.inputs.semantics.stuff_filenames[image_idx]
            pil_image = Image.open(stuff_image_filename)
            stuff_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
            return stuff_semantics
        return None

    def get_data(self, image_idx) -> Dict:
        """Returns the ImageDataset data as a dictionary."""
        image = self.get_image(image_idx)
        mask = self.get_mask(image_idx)
        semantics = self.get_semantics(image_idx)
        data = {"image_idx": image_idx}
        assert is_not_none(image)
        data["image"] = image
        if mask is not None:
            assert mask.shape[:2] == image.shape[:2]
            data["mask"] = mask
        if semantics is not None:
            assert semantics.shape[:2] == image.shape[:2]
            data["semantics"] = semantics
        return data

    def __getitem__(self, image_idx):
        data = self.get_data(image_idx)
        return data
