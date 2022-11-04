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
Some dataset code.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchtyping import TensorType

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics

class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs):
        super().__init__()
        self.dataparser_outputs = dataparser_outputs
        self.has_masks = self.dataparser_outputs.mask_filenames is not None


    def __len__(self):
        return len(self.dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self.dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self.dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self.dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    def get_mask(self, image_idx: int) -> TensorType["image_height", "image_width", 1]:
        """Returns a boolean mask with same spatial shape as the image.

        Args:
            image_idx: The image index in the dataset.
        """
        pil_mask = Image.open(self.dataparser_outputs.mask_filenames[image_idx])
        mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
        return mask_tensor

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        data["image"] = image
        if self.has_masks:
            data["mask"] = self.get_mask(image_idx)
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data
    
    def get_metadata(self, data: Dict) -> Dict:
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data


class SemanticDataset(InputDataset):
    """Dataset that returns images and semantics and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """
    
    def __init__(self, dataparser_outputs: DataparserOutputs):
        super().__init__(dataparser_outputs)
        assert "semantics" in dataparser_outputs.metadata.keys() and type(dataparser_outputs.metadata["semantics"]) == Semantics
        self.semantics = dataparser_outputs.metadata["semantics"]
        self.mask_indices = torch.tensor([self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]).view(1, 1, -1)

    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        image_filename = self.semantics.filenames[data["image_idx"]]
        pil_image = Image.open(image_filename)
        semantic_label = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
        mask = torch.sum(semantic_label == self.mask_indices, dim=-1, keepdim=True) == 0
        if "mask" in data.keys():
            mask = mask & data["mask"]
        return {"mask": mask, "semantics": semantic_label}
