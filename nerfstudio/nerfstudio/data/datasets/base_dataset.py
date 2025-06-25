# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Dataset.
"""

from __future__ import annotations

import io
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path, pil_to_numpy


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    cameras: Cameras

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, cache_compressed_images: bool = False
    ):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.mask_color = dataparser_outputs.metadata.get("mask_color", None)
        self.cache_compressed_images = cache_compressed_images
        """If cache_compressed_images == True, cache all the image files into RAM in their compressed form (jpeg, png, etc. but not as pytorch tensors)"""
        if cache_compressed_images:
            self.binary_images = []
            self.binary_masks = []
            for image_filename in self._dataparser_outputs.image_filenames:
                with open(image_filename, "rb") as f:
                    self.binary_images.append(io.BytesIO(f.read()))
            if self._dataparser_outputs.mask_filenames is not None:
                for mask_filename in self._dataparser_outputs.mask_filenames:
                    with open(mask_filename, "rb") as f:
                        self.binary_masks.append(io.BytesIO(f.read()))

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        if self.cache_compressed_images:
            pil_image = Image.open(self.binary_images[image_idx])
        else:
            pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = pil_to_numpy(pil_image)  # shape is (h, w) or (h, w, 3 or 4) and dtype == "uint8"
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is incorrect."
        return image

    def get_image_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_numpy_image(image_idx)
        image = image / np.float32(255)
        image = torch.from_numpy(image)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_image_uint8(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in uint8 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(
            self.get_numpy_image(image_idx)
        )  # removed astype(np.uint8) because get_numpy_image returns uint8
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * (image[:, :, -1:] / 255.0) + 255.0 * self._dataparser_outputs.alpha_color * (
                1.0 - image[:, :, -1:] / 255.0
            )
            image = torch.clamp(image, min=0, max=255).to(torch.uint8)
        return image

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            if self.cache_compressed_images:
                mask_filepath = self.binary_masks[image_idx]
            else:
                mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert data["mask"].shape[:2] == data["image"].shape[:2], (
                f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
            )
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames
