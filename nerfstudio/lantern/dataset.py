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
HDR Dataset of .exr format
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class HDRInputDataset(InputDataset):
    """Dataset that returns HDR images, OpenEXR.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    # exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    # cameras: Cameras

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        
    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4), with np.float32 values.

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        hdr_image = cv2.imread(str(image_filename),  cv2.IMREAD_UNCHANGED)
        if self.scale_factor != 1.0:
            # Here cv2 img shape is (height, width, BGR)
            height, width = hdr_image.shape[:2]
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            hdr_image = cv2.resize(hdr_image, newsize, interpolation = cv2.INTER_LINEAR)
        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
        hdr_image = hdr_image.astype("float32")  # shape is (h, w) or (h, w, 3 or 4)
        if len(hdr_image.shape) == 2:
            hdr_image = hdr_image[:, :, None].repeat(3, axis=2)
        assert len(hdr_image.shape) == 3
        assert hdr_image.dtype == np.float32
        assert hdr_image.shape[2] in [3, 4], f"Image shape of {hdr_image.shape} is in correct."
        return hdr_image

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        # Compress HDR using: pixel = log(pixel + 1.)
            
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32"))
        
        # Compress HDR using u-law: pixel = log(u * pixel + 1.) / log(u + 1.)
        u = 5000.
        image = torch.log(1. + u * image) / torch.log(torch.tensor(1.+u))
        
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
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
