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

from abc import abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import PIL
import torch
from PIL import Image
from torchtyping import TensorType

from pyrad.data.structs import Semantics

from pyrad.utils.misc import is_not_none


class ImageDataset(torch.utils.data.Dataset):
    """Dataset that returns images."""

    def __init__(
        self,
        image_filenames: List[str],
        downscale_factor: int = 1,
        alpha_color: Optional[TensorType[3]] = None,
        **kwargs,
    ):
        """_summary_

        Args:
            image_filenames (List[str]): List of image filenames
            downscale_factor (int, optional): How much to downscale the image. Defaults to 1.
            alpha_color (TensorType[3], optional): Sets transparent regions to specified color, otherwise black.
        """
        super().__init__()
        assert isinstance(downscale_factor, int)
        self.image_filenames = image_filenames
        self.downscale_factor = downscale_factor
        self.alpha_color = alpha_color
        self.kwargs = kwargs

    def __len__(self):
        return len(self.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image.

        Args:
            image_idx (int): The image index in the dataset.

        Returns:
            np.uint8: an image of shape (H, W, 3 or 4)
        """
        image_filename = self.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.downscale_factor != 1.0:
            image_width, image_height = pil_image.size
            if image_width % self.downscale_factor != 0:
                raise ValueError(
                    f"Image width {image_width} is not divisible by downscale_factor {self.downscale_factor}"
                )
            if image_height % self.downscale_factor != 0:
                raise ValueError(
                    f"Image height {image_height} is not divisible by downscale_factor {self.downscale_factor}"
                )
            pil_image = pil_image.resize(
                (image_width // self.downscale_factor, image_height // self.downscale_factor), PIL.Image.BILINEAR
            )
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int):
        """Returns a 3 channel image."""
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    @abstractmethod
    def get_mask(self, image_idx: int) -> Union[TensorType["image_height", "image_width", 1], None]:
        """Returns a mask, which indicates which pixels are valid to use with nerf."""
        return None

    @abstractmethod
    def get_semantics(self, image_idx: int) -> Union[TensorType["image_height", "image_width", "num_classes"], None]:
        """Returns an image with semantic class values."""
        return None

    def get_data(self, image_idx) -> Dict:
        """Returns the ImageDataset data as a dictionary."""
        image = self.get_image(image_idx)
        mask = self.get_mask(image_idx)
        semantics = self.get_semantics(image_idx)
        data = {"image_idx": image_idx}
        assert is_not_none(image)
        data["image"] = image
        if is_not_none(mask):
            assert mask.shape[:2] == image.shape[:2]
            data["mask"] = mask
        if is_not_none(semantics):
            assert semantics.shape[:2] == image.shape[:2]
            data["semantics"] = semantics
        return data

    def __getitem__(self, image_idx):
        data = self.get_data(image_idx)
        return data


class PanopticImageDataset(ImageDataset):
    """Panoptic image dataset that masks out people."""

    def __init__(
        self,
        semantics: Semantics,
        image_filenames: List[str],
        downscale_factor: int = 1,
        alpha_color: Optional[TensorType[3]] = None,
        **kwargs,
    ):
        self.semantics = semantics
        self.person_index = self.semantics.thing_classes.index("person")
        super().__init__(image_filenames, downscale_factor, alpha_color, **kwargs)

    def get_mask(self, image_idx):
        """Mask out the people. Valid only where there aren't people."""
        thing_image_filename = self.semantics.thing_filenames[image_idx]
        pil_image = Image.open(thing_image_filename)
        if self.downscale_factor != 1.0:
            image_width, image_height = pil_image.size
            # the use of NEAREST is important for semantic classes
            pil_image = pil_image.resize(
                (image_width // self.downscale_factor, image_height // self.downscale_factor), Image.NEAREST
            )
        thing_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
        mask = (thing_semantics != self.person_index).to(torch.float32)  # 1 where valid
        return mask

    def get_semantics(self, image_idx):
        stuff_image_filename = self.semantics.stuff_filenames[image_idx]
        pil_image = Image.open(stuff_image_filename)
        if self.downscale_factor != 1.0:
            image_width, image_height = pil_image.size
            # the use of NEAREST is important for semantic classes
            pil_image = pil_image.resize(
                (image_width // self.downscale_factor, image_height // self.downscale_factor), Image.NEAREST
            )
        stuff_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
        return stuff_semantics
