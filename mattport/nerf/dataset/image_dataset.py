"""
Some dataset code.
"""

from typing import List

import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import default_collate


def collate_batch(batch_list, num_rays_per_batch, keep_full_image: bool = False):
    """_summary_

    Args:
     batch (_type_): _description_
    """
    # TODO(ethan): handle sampling even when in different processes
    # we don't want the same samples for all spawned processed when
    # using distributed training

    batch = default_collate(batch_list)
    num_images, image_height, image_width, _ = batch["image"].shape
    indices = torch.floor(
        torch.rand((num_rays_per_batch, 3)) * torch.tensor([num_images, image_height, image_width])
    ).long()
    c, y, x = [i.flatten() for i in torch.split(indices, 1, dim=-1)]
    pixels = batch["image"][c, y, x]
    mask = batch["mask"][c, y, x]
    assert pixels.shape == (num_rays_per_batch, 3), pixels.shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch = {
        "indices": indices,
        "pixels": pixels,
        "mask": mask,
    }
    if keep_full_image:
        collated_batch["image"] = batch["image"]

    return collated_batch


class ImageDataset(torch.utils.data.Dataset):
    """Dataset that returns images."""

    def __init__(self, image_filenames: List[str], downscale_factor: int = 1):
        super().__init__()
        assert isinstance(downscale_factor, int)
        self.image_filenames = image_filenames
        self.downscale_factor = downscale_factor

    def __len__(self):
        return len(self.image_filenames)

    def get_image(self, image_idx: int) -> np.uint8:
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
        image = np.array(pil_image)  # shape is (h, w, 3 or 4)
        assert len(image.shape) == 3
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def __getitem__(self, image_idx):
        # the image might be RGB or RGBA, so we separate it
        original_image = torch.from_numpy(self.get_image(image_idx).astype("float32") / 255.0)
        image = original_image[:, :, :3]
        num_channels = original_image.shape[2]
        if num_channels == 4:
            mask = original_image[:, :, 3]
        elif num_channels == 3:
            mask = np.ones_like(original_image[:, :, 0])
        else:
            raise ValueError(f"Image shape of {image.shape} is in correct.")

        data = {
            "image_idx": image_idx,
            "image": image,  # the pixels
            "mask": mask,
        }
        return data
