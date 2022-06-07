"""
Code for sampling images from a dataset of images.
"""

from abc import abstractmethod
import random
from typing import Union

import torch
from torch.utils.data import default_collate

from pyrad.utils.misc import get_dict_to_torch


class ImageSampler(torch.utils.data.IterableDataset):
    """Samples image_batch's."""

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()


class CacheImageSampler(ImageSampler):
    """Collated image dataset that implements caching of images.
    Creates batches of the ImageDataset return type.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        num_samples_to_collate (int): How many images are used to sample rays for each batch.
        num_times_to_repeat_images (int): How often to collate new images. Defaults to every iteration.
        device (Union[torch.device, str]): Device to perform computation. Defaults to "cpu".
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_samples_to_collate: int,
        num_times_to_repeat_images: int = 0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.dataset = dataset
        self.num_samples_to_collate = num_samples_to_collate
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.device = device

        self.num_repeated = self.num_times_to_repeat_images  # starting value
        self.cached_collated_batch = None
        self.first_time = True

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""
        indices = random.sample(range(len(self.dataset)), k=self.num_samples_to_collate)
        batch_list = [self.dataset.__getitem__(idx) for idx in indices]
        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = default_collate(batch_list)
        collated_batch = get_dict_to_torch(collated_batch, device=self.device)
        return collated_batch

    def __iter__(self):
        while True:
            if self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch
