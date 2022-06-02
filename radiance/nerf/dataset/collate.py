"""
Classes to help with caching data while training.
"""

import random
from typing import Callable

import torch
from torch.utils.data import default_collate

from radiance.utils.misc import get_dict_to_torch


def collate_batch_size_one(batch_list):
    """Use the default collate function but then squeze to avoid the batch dimension being added."""
    assert len(batch_list) == 1
    collated_batch = default_collate(batch_list)
    for key in collated_batch:
        assert collated_batch[key].shape[0] == 1
        collated_batch[key] = collated_batch[key].squeeze(0)
    return collated_batch


class CollateIterDataset(torch.utils.data.IterableDataset):
    """Collated image dataset that implements caching of images.
    Creates batches of the ImageDataset return type.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_samples_to_collate: int,
        num_times_to_repeat_images: int = 0,
        device="cpu",
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

    def get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""
        indices = random.sample(range(len(self.dataset)), k=self.num_samples_to_collate)
        batch_list = [self.dataset.__getitem__(idx) for idx in indices]
        return batch_list

    def get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self.get_batch_list()
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
                collated_batch = self.get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch
