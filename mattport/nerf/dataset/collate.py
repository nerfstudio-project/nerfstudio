"""
Classes to help with caching data while training.
"""

import random
from typing import Callable

import torch
from torch.utils.data import default_collate


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

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        collate_fn: Callable,
        num_samples_to_collate: int,
        num_times_to_repeat_images: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.num_samples_to_collate = num_samples_to_collate
        self.num_times_to_repeat_images = num_times_to_repeat_images

        self.num_repeated = self.num_times_to_repeat_images  # starting value
        self.cached_batch_list = None

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""
        # sampling without replacement. TODO(ethan): don't use 'random'
        indices = random.sample(range(len(self.dataset)), k=self.num_samples_to_collate)
        batch_list = [self.dataset.__getitem__(idx) for idx in indices]
        return batch_list

    def __iter__(self):
        while True:
            if self.num_repeated >= self.num_times_to_repeat_images:
                # trigger a reset
                self.num_repeated = 0
                batch_list = self.get_batch_list()
                # possibly save a cached item
                self.cached_batch_list = batch_list if self.num_times_to_repeat_images != 0 else None
            else:
                batch_list = self.cached_batch_list
                self.num_repeated += 1
            collated_batch = self.collate_fn(batch_list)
            yield collated_batch
