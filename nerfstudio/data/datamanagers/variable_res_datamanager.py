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
Data loader for variable resolution datasets, where batching raw image tensors isn't possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate


def variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    images = []
    masks = []
    for data in batch:
        image = data.pop("image")
        mask = data.pop("mask", None)
        images.append(image)
        if mask:
            masks.append(mask)

    new_batch: dict = nerfstudio_collate(batch)
    new_batch["image"] = images
    if masks:
        new_batch["mask"] = masks

    return new_batch


@dataclass
class VariableResDataManagerConfig(VanillaDataManagerConfig):
    """A datamanager for variable resolution datasets, with presets to optimize
    for the fact that we are now dealing with lists of images and masks.
    """

    train_num_images_to_sample_from: int = 40
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = 100
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_images_to_sample_from: int = 40
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = 100
    collate_fn = staticmethod(variable_res_collate)
