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
Code for loading the dataset inputs and for caching the data for fast loading.
"""

import logging
from typing import Optional, Union

from omegaconf import ListConfig

from pyrad.data.format.blender import load_blender_data
from pyrad.data.format.friends import load_friends_data
from pyrad.data.format.instant_ngp import load_instant_ngp_data
from pyrad.data.format.mipnerf_360 import load_mipnerf_360_data
from pyrad.data.structs import DatasetInputs
from pyrad.utils.colors import get_color
from pyrad.utils.io import get_absolute_path, load_from_pkl, make_dir, write_to_pkl
from pyrad.utils.misc import get_hash_str_from_dict


def get_cache_filename_from_kwargs(kwargs: dict, split: str):
    """Creates a cache filename from the dataset inputs arguments."""
    dataset_config_hash = get_hash_str_from_dict(kwargs)
    dataset_config_hash_filename = make_dir(
        get_absolute_path(f"cache/dataset_inputs/{dataset_config_hash}-{split}.pkl")
    )
    return dataset_config_hash_filename


def save_dataset_inputs_kwargs_to_cache(kwargs: dict, split: str):
    """Saves the dataset inputs to cache."""
    dataset_inputs = get_dataset_inputs(**kwargs, split=split)
    dataset_config_hash_filename = get_cache_filename_from_kwargs(kwargs, split)
    write_to_pkl(dataset_config_hash_filename, dataset_inputs)


def get_dataset_inputs_from_cache(kwargs: dict, split: str):
    """Loads the dataset inputs from cache."""
    dataset_config_hash_filename = get_cache_filename_from_kwargs(kwargs, split)
    return load_from_pkl(dataset_config_hash_filename)


def get_dataset_inputs(
    data_directory: str,
    dataset_format: str,
    split: str,
    downscale_factor: int = 1,
    alpha_color: Optional[Union[str, list, ListConfig]] = None,
) -> DatasetInputs:
    """Returns the dataset inputs, which will be used with an ImageDataset and RayGenerator.

    Args:
        data_directory (str): Location of data
        dataset_format (str): Name of dataset type
        downscale_factor (int, optional): How much to downscale images. Defaults to 1.0.
        alpha_color (str, list, optional): Sets transparent regions to specified color, otherwise black.
        load_from_cache (bool)
    Returns:
        DatasetInputs: The inputs needed for generating rays.
    """

    if alpha_color is not None:
        alpha_color = get_color(alpha_color)

    if dataset_format == "blender":
        dataset_inputs = load_blender_data(
            get_absolute_path(data_directory),
            downscale_factor=downscale_factor,
            split=split,
            alpha_color=alpha_color,
        )
    elif dataset_format == "friends":
        # TODO(ethan): change this hack, and do proper splits for the friends dataset
        # currently we assume that there is only the training set of images
        dataset_inputs = load_friends_data(
            get_absolute_path(data_directory),
            downscale_factor=downscale_factor,
            split="train",
            include_point_cloud=False,
        )
    elif dataset_format == "instant_ngp":
        dataset_inputs = load_instant_ngp_data(
            get_absolute_path(data_directory),
            downscale_factor=downscale_factor,
            split="train",
        )
    elif dataset_format == "mipnerf_360":
        dataset_inputs = load_mipnerf_360_data(
            get_absolute_path(data_directory),
            downscale_factor=downscale_factor,
            split="train",
        )
    else:
        raise NotImplementedError(f"{dataset_format} is not a valid dataset type")

    return dataset_inputs


def get_dataset_inputs_from_dataset_config(*, split: str, use_cache: bool = False, **kwargs):
    """This may optionally use the cache to return the dataset inputs."""
    if use_cache:
        logging.info("Loading from cache! Be careful with using this when making changes!")
        print(kwargs)
        return get_dataset_inputs_from_cache(kwargs, split)
    return get_dataset_inputs(**kwargs, split=split)
