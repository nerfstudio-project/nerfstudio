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

"""A set of standard datasets."""

from __future__ import annotations

import dataclasses
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from nerfactory.configs import base as cfg
from nerfactory.datamanagers.structs import DatasetInputs
from nerfactory.utils.io import get_absolute_path, load_from_pkl, write_to_pkl
from nerfactory.utils.misc import get_hash_str_from_dict


@dataclass
class DataParser:
    """A dataset."""

    def __init__(self, config: cfg.DataParserConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def _generate_dataset_inputs(self, split: str = "train") -> DatasetInputs:
        """Returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate.

        Returns:
            DatasetInputs
        """

    def get_dataset_inputs(self, split: str = "train", use_preprocessing_cache: bool = False) -> DatasetInputs:
        """Returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate.
            use_preprocessing_cache: Whether to use the cached dataset inputs. Defaults to False.

        Returns:
            DatasetInputs
        """
        if use_preprocessing_cache:
            dataset_inputs = self._get_dataset_inputs_from_cache(split)
        else:
            dataset_inputs = self._generate_dataset_inputs(split)
        return dataset_inputs

    def _get_cache_filename(self, split: str) -> Path:
        """Creates a cache filename from the dataset inputs arguments.

        Args:
            split: Which dataset split to generate filename for.

        Returns:
            filename for cache.
        """
        dataset_config_hash = get_hash_str_from_dict(dataclasses.asdict(self))
        dataset_config_hash_filename = get_absolute_path(
            f"cache/dataset_inputs/{dataset_config_hash}-{split}.pkl"
        ).parent.mkdir()
        assert dataset_config_hash_filename is not None, "dataset hash is None"
        return dataset_config_hash_filename

    def save_dataset_inputs_to_cache(self, split: str):
        """Saves the dataset inputs to cache.

        Args:
            split: Which dataset split to save.
        """
        dataset_inputs = self.get_dataset_inputs(split=split)
        dataset_config_hash_filename = self._get_cache_filename(split)
        write_to_pkl(dataset_config_hash_filename, dataset_inputs)

    def _get_dataset_inputs_from_cache(self, split: str) -> DatasetInputs:
        """Loads the dataset inputs from cache. If the cache does not exist, it will be created.

        Args:
            split: Which dataset split to load.
        """
        dataset_config_hash_filename = self._get_cache_filename(split)
        if dataset_config_hash_filename.exists():
            logging.info("Loading dataset from cache.")
            dataset_inputs = load_from_pkl(dataset_config_hash_filename)
        else:
            logging.info("Cache file not found. Generating and saving dataset to cache.")
            dataset_inputs = self._generate_dataset_inputs(split=split)
            self.save_dataset_inputs_to_cache(split)
        return dataset_inputs
