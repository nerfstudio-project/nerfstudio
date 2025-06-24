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
"""Data parser utils for nerfstudio datasets."""

import math
import os
from typing import List, Tuple

import numpy as np


def get_train_eval_split_fraction(image_filenames: List, train_split_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split fraction based on the number of images and the train split fraction.

    Args:
        image_filenames: list of image filenames
        train_split_fraction: fraction of images to use for training
    """

    # filter image_filenames and poses based on train/eval split percentage
    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images

    return i_train, i_eval


def get_train_eval_split_filename(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the filename of the images.

    Args:
        image_filenames: list of image filenames
    """

    num_images = len(image_filenames)
    basenames = [os.path.basename(image_filename) for image_filename in image_filenames]
    i_all = np.arange(num_images)
    i_train = []
    i_eval = []
    for idx, basename in zip(i_all, basenames):
        # check the frame index
        if "train" in basename:
            i_train.append(idx)
        elif "eval" in basename:
            i_eval.append(idx)
        else:
            raise ValueError("frame should contain train/eval in its name to use this eval-frame-index eval mode")

    return np.array(i_train), np.array(i_eval)


def get_train_eval_split_interval(image_filenames: List, eval_interval: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the interval of the images.

    Args:
        image_filenames: list of image filenames
        eval_interval: interval of images to use for eval
    """

    num_images = len(image_filenames)
    all_indices = np.arange(num_images)
    train_indices = all_indices[all_indices % eval_interval != 0]
    eval_indices = all_indices[all_indices % eval_interval == 0]
    i_train = train_indices
    i_eval = eval_indices

    return i_train, i_eval


def get_train_eval_split_all(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split where all indices are used for both train and eval.

    Args:
        image_filenames: list of image filenames
    """
    num_images = len(image_filenames)
    i_all = np.arange(num_images)
    i_train = i_all
    i_eval = i_all
    return i_train, i_eval
