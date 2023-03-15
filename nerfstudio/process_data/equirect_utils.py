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

"""Helper utils for processing equirectangular data."""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from equilib import Equi2Pers
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


def _crop_bottom(bound_arr: list, fov: int, crop_factor: float) -> List[float]:
    """Returns a list of vertical bounds with the bottom cropped.

    Args:
        bound_arr (list): List of vertical bounds in ascending order.
        fov (int): Field of view of the camera.
        crop_factor (float): Portion of the image to crop from the bottom.

    Returns:
        list: A new list of bounds with the bottom cropped.
    """
    degrees_chopped = 180 * crop_factor
    new_bottom_start = 90 - degrees_chopped - fov / 2
    for i, el in reversed(list(enumerate(bound_arr))):
        if el > new_bottom_start + fov / 2:
            bound_arr[i] = None
        elif el > new_bottom_start:
            diff = el - new_bottom_start
            bound_arr[i] = new_bottom_start
            for j in range(i - 1, -1, -1):
                bound_arr[j] -= diff / (2 ** (i - j))
            break

    return bound_arr


def _crop_top(bound_arr: list, fov: int, crop_factor: float) -> List[float]:
    """Returns a list of vertical bounds with the top cropped.

    Args:
        bound_arr (list): List of vertical bounds in ascending order.
        fov (int): Field of view of the camera.
        crop_factor (float): Portion of the image to crop from the top.

    Returns:
        list: A new list of bounds with the top cropped.
    """
    degrees_chopped = 180 * crop_factor
    new_top_start = -90 + degrees_chopped + fov / 2
    for i, el in enumerate(bound_arr):
        if el < new_top_start - fov / 2:
            bound_arr[i] = None
        elif el < new_top_start:
            diff = new_top_start - el
            bound_arr[i] = new_top_start
            for j in range(i + 1, len(bound_arr)):
                bound_arr[j] += diff / (2 ** (j - i))
            break

    return bound_arr


def _crop_bound_arr_vertical(
    bound_arr: list, fov: int, crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
) -> list:
    """Returns a list of vertical bounds adjusted for cropping.

    Args:
        bound_arr (list): Original list of vertical bounds in ascending order.
        fov (int): Field of view of the camera.
        crop_factor (Tuple[float, float, float, float]): Crop arr (top, bottom, left, right).

    Returns:
        list: Cropped bound arr
    """
    if crop_factor[1] > 0:
        bound_arr = _crop_bottom(bound_arr, fov, crop_factor[1])
    if crop_factor[0] > 0:
        bound_arr = _crop_top(bound_arr, fov, crop_factor[0])
    return bound_arr


def generate_planar_projections_from_equirectangular(
    image_dir: Path,
    planar_image_size: Tuple[int, int],
    samples_per_im: int,
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> Path:
    """Generate planar projections from an equirectangular image.

    Args:
        image_dir: The directory containing the equirectangular image.
        planar_image_size: The size of the planar projections [width, height].
        samples_per_im: The number of samples to take per image.
        crop_factor: The portion of the image to crop from the (top, bottom, left, and right).
                    Values should be in [0, 1].
    returns:
        The path to the planar projections directory.
    """

    for i in crop_factor:
        if i < 0 or i > 1:
            CONSOLE.print("[bold red] Invalid crop factor. All values must be in [0,1].")
            sys.exit(1)

    device = torch.device("cuda")

    fov = 120
    yaw_pitch_pairs = []
    left_bound, right_bound = -180, 180
    if crop_factor[3] > 0:
        left_bound = -180 + 360 * crop_factor[3]
    if crop_factor[2] > 0:
        right_bound = 180 - 360 * crop_factor[2]

    if samples_per_im == 8:
        fov = 120
        bound_arr = [-45, 0, 45]
        bound_arr = _crop_bound_arr_vertical(bound_arr, fov, crop_factor)
        if bound_arr[1] is not None:
            for i in np.arange(left_bound, right_bound, 90):
                yaw_pitch_pairs.append((i, bound_arr[1]))
        if bound_arr[2] is not None:
            for i in np.arange(left_bound, right_bound, 180):
                yaw_pitch_pairs.append((i, bound_arr[2]))
        if bound_arr[0] is not None:
            for i in np.arange(left_bound, right_bound, 180):
                yaw_pitch_pairs.append((i, bound_arr[0]))
    elif samples_per_im == 14:
        fov = 110
        bound_arr = [-45, 0, 45]
        bound_arr = _crop_bound_arr_vertical(bound_arr, fov, crop_factor)
        if bound_arr[1] is not None:
            for i in np.arange(left_bound, right_bound, 60):
                yaw_pitch_pairs.append((i, bound_arr[1]))
        if bound_arr[2] is not None:
            for i in np.arange(left_bound, right_bound, 90):
                yaw_pitch_pairs.append((i, bound_arr[2]))
        if bound_arr[0] is not None:
            for i in np.arange(left_bound, right_bound, 90):
                yaw_pitch_pairs.append((i, bound_arr[0]))

    equi2pers = Equi2Pers(height=planar_image_size[1], width=planar_image_size[0], fov_x=fov, mode="bilinear")
    frame_dir = image_dir
    output_dir = image_dir / "planar_projections"
    output_dir.mkdir(exist_ok=True)
    num_ims = len(os.listdir(frame_dir))
    progress = Progress(
        TextColumn("[bold blue]Generating Planar Images", justify="right"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="equirect frames/s"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )

    with progress:
        for i in progress.track(os.listdir(frame_dir), description="", total=num_ims):
            if i.lower().endswith((".jpg", ".png", ".jpeg")):
                im = np.array(cv2.imread(os.path.join(frame_dir, i)))
                im = torch.tensor(im, dtype=torch.float32, device=device)
                im = torch.permute(im, (2, 0, 1)) / 255.0
                count = 0
                for u_deg, v_deg in yaw_pitch_pairs:
                    v_rad = torch.pi * v_deg / 180.0
                    u_rad = torch.pi * u_deg / 180.0
                    pers_image = equi2pers(im, rots={"roll": 0, "pitch": v_rad, "yaw": u_rad}) * 255.0
                    pers_image = (pers_image.permute(1, 2, 0)).type(torch.uint8).to("cpu").numpy()
                    cv2.imwrite(f"{output_dir}/{i[:-4]}_{count}.jpg", pers_image)
                    count += 1

    return output_dir


def compute_resolution_from_equirect(image_dir: Path, num_images: int) -> Tuple[int, int]:
    """Compute the resolution of the persepctive projections of equirectangular images
       from the heuristic: num_image * res**2 = orig_height * orig_width.

    Args:
        image_dir: The directory containing the equirectangular images.
    returns:
        The target resolution of the perspective projections.
    """

    for i in os.listdir(image_dir):
        if i.lower().endswith((".jpg", ".png", ".jpeg")):
            im = np.array(cv2.imread(os.path.join(image_dir, i)))
            res_squared = (im.shape[0] * im.shape[1]) / num_images
            return (int(np.sqrt(res_squared)), int(np.sqrt(res_squared)))
    raise ValueError("No images found in the directory.")
