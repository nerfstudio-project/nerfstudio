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

"""Helper utils for processing equirectangular data."""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn


# https://gist.github.com/fgolemo/94b5caf0e209a6e71ab0ce2d75ad3ed8
def euler_rodriguez_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Generates a 3x3 rotation matrix from an axis and angle using Euler-Rodriguez formula.

    Args:
        axis (torch.Tensor): Axis about which to rotate.
        theta (torch.Tloat): Angle to rotate by.

    Returns:
        torch.Tensor: 3x3 Rotation matrix.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def remap_cubic(
    img: torch.Tensor, map_x: torch.Tensor, map_y: torch.Tensor, border_mode: str = "border"
) -> torch.Tensor:
    """Remap image using bicubic interpolation.

    Args:
        img (torch.Tensor): Image tensor
        map_x (torch.Tensor): x mapping
        map_y (torch.Tensor): y mapping
        border_mode (str, optional): What to do with borders. Defaults to "border".

    Returns:
        torch.Tensor: _description_
    """
    batch_size, channels, height, width = img.shape

    grid_x = (map_x / width + 1) * 2 - 1
    grid_y = (map_y / height + 1) * 2 - 1

    if border_mode == "border":
        grid_x = torch.clamp(grid_x, -1, 1)
        grid_y = torch.clamp(grid_y, -1, 1)
    elif border_mode == "wrap":
        grid_x = torch.remainder(grid_x + 1, 2) - 1
        grid_y = torch.remainder(grid_y + 1, 2) - 1

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    return torch.nn.functional.grid_sample(img, grid, mode="bicubic", padding_mode="zeros")


def equirect2persp(img: torch.Tensor, fov: int, theta: int, phi: int, hd: int, wd: int) -> torch.Tensor:
    """Pytorch reimlement of https://github.com/kaustubh-sadekar/OmniCV-Lib for equirectangular to perspective projection.

    Args:
        img (torch.Tensor): Image tensor
        fov (int): Horizontal field of view in degrees
        theta (int): Horizontal angle in degrees
        phi (int): Vertical angle in degrees
        hd (int): Number of pixels in height
        wd (int): Number of pixels in width

    Returns:
        torch.Tensor: Planar image tensor
    """
    device = img.device
    theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device)
    phi_tensor = torch.tensor(phi, dtype=torch.float32, device=device)
    # theta is left/right angle, phi is up/down angle, both in degree
    equ_h, equ_w = img.shape[2:]

    equ_cx = (equ_w) / 2.0
    equ_cy = (equ_h) / 2.0

    wfov = fov
    hfov = float(hd) / wd * wfov

    c_x = (wd) / 2.0
    c_y = (hd) / 2.0

    w_len = 2 * torch.tan(torch.deg2rad(torch.tensor(wfov / 2.0, device=device)))
    w_interval = w_len / wd

    h_len = 2 * torch.tan(torch.deg2rad(torch.tensor(hfov / 2.0, device=device)))
    h_interval = h_len / hd

    x_map = torch.zeros([hd, wd], dtype=torch.float32, device=device) + 1
    y_map = torch.tile((torch.arange(0, wd, device=device) - c_x) * w_interval, [hd, 1])
    z_map = -torch.tile((torch.arange(0, hd, device=device) - c_y) * h_interval, [wd, 1]).T
    D = torch.sqrt(x_map**2 + y_map**2 + z_map**2)

    xyz = torch.zeros([hd, wd, 3], dtype=torch.float32, device=device)
    xyz[:, :, 0] = (x_map / D)[:, :]
    xyz[:, :, 1] = (y_map / D)[:, :]
    xyz[:, :, 2] = (z_map / D)[:, :]

    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    R1 = euler_rodriguez_rotation_matrix(z_axis, torch.deg2rad(theta_tensor)).to(device)
    R2 = euler_rodriguez_rotation_matrix(torch.mm(R1, y_axis.view(3, 1)).squeeze(), torch.deg2rad(-phi_tensor)).to(
        device
    )

    xyz = xyz.view(hd * wd, 3).T
    xyz = torch.mm(R1, xyz)
    xyz = torch.mm(R2, xyz).T
    lat = torch.arcsin(xyz[:, 2] / 1)
    lon = torch.zeros([hd * wd], dtype=torch.float32, device=device)
    theta_tensor = torch.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0
    idx3 = ~idx1 & idx2
    idx4 = ~idx1 & ~idx2

    lon[idx1] = theta_tensor[idx1]
    lon[idx3] = theta_tensor[idx3] + np.pi
    lon[idx4] = theta_tensor[idx4] - np.pi

    lon = lon.view(hd, wd) / torch.pi * 180
    lat = -lat.view(hd, wd) / torch.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    return remap_cubic(img, lon, lat, border_mode="wrap")


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
                im = torch.permute(im, (2, 0, 1)).unsqueeze(0) / 255.0
                count = 0
                for u_deg, v_deg in yaw_pitch_pairs:
                    omnicv_pers_tensor = (
                        equirect2persp(im, fov, u_deg, v_deg, planar_image_size[1], planar_image_size[0]) * 255.0
                    )
                    pers_image = omnicv_pers_tensor.squeeze().permute(1, 2, 0).type(torch.uint8).to("cpu").numpy()
                    cv2.imwrite(f"{output_dir}/{i[:-4]}_{count}.jpg", pers_image)
                    count += 1

    return output_dir


def compute_resolution_from_equirect(image_dir: Path, num_images: int) -> Tuple[int, int]:
    """Compute the resolution of the perspective projections of equirectangular images
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
