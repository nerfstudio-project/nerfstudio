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

"""Helper utils for processing data into the nerfstudio format."""

import os
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
}


def get_num_frames_in_video(video: Path) -> int:
    """Returns the number of frames in a video.

    Args:
        video: Path to a video.

    Returns:
        The number of frames in a video.
    """
    cmd = f"ffprobe -v error -select_streams v:0 -count_packets \
            -show_entries stream=nb_read_packets -of csv=p=0 {video}"
    output = run_command(cmd)
    assert output is not None
    output = output.strip(" ,\t\n\r")
    return int(output)


def convert_video_to_images(
    video_path: Path, image_dir: Path, num_frames_target: int, verbose: bool = False
) -> Tuple[List[str], int]:
    """Converts a video into a sequence of images.

    Args:
        video_path: Path to the video.
        output_dir: Path to the output directory.
        num_frames_target: Number of frames to extract.
        verbose: If True, logs the output of the command.
    Returns:
        A tuple containing summary of the conversion and the number of extracted frames.
    """

    with status(msg="Converting video to images...", spinner="bouncingBall", verbose=verbose):
        # delete existing images in folder
        for img in image_dir.glob("*.png"):
            if verbose:
                CONSOLE.log(f"Deleting {img}")
            img.unlink()

        num_frames = get_num_frames_in_video(video_path)
        if num_frames == 0:
            CONSOLE.print(f"[bold red]Error: Video has no frames: {video_path}")
            sys.exit(1)
        print("Number of frames in video:", num_frames)

        out_filename = image_dir / "frame_%05d.png"
        ffmpeg_cmd = f"ffmpeg -i {video_path}"
        spacing = num_frames // num_frames_target

        if spacing > 1:
            ffmpeg_cmd += f" -vf thumbnail={spacing},setpts=N/TB -r 1"
        else:
            CONSOLE.print("[bold red]Can't satisfy requested number of frames. Extracting all frames.")
            ffmpeg_cmd += " -pix_fmt bgr8"

        ffmpeg_cmd += f" {out_filename}"

        run_command(ffmpeg_cmd, verbose=verbose)

    num_final_frames = len(list(image_dir.glob("*.png")))
    summary_log = []
    summary_log.append(f"Starting with {num_frames} video frames")
    summary_log.append(f"We extracted {num_final_frames} images")
    CONSOLE.log("[bold green]:tada: Done converting video to images.")

    return summary_log, num_final_frames


def copy_images_list(
    image_paths: List[Path], image_dir: Path, crop_border_pixels: Optional[int] = None, verbose: bool = False
) -> List[Path]:
    """Copy all images in a list of Paths. Useful for filtering from a directory.
    Args:
        image_paths: List of Paths of images to copy to a new directory.
        image_dir: Path to the output directory.
        crop_border_pixels: If not None, crops each edge by the specified number of pixels.
        verbose: If True, print extra logging.
    Returns:
        A list of the copied image Paths.
    """

    # Remove original directory only if we provide a proper image folder path
    if image_dir.is_dir() and len(image_paths):
        shutil.rmtree(image_dir, ignore_errors=True)
        image_dir.mkdir(exist_ok=True, parents=True)

    copied_image_paths = []

    # Images should be 1-indexed for the rest of the pipeline.
    for idx, image_path in enumerate(image_paths):
        if verbose:
            CONSOLE.log(f"Copying image {idx + 1} of {len(image_paths)}...")
        copied_image_path = image_dir / f"frame_{idx + 1:05d}{image_path.suffix}"
        shutil.copy(image_path, copied_image_path)
        copied_image_paths.append(copied_image_path)

    if crop_border_pixels is not None:
        file_type = image_paths[0].suffix
        filename = f"frame_%05d{file_type}"
        crop = f"crop=iw-{crop_border_pixels*2}:ih-{crop_border_pixels*2}"
        ffmpeg_cmd = f"ffmpeg -y -i {image_dir / filename} -q:v 2 -vf {crop} {image_dir / filename}"
        run_command(ffmpeg_cmd, verbose=verbose)

    num_frames = len(image_paths)

    if num_frames == 0:
        CONSOLE.log("[bold red]:skull: No usable images in the data folder.")
    else:
        CONSOLE.log("[bold green]:tada: Done copying images.")

    return copied_image_paths


def copy_images(data: Path, image_dir: Path, verbose) -> int:
    """Copy images from a directory to a new directory.

    Args:
        data: Path to the directory of images.
        image_dir: Path to the output directory.
        verbose: If True, print extra logging.
    Returns:
        The number of images copied.
    """
    with status(msg="[bold yellow]Copying images...", spinner="bouncingBall", verbose=verbose):
        allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        image_paths = sorted([p for p in data.glob("[!.]*") if p.suffix.lower() in allowed_exts])

        if len(image_paths) == 0:
            CONSOLE.log("[bold red]:skull: No usable images in the data folder.")
            sys.exit(1)

        num_frames = len(copy_images_list(image_paths, image_dir, verbose))

    return num_frames


def downscale_images(image_dir: Path, num_downscales: int, verbose: bool = False) -> str:
    """Downscales the images in the directory. Uses FFMPEG.

    Assumes images are named frame_00001.png, frame_00002.png, etc.

    Args:
        image_dir: Path to the directory containing the images.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time.
        verbose: If True, logs the output of the command.

    Returns:
        Summary of downscaling.
    """

    if num_downscales == 0:
        return "No downscaling performed."

    with status(msg="[bold yellow]Downscaling images...", spinner="growVertical", verbose=verbose):
        downscale_factors = [2**i for i in range(num_downscales + 1)[1:]]
        for downscale_factor in downscale_factors:
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)
            downscale_dir = image_dir.parent / f"images_{downscale_factor}"
            downscale_dir.mkdir(parents=True, exist_ok=True)
            # Using %05d ffmpeg commands appears to be unreliable (skips images), so use scandir.
            files = os.scandir(image_dir)
            for f in files:
                filename = f.name
                ffmpeg_cmd = [
                    f"ffmpeg -y -i {image_dir / filename} ",
                    f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor} ",
                    f"{downscale_dir / filename}",
                ]
                ffmpeg_cmd = " ".join(ffmpeg_cmd)
                run_command(ffmpeg_cmd, verbose=verbose)

    CONSOLE.log("[bold green]:tada: Done downscaling images.")
    downscale_text = [f"[bold blue]{2**(i+1)}x[/bold blue]" for i in range(num_downscales)]
    downscale_text = ", ".join(downscale_text[:-1]) + " and " + downscale_text[-1]
    return f"We downsampled the images by {downscale_text}"


def find_tool_feature_matcher_combination(
    sfm_tool: Literal["any", "colmap", "hloc"],
    feature_type: Literal[
        "any",
        "sift",
        "superpoint",
        "superpoint_aachen",
        "superpoint_max",
        "superpoint_inloc",
        "r2d2",
        "d2net-ss",
        "sosnet",
        "disk",
    ],
    matcher_type: Literal[
        "any", "NN", "superglue", "superglue-fast", "NN-superpoint", "NN-ratio", "NN-mutual", "adalam"
    ],
):
    """Find a valid combination of sfm tool, feature type, and matcher type.
    Basically, replace the default parameters 'any' by usable value

    Args:
        sfm_tool: Sfm tool name (any, colmap, hloc)
        feature_type: Type of image features (any, sift, superpoint, ...)
        matcher_type: Type of matching algorithm (any, NN, superglue,...)

    Returns:
        Tuple of sfm tool, feature type, and matcher type.
        Returns (None,None,None) if no valid combination can be found
    """
    if sfm_tool == "any":
        if (feature_type in ("any", "sift")) and (matcher_type in ("any", "NN")):
            sfm_tool = "colmap"
        else:
            sfm_tool = "hloc"

    if sfm_tool == "colmap":
        if (feature_type not in ("any", "sift")) or (matcher_type not in ("any", "NN")):
            return (None, None, None)
        return ("colmap", "sift", "NN")
    if sfm_tool == "hloc":
        if feature_type in ("any", "superpoint"):
            feature_type = "superpoint_aachen"

        if matcher_type == "any":
            matcher_type = "superglue"
        elif matcher_type == "NN":
            matcher_type = "NN-mutual"

        return (sfm_tool, feature_type, matcher_type)
    return (None, None, None)


def generate_circle_mask(height: int, width: int, percent_radius) -> Optional[np.ndarray]:
    """generate a circle mask of the given size.

    Args:
        height: The height of the mask.
        width: The width of the mask.
        percent_radius: The radius of the circle as a percentage of the image diagonal size.

    Returns:
        The mask or None if the radius is too large.
    """
    if percent_radius <= 0.0:
        CONSOLE.log("[bold red]:skull: The radius of the circle mask must be positive.")
        sys.exit(1)
    if percent_radius >= 1.0:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(percent_radius * np.sqrt(width**2 + height**2) / 2.0)
    cv2.circle(mask, center, radius, 1, -1)
    return mask


def generate_crop_mask(
    height: int, width: int, percent_crop: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """generate a crop mask of the given size.

    Args:
        height: The height of the mask.
        width: The width of the mask.
        percent_crop: The percent of the image to crop in each direction [top, bottom, left, right].

    Returns:
        The mask or None if no cropping is performed.
    """
    if np.all(np.array(percent_crop) == 0.0):
        return None
    if np.any(np.array(percent_crop) < 0.0) or np.any(np.array(percent_crop) > 1.0):
        CONSOLE.log("[bold red]Invalid crop percentage, must be between 0 and 1.")
        sys.exit(1)
    top, bottom, left, right = percent_crop
    mask = np.zeros((height, width), dtype=np.uint8)
    top = int(top * height)
    bottom = int(bottom * height)
    left = int(left * width)
    right = int(right * width)
    mask[top : height - bottom, left : width - right] = 1.0
    return mask


def generate_mask(
    height: int, width: int, percent_crop: Tuple[float, float, float, float], percent_radius: float
) -> Optional[np.ndarray]:
    """generate a mask of the given size.

    Args:
        height: The height of the mask.
        width: The width of the mask.
        percent_crop: The percent of the image to crop in each direction [top, bottom, left, right].
        percent_radius: The radius of the circle as a percentage of the image diagonal size.

    Returns:
        The mask or None if no mask is needed.
    """
    crop_mask = generate_crop_mask(height, width, percent_crop)
    circle_mask = generate_circle_mask(height, width, percent_radius)
    if crop_mask is None:
        return circle_mask
    if circle_mask is None:
        return crop_mask
    return crop_mask * circle_mask


def save_mask(
    image_dir: Path,
    num_downscales: int,
    percent_crop: Tuple[float, float, float, float] = (0, 0, 0, 0),
    percent_radius: float = 1.0,
) -> Optional[Path]:
    """Save a mask for each image in the image directory.

    Args:
        image_dir: The directory containing the images.
        num_downscales: The number of downscaling levels.
        percent_crop: The percent of the image to crop in each direction [top, bottom, left, right].
        percent_radius: The radius of the circle as a percentage of the image diagonal size.

    Returns:
        The path to the mask file or None if no mask is needed.
    """
    image_path = next(image_dir.glob("frame_*"))
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    mask = generate_mask(height, width, percent_crop, percent_radius)
    if mask is None:
        return None
    mask *= 255
    mask_path = image_dir.parent / "masks"
    mask_path.mkdir(exist_ok=True)
    cv2.imwrite(str(mask_path / "mask.png"), mask)
    downscale_factors = [2**i for i in range(num_downscales + 1)[1:]]
    for downscale in downscale_factors:
        mask_path_i = image_dir.parent / f"masks_{downscale}"
        mask_path_i.mkdir(exist_ok=True)
        mask_path_i = mask_path_i / "mask.png"
        mask_i = cv2.resize(mask, (width // downscale, height // downscale), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(mask_path_i), mask_i)
    CONSOLE.log(":tada: Generated and saved masks.")
    return mask_path / "mask.png"
