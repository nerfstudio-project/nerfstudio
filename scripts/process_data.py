"""Processes a video or image sequence to a nerfactory compatible dataset."""

import json
import subprocess
import sys
from enum import Enum
from pathlib import Path
from shutil import which
from typing import Literal, Optional

import dcargs
import numpy as np

from nerfactory.utils import colmap_utils


class CameraType(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraType.OPENCV,
    "fisheye": CameraType.OPENCV_FISHEYE,
}


def _print_red(text):
    """Prints text in red."""
    print(f"\033[91m{text}\033[00m")


def _print_green(text):
    """Prints text in green."""
    print(f"\033[92m{text}\033[00m")


def _print_section_header(text):
    """Prints a section header."""
    _print_green("=" * 80)
    _print_green(text)
    _print_green("=" * 80)


def check_ffmpeg_installed():
    """Checks if ffmpeg is installed."""
    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path is None:
        _print_red("Could not find ffmpeg. Please install ffmpeg.")
        print("See https://ffmpeg.org/download.html for installation instructions.")
        print("ffmpeg is only necissary if using videos as input.")
        sys.exit(1)


def check_colmap_installed():
    """Checks if colmap is installed."""
    colmap_path = which("colmap")
    if colmap_path is None:
        _print_red("Could not find COLMAP. Please install COLMAP.")
        print("See https://colmap.github.io/install.html for installation instructions.")
        sys.exit(1)


def get_colmap_version(default_version=3.8) -> float:
    """Returns the version of COLMAP.
    This code assumes that colmap returns a version string of the form
    "COLMAP 3.8 ..." which may not be true for all versions of COLMAP.

    Args:
        default_version: Default version to return if COLMAP version can't be determined.
    Returns:
        The version of COLMAP.
    """
    output = run_command("colmap", return_output=True)
    assert output is not None
    for line in output.split("\n"):
        if line.startswith("COLMAP"):
            return float(line.split(" ")[1])
    _print_red(f"Could not find COLMAP version. Using default {default_version}")
    return default_version


def run_command(cmd, return_output=False) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        return_output: If True, returns the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    if return_output:
        rc, output = subprocess.getstatusoutput(cmd)
        if rc != 0:
            _print_red(f"Error running command: {cmd}")
            _print_red(f"Got: {output}")
            sys.exit(1)
        return output
    rc = subprocess.run(cmd, shell=True, check=True).returncode
    if rc != 0:
        _print_red(f"Error running command: {cmd}")
        sys.exit(1)
    return None


def convert_video_to_images(video_path: Path, image_dir: Path, num_frames_target: int):
    """Converts a video into a sequence of images.

    Args:
        video_path: Path to the video.
        output_dir: Path to the output directory.
        fps: Frames per second.
    """
    cmd = f"ffprobe -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets -of csv=p=0 {video_path}"
    output = run_command(cmd, return_output=True)
    assert output is not None
    num_frames = int(output)
    print("Number of frames in video:", num_frames)

    out_filename = image_dir / "frame_%05d.png"

    ffmpeg_cmd = f"ffmpeg -i {video_path}"

    spacing = num_frames // num_frames_target

    if spacing > 1:
        ffmpeg_cmd += f" -vf thumbnail={spacing},setpts=N/TB -r 1"
    else:
        _print_red("Can't satify requested number of frames. Extracting all frames.")

    ffmpeg_cmd += f" {out_filename}"

    run_command(ffmpeg_cmd)


def run_colmap(image_dir: Path, colmap_dir: Path, camera_model: CameraType, gpu: bool = True) -> None:
    """Runs COLMAP on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        gpu: If True, use GPU.
    """

    colmap_version = get_colmap_version()

    # Feature extraction
    feature_extractor_cmd = [
        "colmap feature_extractor",
        f"--database_path {colmap_dir}/database.db",
        f"--image_path {image_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {camera_model.value}",
        f"--SiftExtraction.use_gpu {int(gpu)}",
    ]
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    run_command(feature_extractor_cmd)

    # Feature matching
    feature_matcher_cmd = [
        "colmap exhaustive_matcher",
        f"--database_path {colmap_dir}/database.db",
        f"--SiftMatching.use_gpu {int(gpu)}",
    ]
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    run_command(feature_matcher_cmd)

    # Bundle adjustment
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    bundle_adjuster_cmd = [
        "colmap mapper",
        f"--database_path {colmap_dir}/database.db",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]
    if colmap_version >= 3.7:
        bundle_adjuster_cmd.append("--Mapper.ba_global_function_tolerance 1e-6")

    bundle_adjuster_cmd = " ".join(bundle_adjuster_cmd)
    run_command(bundle_adjuster_cmd)


def colmap_to_json(cameras_path: Path, images_path: Path, output_dir: Path) -> None:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        cameras_path: Path to the cameras.bin file.
        images_path: Path to the images.bin file.
        output_dir: Path to the output directory.
    """

    cameras = colmap_utils.read_cameras_binary(cameras_path)
    images = colmap_utils.read_images_binary(images_path)

    # Only supports one camera
    camera_params = cameras[1].params

    frames = []
    for _, im_data in images.items():
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = f"./images/{im_data.name}"

        frame = {
            "file_path": name,
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[1]),
        "k1": float(camera_params[4]),
        "k2": float(camera_params[5]),
        "p1": float(camera_params[6]),
        "p2": float(camera_params[7]),
        "cx": float(camera_params[2]),
        "cy": float(camera_params[3]),
        "w": cameras[1].width,
        "h": cameras[1].height,
        "frames": frames,
    }

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)


def main(
    data: Path,
    output_dir: Path,
    num_frames_target: int,
    camera_type: Literal["perspective", "fisheye"] = "perspective",
    gpu: bool = True,
):
    """Process images or videos into a Nerfactory dataset.

    This script does the following:
    1) Converts video into images (if video is provided).
    2) Scales images to a specified size.
    3) Calculates and stores the sharpness of each image.
    4) Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.


    Args:
        data: Path the data, either a video file or a directory of images.
        output_dir: Path to the output directory.
        num_frames: Target number of frames to use for the dataset, results may not be exact.
        camera_type: Camera model to use.
        gpu: If True, use GPU.
    """

    print(data)

    check_ffmpeg_installed()
    check_colmap_installed()

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if data.is_file():
        _print_section_header("Converting video into images.")
        convert_video_to_images(data, image_dir=image_dir, num_frames_target=num_frames_target)

    _print_section_header("Running Colmap.")
    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    camera_model = CAMERA_MODELS[camera_type]
    run_colmap(image_dir=image_dir, colmap_dir=colmap_dir, camera_model=camera_model, gpu=gpu)

    _print_section_header("Saving output to JSON.")
    colmap_to_json(
        cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
        images_path=colmap_dir / "sparse" / "0" / "images.bin",
        output_dir=output_dir,
    )


if __name__ == "__main__":
    dcargs.cli(main)
