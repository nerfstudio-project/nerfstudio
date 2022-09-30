#!/usr/bin/env python
"""Processes a video or image sequence to a nerfactory compatible dataset."""

import json
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Tuple

import appdirs
import dcargs
import numpy as np
import requests
from rich.console import Console
from rich.progress import track

from nerfactory.utils import colmap_utils

CONSOLE = Console(width=120)


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
}


def check_ffmpeg_installed():
    """Checks if ffmpeg is installed."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        CONSOLE.print("[bold red]Could not find ffmpeg. Please install ffmpeg.")
        print("See https://ffmpeg.org/download.html for installation instructions.")
        print("ffmpeg is only necissary if using videos as input.")
        sys.exit(1)


def check_colmap_installed():
    """Checks if colmap is installed."""
    colmap_path = shutil.which("colmap")
    if colmap_path is None:
        CONSOLE.print("[bold red]Could not find COLMAP. Please install COLMAP.")
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
    output = run_command("colmap", verbose=False)
    assert output is not None
    for line in output.split("\n"):
        if line.startswith("COLMAP"):
            return float(line.split(" ")[1])
    CONSOLE.print(f"[bold red]Could not find COLMAP version. Using default {default_version}")
    return default_version


def get_vocab_tree() -> Path:
    """Return path to vocab tree. Downloads vocab tree if it doesn't exist.

    Returns:
        The path to the vocab tree.
    """
    vocab_tree_filename = Path(appdirs.user_data_dir("nerfactory")) / "vocab_tree.fbow"

    if not vocab_tree_filename.exists():
        r = requests.get("https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin", stream=True)
        vocab_tree_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_tree_filename, "wb") as f:
            total_length = r.headers.get("content-length")
            assert total_length is not None
            for chunk in track(
                r.iter_content(chunk_size=1024),
                total=int(total_length) / 1024 + 1,
                description="Downloading vocab tree...",
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return vocab_tree_filename


def run_command(cmd, verbose=False) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        CONSOLE.rule("[bold red] :skull: :skull: :skull: ERROR :skull: :skull: :skull: ", style="red")
        CONSOLE.print(f"[bold red]Error running command: {cmd}")
        CONSOLE.rule(style="red")
        CONSOLE.print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


def convert_video_to_images(
    video_path: Path, image_dir: Path, num_frames_target: int, verbose: bool = False
) -> Tuple[int, int]:
    """Converts a video into a sequence of images.

    Args:
        video_path: Path to the video.
        output_dir: Path to the output directory.
        num_frames_target: Number of frames to extract.
        verbose: If True, logs the output of the command.
    Returns:
        A tuple containing the number of frames in the video and the number of frames extracted.
    """

    # delete existing images in folder
    for img in image_dir.glob("*.png"):
        if verbose:
            CONSOLE.log(f"Deleting {img}")
        img.unlink()

    cmd = f"ffprobe -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets -of csv=p=0 {video_path}"
    output = run_command(cmd, verbose=False)
    assert output is not None
    output = output.strip(" ,\t\n\r")

    num_frames = int(output)
    print("Number of frames in video:", num_frames)

    out_filename = image_dir / "frame_%05d.png"

    ffmpeg_cmd = f"ffmpeg -i {video_path}"

    spacing = num_frames // num_frames_target

    if spacing > 1:
        ffmpeg_cmd += f" -vf thumbnail={spacing},setpts=N/TB -r 1"
    else:
        CONSOLE.print("[bold red]Can't satify requested number of frames. Extracting all frames.")

    ffmpeg_cmd += f" {out_filename}"

    run_command(ffmpeg_cmd, verbose=verbose)

    return num_frames, len(list(image_dir.glob("*.png")))


def copy_images(data, image_dir, verbose) -> int:
    """Copy images from a directory to a new directory.

    Args:
        data: Path to the directory of images.
        image_dir: Path to the output directory.
        verbose: If True, print extra logging.
    Returns:
        The number of images copied.
    """
    image_paths = sorted(data.glob("*"))
    for i, image_path in enumerate(image_paths):
        i = i + 1  # 1-indexed
        if verbose:
            CONSOLE.log(f"Copying image {i + 1} of {len(image_paths)}...")
        shutil.copy(image_path, image_dir / f"frame_{i:05d}{image_path.suffix}")

    return len(image_paths)


def downscale_images(image_dir: Path, num_downscales: int, verbose: bool = False) -> None:
    """Downscales the images in the directory. Uses FFMPEG.

    Assumes images are named frame_00001.png, frame_00002.png, etc.

    Args:
        image_dir: Path to the directory containing the images.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time.
        verbose: If True, logs the output of the command.
    """
    downscale_factors = [2**i for i in range(num_downscales + 1)[1:]]
    for downscale_factor in downscale_factors:
        assert downscale_factor > 1
        assert isinstance(downscale_factor, int)
        downscale_dir = image_dir.parent / f"images_{downscale_factor}"
        downscale_dir.mkdir(parents=True, exist_ok=True)
        file_type = image_dir.glob("frame_*").__next__().suffix
        filename = f"frame_%05d{file_type}"
        ffmpeg_cmd = [
            f"ffmpeg -i {image_dir / filename} ",
            f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor} ",
            f"{downscale_dir / filename}",
        ]
        ffmpeg_cmd = " ".join(ffmpeg_cmd)
        run_command(ffmpeg_cmd, verbose=verbose)


def run_colmap(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    gpu: bool = True,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
) -> None:
    """Runs COLMAP on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
    """

    colmap_version = get_colmap_version()

    (colmap_dir / "database.db").unlink(missing_ok=True)

    # Feature extraction
    feature_extractor_cmd = [
        "colmap feature_extractor",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {camera_model.value}",
        f"--SiftExtraction.use_gpu {int(gpu)}",
    ]
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    if not verbose:
        with CONSOLE.status("[bold yellow]Running COLMAP feature extractor...", spinner="moon"):
            run_command(feature_extractor_cmd, verbose=verbose)
    else:
        run_command(feature_extractor_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done extracting COLMAP features.")

    # Feature matching
    feature_matcher_cmd = [
        f"colmap {matching_method}_matcher",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--SiftMatching.use_gpu {int(gpu)}",
    ]
    if matching_method == "vocab_tree":
        vocab_tree_filename = get_vocab_tree()
        feature_matcher_cmd.append(f"--VocabTreeMatching.vocab_tree_path {vocab_tree_filename}")
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    if not verbose:
        with CONSOLE.status("[bold yellow]Running COLMAP feature matcher...", spinner="runner"):
            run_command(feature_matcher_cmd, verbose=verbose)
    else:
        run_command(feature_matcher_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done matching COLMAP features.")

    # Bundle adjustment
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    bundle_adjuster_cmd = [
        "colmap mapper",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]
    if colmap_version >= 3.7:
        bundle_adjuster_cmd.append("--Mapper.ba_global_function_tolerance 1e-6")

    bundle_adjuster_cmd = " ".join(bundle_adjuster_cmd)
    if not verbose:
        with CONSOLE.status(
            "[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
            spinner="clock",
        ):
            run_command(bundle_adjuster_cmd, verbose=verbose)
    else:
        run_command(bundle_adjuster_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment.")


def colmap_to_json(cameras_path: Path, images_path: Path, output_dir: Path, camera_model: CameraModel) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        cameras_path: Path to the cameras.bin file.
        images_path: Path to the images.bin file.
        output_dir: Path to the output directory.
        camera_model: Camera model used.

    Returns:
        The number of registered images.
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

        name = Path(f"./images/{im_data.name}")

        frame = {
            "file_path": str(name),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[1]),
        "cx": float(camera_params[2]),
        "cy": float(camera_params[3]),
        "w": cameras[1].width,
        "h": cameras[1].height,
        "camera_model": camera_model.value,
    }

    if camera_model == CameraModel.OPENCV:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "p1": float(camera_params[6]),
                "p2": float(camera_params[7]),
            }
        )
    if camera_model == CameraModel.OPENCV_FISHEYE:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "k3": float(camera_params[6]),
                "k4": float(camera_params[7]),
            }
        )

    out["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)


# pylint: disable=too-many-statements
def main(
    data: Path,
    output_dir: Path,
    num_frames_target: int = 300,
    camera_type: Literal["perspective", "fisheye"] = "perspective",
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree",
    num_downscales: int = 3,
    skip_colmap: bool = False,
    gpu: bool = True,
    verbose: bool = False,
):
    """Process images or videos into a Nerfactory dataset.

    This script does the following:
    1) Converts the video into images (if video is provided).
    2) Scales images to a specified size.
    3) Calculates and stores the sharpness of each image.
    4) Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.


    Args:
        data: Path the data, either a video file or a directory of images.
        output_dir: Path to the output directory.
        num_frames: Target number of frames to use for the dataset, results may not be exact.
        camera_type: Camera model to use.
        matching_method: Feature matching method to use. Vocab tree is recommended for a balance of speed and
            accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
            will downscale the images by 2x, 4x, and 8x.
        skip_colmap: If True, skips COLMAP and generates transforms.json if possible.
        gpu: If True, use GPU.
        verbose: If True, print extra logging.
    """

    print(data)

    check_ffmpeg_installed()
    check_colmap_installed()

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    summary_log = []

    if data.is_file():
        if not verbose:
            with CONSOLE.status("[bold yellow]Converting video to images...", spinner="bouncingBall"):
                num_vid_frames, num_frames = convert_video_to_images(
                    data, image_dir=image_dir, num_frames_target=num_frames_target, verbose=verbose
                )
        else:
            num_vid_frames, num_frames = convert_video_to_images(
                data, image_dir=image_dir, num_frames_target=num_frames_target, verbose=verbose
            )
        summary_log.append(f"Starting with {num_vid_frames} video frames")
        summary_log.append(f"We extracted {num_frames} images")
        CONSOLE.log("[bold green]:tada: Done converting video to images.")
    else:
        if num_frames_target is not None:
            CONSOLE.log("[bold yellow]Warning: num_frames_target is ignored when data is a directory of images.")
        if not verbose:
            with CONSOLE.status("[bold yellow]Copying images...", spinner="bouncingBall"):
                num_frames = copy_images(data, image_dir=image_dir, verbose=verbose)
        else:
            num_frames = copy_images(data, image_dir=image_dir, verbose=verbose)
        CONSOLE.log("[bold green]:tada: Done copying images.")
        summary_log.append(f"Starting with {num_frames} images")

    if num_downscales > 0:
        if not verbose:
            with CONSOLE.status("[bold yellow]Downscaling images...", spinner="growVertical"):
                downscale_images(image_dir, num_downscales, verbose=verbose)
        else:
            downscale_images(image_dir, num_downscales, verbose=verbose)
        CONSOLE.log("[bold green]:tada: Done downscaling images.")
        downscale_text = [f"[bold blue]{2**(i+1)}x[/bold blue]" for i in range(num_downscales)]
        downscale_text = ", ".join(downscale_text[:-1]) + " and " + downscale_text[-1]
        summary_log.append(f"We downsampled the images by {downscale_text}")

    camera_model = CAMERA_MODELS[camera_type]
    colmap_dir = output_dir / "colmap"
    if not skip_colmap:
        colmap_dir.mkdir(parents=True, exist_ok=True)

        run_colmap(
            image_dir=image_dir,
            colmap_dir=colmap_dir,
            camera_model=camera_model,
            gpu=gpu,
            verbose=verbose,
            matching_method=matching_method,
        )

    if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
        with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
            num_matched_frames = colmap_to_json(
                cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                images_path=colmap_dir / "sparse" / "0" / "images.bin",
                output_dir=output_dir,
                camera_model=camera_model,
            )
            summary_log.append(f"Colmap matched {num_matched_frames} images")
    else:
        CONSOLE.log("[bold yellow]Warning: could not find existing COLMAP results. Not generating transforms.json")

    CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

    for summary in summary_log:
        CONSOLE.print(summary, justify="center")
    CONSOLE.rule()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(main)


if __name__ == "__main__":
    entrypoint()
