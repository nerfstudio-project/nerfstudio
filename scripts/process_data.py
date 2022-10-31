#!/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""

import json
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import appdirs
import numpy as np
import requests
import tyro
from rich.console import Console
from rich.progress import track
from typing_extensions import Annotated, Literal

from nerfstudio.utils import colmap_utils, install_checks

CONSOLE = Console(width=120)


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
}


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)


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
    vocab_tree_filename = Path(appdirs.user_data_dir("nerfstudio")) / "vocab_tree.fbow"

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


def run_command(cmd: str, verbose=False) -> Optional[str]:
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
            CONSOLE.print("[bold red]Can't satify requested number of frames. Extracting all frames.")

        ffmpeg_cmd += f" {out_filename}"

        run_command(ffmpeg_cmd, verbose=verbose)

    num_final_frames = len(list(image_dir.glob("*.png")))
    summary_log = []
    summary_log.append(f"Starting with {num_frames} video frames")
    summary_log.append(f"We extracted {num_final_frames} images")
    CONSOLE.log("[bold green]:tada: Done converting video to images.")

    return summary_log, num_final_frames


def convert_insta360_to_images(
    video_front: Path,
    video_back: Path,
    image_dir: Path,
    num_frames_target: int,
    crop_percentage: float = 0.7,
    verbose: bool = False,
) -> Tuple[List[str], int]:
    """Converts a video into a sequence of images.

    Args:
        video_front: Path to the front video.
        video_back: Path to the back video.
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

        num_frames_front = get_num_frames_in_video(video_front)
        num_frames_back = get_num_frames_in_video(video_back)
        if num_frames_front == 0:
            CONSOLE.print(f"[bold red]Error: Video has no frames: {video_front}")
            sys.exit(1)
        if num_frames_back == 0:
            CONSOLE.print(f"[bold red]Error: Video has no frames: {video_front}")
            sys.exit(1)

        spacing = num_frames_front // (num_frames_target // 2)
        vf_cmds = []
        if spacing > 1:
            vf_cmds = [f"thumbnail={spacing}", "setpts=N/TB"]
        else:
            CONSOLE.print("[bold red]Can't satify requested number of frames. Extracting all frames.")

        vf_cmds.append(f"crop=iw*{crop_percentage}:ih*{crop_percentage}")

        front_vf_cmds = vf_cmds + ["transpose=2"]
        back_vf_cmds = vf_cmds + ["transpose=1"]

        front_ffmpeg_cmd = f"ffmpeg -i {video_front} -vf {','.join(front_vf_cmds)} -r 1 {image_dir / 'frame_%05d.png'}"
        back_ffmpeg_cmd = (
            f"ffmpeg -i {video_back} -vf {','.join(back_vf_cmds)} -r 1 {image_dir / 'back_frame_%05d.png'}"
        )

        run_command(front_ffmpeg_cmd, verbose=verbose)
        run_command(back_ffmpeg_cmd, verbose=verbose)

        num_extracted_front_frames = len(list(image_dir.glob("frame*.png")))
        for i, img in enumerate(image_dir.glob("back_frame_*.png")):
            img.rename(image_dir / f"frame_{i+1+num_extracted_front_frames:05d}.png")

    num_final_frames = len(list(image_dir.glob("*.png")))
    summary_log = []
    summary_log.append(f"Starting with {num_frames_front + num_frames_back} video frames")
    summary_log.append(f"We extracted {num_final_frames} images")
    CONSOLE.log("[bold green]:tada: Done converting insta360 to images.")

    return summary_log, num_final_frames


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

        # Remove original directory only if we provide a proper image folder path
        if image_dir.is_dir() and len(image_paths):
            shutil.rmtree(image_dir, ignore_errors=True)
            image_dir.mkdir(exist_ok=True, parents=True)

        # Images should be 1-indexed for the rest of the pipeline.
        for idx, image_path in enumerate(image_paths):
            if verbose:
                CONSOLE.log(f"Copying image {idx + 1} of {len(image_paths)}...")
            shutil.copy(image_path, image_dir / f"frame_{idx + 1:05d}{image_path.suffix}")

        num_frames = len(image_paths)

    if num_frames == 0:
        CONSOLE.log("[bold red]:skull: No usable images in the data folder.")
    else:
        CONSOLE.log("[bold green]:tada: Done copying images.")

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
            file_type = image_dir.glob("frame_*").__next__().suffix
            filename = f"frame_%05d{file_type}"
            ffmpeg_cmd = [
                f"ffmpeg -i {image_dir / filename} ",
                f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor} ",
                f"{downscale_dir / filename}",
            ]
            ffmpeg_cmd = " ".join(ffmpeg_cmd)
            run_command(ffmpeg_cmd, verbose=verbose)

    CONSOLE.log("[bold green]:tada: Done downscaling images.")
    downscale_text = [f"[bold blue]{2**(i+1)}x[/bold blue]" for i in range(num_downscales)]
    downscale_text = ", ".join(downscale_text[:-1]) + " and " + downscale_text[-1]
    return f"We downsampled the images by {downscale_text}"


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

    colmap_database_path = colmap_dir / "database.db"
    if colmap_database_path.exists():
        # Can't use missing_ok argument because of Python 3.7 compatibility.
        colmap_database_path.unlink()

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
    with status(msg="[bold yellow]Running COLMAP feature extractor...", spinner="moon", verbose=verbose):
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
    with status(msg="[bold yellow]Running COLMAP feature matcher...", spinner="runner", verbose=verbose):
        run_command(feature_matcher_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done matching COLMAP features.")

    # Bundle adjustment
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        "colmap mapper",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]
    if colmap_version >= 3.7:
        mapper_cmd.append("--Mapper.ba_global_function_tolerance 1e-6")

    mapper_cmd = " ".join(mapper_cmd)

    with status(
        msg="[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(mapper_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment.")
    with status(msg="[bold yellow]Refine intrinsics...", spinner="dqpb", verbose=verbose):
        bundle_adjuster_cmd = [
            "colmap bundle_adjuster",
            f"--input_path {sparse_dir}/0",
            f"--output_path {sparse_dir}/0",
            "--BundleAdjustment.refine_principal_point 1",
        ]
        run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done refining intrinsics.")


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
            "file_path": name.as_posix(),
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


def get_insta360_filenames(data: Path) -> Tuple[Path, Path]:
    """Returns the filenames of the Insta360 videos from a single video file.

    Example input name: VID_20220212_070353_00_003.insv

    Args:
        data: Path to a Insta360 file.

    Returns:
        The filenames of the Insta360 videios.
    """
    if data.suffix != ".insv":
        raise ValueError("The input file must be an .insv file.")
    file_parts = data.stem.split("_")

    stem_back = f"VID_{file_parts[1]}_{file_parts[2]}_00_{file_parts[4]}.insv"
    stem_front = f"VID_{file_parts[1]}_{file_parts[2]}_10_{file_parts[4]}.insv"

    filename_back = data.parent / stem_back
    filename_front = data.parent / stem_front

    if not filename_back.exists():
        raise FileNotFoundError(f"Could not find {filename_back}")
    if not filename_front.exists():
        raise FileNotFoundError(f"Could not find {filename_front}")

    return filename_back, filename_front


def get_matching_summary(num_intial_frames: int, num_matched_frames: int) -> str:
    """Returns a summary of the matching results.

    Args:
        num_intial_frames: The number of initial frames.
        num_matched_frames: The number of matched frames.

    Returns:
        A summary of the matching results.
    """
    match_ratio = num_matched_frames / num_intial_frames
    if match_ratio == 1:
        return "[bold green]COLAMP found poses for all images, CONGRATS!"
    if match_ratio < 0.4:
        result = f"[bold red]COLMAP only found poses for {num_matched_frames / num_intial_frames * 100:.2f}%"
        result += " of the images. This is low.\nThis can be caused by a variety of reasons,"
        result += " such poor scene coverage, blurry images, or large exposure changes."
        return result
    if match_ratio < 0.8:
        result = f"[bold yellow]COLMAP only found poses for {num_matched_frames / num_intial_frames * 100:.2f}%"
        result += " of the images.\nThis isn't great, but may be ok."
        result += "\nMissing poses can be caused by a variety of reasons, such poor scene coverage, blurry images,"
        result += " or large exposure changes."
        return result
    return f"[bold green]COLMAP found poses for {num_matched_frames / num_intial_frames * 100:.2f}% of the images."


@dataclass
class ProcessImages:
    """Process images into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    data: Path
    """Path the data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    camera_type: Literal["perspective", "fisheye"] = "perspective"
    """Camera model to use."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed and
        accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    gpu: bool = True
    """If True, use GPU."""
    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""
        install_checks.check_ffmpeg_installed()
        install_checks.check_colmap_installed()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        # Copy images to output directory
        num_frames = copy_images(self.data, image_dir=image_dir, verbose=self.verbose)
        summary_log.append(f"Starting with {num_frames} images")

        # Downscale images
        summary_log.append(downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run COLMAP
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            run_colmap(
                image_dir=image_dir,
                colmap_dir=colmap_dir,
                camera_model=CAMERA_MODELS[self.camera_type],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
            )

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(get_matching_summary(num_frames, num_matched_frames))
        else:
            CONSOLE.log("[bold yellow]Warning: could not find existing COLMAP results. Not generating transforms.json")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessVideo:
    """Process videos into a nerfstudio dataset.

    This script does the following:

    1. Converts the video into images.
    2. Scales images to a specified size.
    3. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    data: Path
    """Path the data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    num_frames_target: int = 300
    """Target number of frames to use for the dataset, results may not be exact."""
    camera_type: Literal["perspective", "fisheye"] = "perspective"
    """Camera model to use."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed and
        accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    gpu: bool = True
    """If True, use GPU."""
    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:
        """Process video into a nerfstudio dataset."""
        install_checks.check_ffmpeg_installed()
        install_checks.check_colmap_installed()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Convert video to images
        summary_log, num_extracted_frames = convert_video_to_images(
            self.data, image_dir=image_dir, num_frames_target=self.num_frames_target, verbose=self.verbose
        )

        # Downscale images
        summary_log.append(downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run Colmap
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            run_colmap(
                image_dir=image_dir,
                colmap_dir=colmap_dir,
                camera_model=CAMERA_MODELS[self.camera_type],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
            )

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(get_matching_summary(num_extracted_frames, num_matched_frames))
        else:
            CONSOLE.log("[bold yellow]Warning: could not find existing COLMAP results. Not generating transforms.json")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessInsta360:
    """Process Insta360 videos into a nerfstudio dataset. Currently this uses a center crop of the raw data
    so data at the extreme edges of the video will be lost.

    Expects data from a 2 camera Insta360, single or >2 camera models will not work.
    (tested with Insta360 One X2)

    This script does the following:

    1. Converts the videos into images.
    2. Scales images to a specified size.
    3. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    data: Path
    """Path the data, It should be one of the 3 .insv files saved with each capture (Any work)."""
    output_dir: Path
    """Path to the output directory."""
    num_frames_target: int = 400
    """Target number of frames to use for the dataset, results may not be exact."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed and
        accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    gpu: bool = True
    """If True, use GPU."""
    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:
        """Process video into a nerfstudio dataset."""
        install_checks.check_ffmpeg_installed()
        install_checks.check_colmap_installed()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        filename_back, filename_front = get_insta360_filenames(self.data)

        # Convert video to images
        summary_log, num_extracted_frames = convert_insta360_to_images(
            video_front=filename_front,
            video_back=filename_back,
            image_dir=image_dir,
            num_frames_target=self.num_frames_target,
            verbose=self.verbose,
        )

        # Downscale images
        summary_log.append(downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run Colmap
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            run_colmap(
                image_dir=image_dir,
                colmap_dir=colmap_dir,
                camera_model=CAMERA_MODELS["fisheye"],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
            )

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS["fisheye"],
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(get_matching_summary(num_extracted_frames, num_matched_frames))
        else:
            CONSOLE.log("[bold yellow]Warning: could not find existing COLMAP results. Not generating transforms.json")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


Commands = Union[
    Annotated[ProcessImages, tyro.conf.subcommand(name="images")],
    Annotated[ProcessVideo, tyro.conf.subcommand(name="video")],
    Annotated[ProcessInsta360, tyro.conf.subcommand(name="insta360")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # type: ignore
