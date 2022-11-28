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

"""Helper utils for processing insta360 data."""

import sys
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

from nerfstudio.process_data.process_data_utils import get_num_frames_in_video
from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)


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

    return filename_back, filename_front


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
        crop_percentage: Percentage used to calculate the cropped dimentions of extracted frames. Currently used to crop
         out the curved portions of the fish-eye lens.
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


def convert_insta360_single_file_to_images(
    video: Path,
    image_dir: Path,
    num_frames_target: int,
    crop_percentage: float = 0.7,
    verbose: bool = False,
) -> Tuple[List[str], int]:
    """Converts a video into a sequence of images.

    Args:
        video: Path to the video.
        output_dir: Path to the output directory.
        num_frames_target: Number of frames to extract.
        crop_percentage: Percentage used to calculate the cropped dimentions of extracted frames. Currently used to crop
         out the curved portions of the fish-eye lens.
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

        num_frames = get_num_frames_in_video(video)
        if num_frames == 0:
            CONSOLE.print(f"[bold red]Error: Video has no frames: {video}")
            sys.exit(1)

        spacing = num_frames // (num_frames_target // 2)
        vf_cmds = []
        if spacing > 1:
            vf_cmds = [f"thumbnail={spacing}", "setpts=N/TB"]
        else:
            CONSOLE.print("[bold red]Can't satify requested number of frames. Extracting all frames.")

        vf_cmds_back = vf_cmds.copy()
        vf_cmds_front = vf_cmds.copy()

        vf_cmds_back.append(
            f"crop=ih*{crop_percentage}:ih*{crop_percentage}:ih*({crop_percentage}/4):ih*({crop_percentage}/4)"
        )
        vf_cmds_front.append(
            f"crop=ih*{crop_percentage}:ih*{crop_percentage}:iw/2+ih*{crop_percentage/4}:ih*{crop_percentage/4}"
        )

        front_ffmpeg_cmd = f"ffmpeg -i {video} -vf {','.join(vf_cmds_front)} -r 1 {image_dir / 'frame_%05d.png'}"
        back_ffmpeg_cmd = f"ffmpeg -i {video} -vf {','.join(vf_cmds_back)} -r 1 {image_dir / 'back_frame_%05d.png'}"

        run_command(back_ffmpeg_cmd, verbose=verbose)
        run_command(front_ffmpeg_cmd, verbose=verbose)

        num_extracted_frames = len(list(image_dir.glob("frame*.png")))
        for i, img in enumerate(image_dir.glob("back_frame_*.png")):
            img.rename(image_dir / f"frame_{i+1+num_extracted_frames:05d}.png")

    num_final_frames = len(list(image_dir.glob("*.png")))
    summary_log = []
    summary_log.append(f"Starting with {num_frames} video frames")
    summary_log.append(f"We extracted {num_final_frames} images")
    CONSOLE.log("[bold green]:tada: Done converting insta360 to images.")

    return summary_log, num_final_frames
