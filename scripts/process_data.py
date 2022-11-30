#!/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""


import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import tyro
from rich.console import Console
from typing_extensions import Annotated, Literal

from nerfstudio.process_data import (
    colmap_utils,
    insta360_utils,
    metashape_utils,
    polycam_utils,
    process_data_utils,
    record3d_utils,
)
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import install_checks

CONSOLE = Console(width=120)


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
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
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
        num_frames = process_data_utils.copy_images(self.data, image_dir=image_dir, verbose=self.verbose)
        summary_log.append(f"Starting with {num_frames} images")

        # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run COLMAP
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            colmap_utils.run_colmap(
                image_dir=image_dir,
                colmap_dir=colmap_dir,
                camera_model=CAMERA_MODELS[self.camera_type],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                colmap_cmd=self.colmap_cmd,
            )

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_frames, num_matched_frames))
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
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
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
        summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
            self.data, image_dir=image_dir, num_frames_target=self.num_frames_target, verbose=self.verbose
        )

        # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run Colmap
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            colmap_utils.run_colmap(
                image_dir=image_dir,
                colmap_dir=colmap_dir,
                camera_model=CAMERA_MODELS[self.camera_type],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                colmap_cmd=self.colmap_cmd,
            )

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_extracted_frames, num_matched_frames))
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
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
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

        filename_back, filename_front = insta360_utils.get_insta360_filenames(self.data)

        if not filename_back.exists():
            raise FileNotFoundError(f"Could not find {filename_back}")

        ffprobe_cmd = f"ffprobe -v quiet -print_format json -show_streams -select_streams v:0 {filename_back}"

        ffprobe_output = process_data_utils.run_command(ffprobe_cmd)

        assert ffprobe_output is not None
        ffprobe_decoded = json.loads(ffprobe_output)

        width, height = ffprobe_decoded["streams"][0]["width"], ffprobe_decoded["streams"][0]["height"]

        summary_log, num_extracted_frames = [], 0

        if width / height == 1:
            if not filename_front.exists():
                raise FileNotFoundError(f"Could not find {filename_front}")
            # Convert video to images
            summary_log, num_extracted_frames = insta360_utils.convert_insta360_to_images(
                video_front=filename_front,
                video_back=filename_back,
                image_dir=image_dir,
                num_frames_target=self.num_frames_target,
                verbose=self.verbose,
            )
        else:
            summary_log, num_extracted_frames = insta360_utils.convert_insta360_single_file_to_images(
                video=filename_back,
                image_dir=image_dir,
                num_frames_target=self.num_frames_target,
                verbose=self.verbose,
            )

        # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run Colmap
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            colmap_utils.run_colmap(
                image_dir=image_dir,
                colmap_dir=colmap_dir,
                camera_model=CAMERA_MODELS["fisheye"],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                colmap_cmd=self.colmap_cmd,
            )

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS["fisheye"],
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_extracted_frames, num_matched_frames))
        else:
            CONSOLE.log("[bold yellow]Warning: could not find existing COLMAP results. Not generating transforms.json")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessRecord3D:
    """Process Record3D data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Record3D poses into the nerfstudio format.
    """

    data: Path
    """Path to the record3D data."""
    output_dir: Path
    """Path to the output directory."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size: int = 300
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        record3d_image_dir = self.data / "rgb"

        if not record3d_image_dir.exists():
            raise ValueError(f"Image directory {record3d_image_dir} doesn't exist")

        record3d_image_filenames = []
        for f in record3d_image_dir.iterdir():
            if f.stem.isdigit():  # removes possible duplicate images (for example, 123(3).jpg)
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    record3d_image_filenames.append(f)

        record3d_image_filenames = sorted(record3d_image_filenames, key=lambda fn: int(fn.stem))
        num_images = len(record3d_image_filenames)
        idx = np.arange(num_images)
        if self.max_dataset_size != -1 and num_images > self.max_dataset_size:
            idx = np.round(np.linspace(0, num_images - 1, self.max_dataset_size)).astype(int)

        record3d_image_filenames = list(np.array(record3d_image_filenames)[idx])
        # Copy images to output directory
        copied_image_paths = process_data_utils.copy_images_list(
            record3d_image_filenames, image_dir=image_dir, verbose=self.verbose
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        summary_log.append(f"Used {num_frames} images out of {num_images} total")
        if self.max_dataset_size > 0:
            summary_log.append(
                "To change the size of the dataset add the argument --max_dataset_size to larger than the "
                f"current value ({self.max_dataset_size}), or -1 to use all images."
            )

        # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        metadata_path = self.data / "metadata.json"
        record3d_utils.record3d_to_json(copied_image_paths, metadata_path, self.output_dir, indices=idx)
        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessPolycam:
    """Process Polycam data into a nerfstudio dataset.

    To capture data, use the Polycam app on an iPhone or iPad with LiDAR. The capture must be in LiDAR or ROOM mode.
    Developer mode must be enabled in the app settings, this will enable a raw data export option in the export menus.
    The exported data folder is used as the input to this script.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Polycam poses into the nerfstudio format.
    """

    data: Path
    """Path the polycam export data folder. Can be .zip file or folder."""
    output_dir: Path
    """Path to the output directory."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    use_uncorrected_images: bool = False
    """If True, use the raw images from the polycam export. If False, use the corrected images."""
    max_dataset_size: int = 600
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""
    min_blur_score: float = 25
    """Minimum blur score to use an image. If the blur score is below this value, the image will be skipped."""
    crop_border_pixels: int = 15
    """Number of pixels to crop from each border of the image. Useful as borders may be black due to undistortion."""

    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        if self.data.suffix == ".zip":
            with zipfile.ZipFile(self.data, "r") as zip_ref:
                zip_ref.extractall(self.output_dir)
                extracted_folder = zip_ref.namelist()[0].split("/")[0]
            self.data = self.output_dir / extracted_folder

        if (self.data / "keyframes" / "corrected_images").exists() and not self.use_uncorrected_images:
            polycam_image_dir = self.data / "keyframes" / "corrected_images"
            polycam_cameras_dir = self.data / "keyframes" / "corrected_cameras"
        else:
            polycam_image_dir = self.data / "keyframes" / "images"
            polycam_cameras_dir = self.data / "keyframes" / "cameras"
            self.crop_border_pixels = 0
            if not self.use_uncorrected_images:
                CONSOLE.print("[bold yellow]Corrected images not found, using raw images.")

        if not polycam_image_dir.exists():
            raise ValueError(f"Image directory {polycam_image_dir} doesn't exist")

        # Copy images to output directory

        polycam_image_filenames = []
        for f in polycam_image_dir.iterdir():
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                polycam_image_filenames.append(f)
        polycam_image_filenames = sorted(polycam_image_filenames, key=lambda fn: int(fn.stem))
        num_images = len(polycam_image_filenames)
        idx = np.arange(num_images)
        if self.max_dataset_size != -1 and num_images > self.max_dataset_size:
            idx = np.round(np.linspace(0, num_images - 1, self.max_dataset_size)).astype(int)

        polycam_image_filenames = list(np.array(polycam_image_filenames)[idx])
        # Copy images to output directory
        copied_image_paths = process_data_utils.copy_images_list(
            polycam_image_filenames,
            image_dir=image_dir,
            crop_border_pixels=self.crop_border_pixels,
            verbose=self.verbose,
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]

        if self.max_dataset_size > 0 and num_frames != num_images:
            summary_log.append(f"Started with {num_frames} images out of {num_images} total")
            summary_log.append(
                "To change the size of the dataset add the argument --max_dataset_size to larger than the "
                f"current value ({self.max_dataset_size}), or -1 to use all images."
            )
        else:
            summary_log.append(f"Started with {num_frames} images")

        # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Save json
        if num_frames == 0:
            CONSOLE.print("[bold red]No images found, exiting")
            sys.exit(1)
        summary_log.extend(
            polycam_utils.polycam_to_json(
                image_filenames=polycam_image_filenames,
                cameras_dir=polycam_cameras_dir,
                output_dir=self.output_dir,
                min_blur_score=self.min_blur_score,
                crop_border_pixels=self.crop_border_pixels,
            )
        )

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessMetashape:
    """Process Metashape data into a nerfstudio dataset.

    This script assumes that cameras have been aligned using Metashape. After alignment, it is necessary to export the
    camera poses as a `.xml` file. This option can be found under `File > Export > Export Cameras`.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Metashape poses into the nerfstudio format.
    """

    data: Path
    """Path to a folder of images."""
    xml: Path
    """Path to the Metashape xml file."""
    output_dir: Path
    """Path to the output directory."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size: int = 600
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""
    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        if self.xml.suffix != ".xml":
            raise ValueError(f"XML file {self.xml} must have a .xml extension")
        if not self.xml.exists:
            raise ValueError(f"XML file {self.xml} doesn't exist")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        # Copy images to output directory
        image_filenames = []
        for f in self.data.iterdir():
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                image_filenames.append(f)
        image_filenames = sorted(image_filenames, key=lambda fn: fn.stem)
        num_images = len(image_filenames)
        idx = np.arange(num_images)
        if self.max_dataset_size != -1 and num_images > self.max_dataset_size:
            idx = np.round(np.linspace(0, num_images - 1, self.max_dataset_size)).astype(int)

        image_filenames = list(np.array(image_filenames)[idx])
        # Copy images to output directory
        copied_image_paths = process_data_utils.copy_images_list(
            image_filenames,
            image_dir=image_dir,
            verbose=self.verbose,
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        original_names = [image_path.stem for image_path in image_filenames]
        image_filename_map = dict(zip(original_names, copied_image_paths))

        if self.max_dataset_size > 0 and num_frames != num_images:
            summary_log.append(f"Started with {num_frames} images out of {num_images} total")
            summary_log.append(
                "To change the size of the dataset add the argument --max_dataset_size to larger than the "
                f"current value ({self.max_dataset_size}), or -1 to use all images."
            )
        else:
            summary_log.append(f"Started with {num_frames} images")

        # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Save json
        if num_frames == 0:
            CONSOLE.print("[bold red]No images found, exiting")
            sys.exit(1)
        summary_log.extend(
            metashape_utils.metashape_to_json(
                image_filename_map=image_filename_map,
                xml_filename=self.xml,
                output_dir=self.output_dir,
                verbose=self.verbose,
            )
        )

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


Commands = Union[
    Annotated[ProcessImages, tyro.conf.subcommand(name="images")],
    Annotated[ProcessVideo, tyro.conf.subcommand(name="video")],
    Annotated[ProcessPolycam, tyro.conf.subcommand(name="polycam")],
    Annotated[ProcessMetashape, tyro.conf.subcommand(name="metashape")],
    Annotated[ProcessInsta360, tyro.conf.subcommand(name="insta360")],
    Annotated[ProcessRecord3D, tyro.conf.subcommand(name="record3d")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # type: ignore
