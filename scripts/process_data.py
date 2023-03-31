#!/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""


import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tyro
from rich.console import Console
from typing_extensions import Annotated, Literal

from nerfstudio.process_data import (
    colmap_utils,
    equirect_utils,
    hloc_utils,
    metashape_utils,
    polycam_utils,
    process_data_utils,
    realitycapture_utils,
    record3d_utils,
)
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import install_checks

CONSOLE = Console(width=120)
DEFAULT_COLMAP_PATH = Path("colmap/sparse/0")


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
    camera_type: Literal["perspective", "fisheye", "equirectangular"] = "perspective"
    """Camera model to use."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed and
        accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos."""
    sfm_tool: Literal["any", "colmap", "hloc"] = "any"
    """Structure from motion tool to use. Colmap will use sift features, hloc can use many modern methods
       such as superpoint features and superglue matcher"""
    refine_pixsfm: bool = False
    """If True, runs refinement using Pixel Perfect SFM. Only works with hloc sfm_tool"""
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
    ] = "any"
    """Type of feature to use."""
    matcher_type: Literal[
        "any", "NN", "superglue", "superglue-fast", "NN-superpoint", "NN-ratio", "NN-mutual", "adalam"
    ] = "any"
    """Matching algorithm."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    skip_image_processing: bool = False
    """If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled"""
    colmap_model_path: Path = DEFAULT_COLMAP_PATH
    """Optionally sets the path of the colmap model. Used only when --skip-colmap is set to True.
       The path is relative to the output directory.
    """
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
    images_per_equirect: Literal[8, 14] = 8
    """Number of samples per image to take from each equirectangular image.
       Used only when camera-type is equirectangular.
    """
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""
    gpu: bool = True
    """If True, use GPU."""
    use_sfm_depth: bool = False
    """If True, export and use depth maps induced from SfM points."""
    include_depth_debug: bool = False
    """If --use-sfm-depth and this flag is True, also export debug images showing SfM overlaid upon input images."""
    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:  # pylint: disable=R0915
        """Process images into a nerfstudio dataset."""
        # pylint: disable=too-many-statements
        require_cameras_exist = False
        colmap_model_path = self.output_dir / Path(self.colmap_model_path)
        if self.colmap_model_path != DEFAULT_COLMAP_PATH:
            if not self.skip_colmap:
                CONSOLE.log("[bold red]The --colmap-model-path can only be used when --skip-colmap is not set.")
                sys.exit(1)
            elif not (self.output_dir / self.colmap_model_path).exists():
                CONSOLE.log(
                    f"[bold red]The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist."
                )
                sys.exit(1)
            require_cameras_exist = True

        install_checks.check_ffmpeg_installed()
        install_checks.check_colmap_installed()

        image_rename_map: Optional[Dict[str, str]] = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Generate planar projections if equirectangular
        if self.camera_type == "equirectangular":
            pers_size = equirect_utils.compute_resolution_from_equirect(self.data, self.images_per_equirect)
            CONSOLE.log(f"Generating {self.images_per_equirect} {pers_size} sized images per equirectangular image")
            self.data = equirect_utils.generate_planar_projections_from_equirectangular(
                self.data, pers_size, self.images_per_equirect, crop_factor=self.crop_factor
            )

        summary_log = []

        # Copy and downscale images
        if not self.skip_image_processing:
            # Copy images to output directory
            image_rename_map_paths = process_data_utils.copy_images(
                self.data, image_dir=image_dir, crop_factor=self.crop_factor, verbose=self.verbose
            )
            image_rename_map = dict((a.name, b.name) for a, b in image_rename_map_paths.items())
            num_frames = len(image_rename_map)
            summary_log.append(f"Starting with {num_frames} images")

            # Downscale images
            summary_log.append(
                process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose)
            )
        else:
            num_frames = len(process_data_utils.list_images(self.data))
            if num_frames == 0:
                CONSOLE.log("[bold red]:skull: No usable images in the data folder.")
                sys.exit(1)
            summary_log.append(f"Starting with {num_frames} images")

        # Run COLMAP
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)
            colmap_model_path = colmap_dir / "sparse" / "0"
            require_cameras_exist = True

            self._run_colmap(image_dir, colmap_dir)

            # Colmap uses renamed images
            image_rename_map = None

        # Export depth maps
        if self.use_sfm_depth:
            depth_dir = self.output_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            image_id_to_depth_path = colmap_utils.create_sfm_depth(
                recon_dir=colmap_dir / "sparse" / "0",
                output_dir=depth_dir,
                include_depth_debug=self.include_depth_debug,
                input_images_dir=image_dir,
                verbose=self.verbose,
            )
            summary_log.append(
                process_data_utils.downscale_images(
                    depth_dir, self.num_downscales, folder_name="depths", nearest_neighbor=True, verbose=self.verbose
                )
            )
        else:
            image_id_to_depth_path = None

        # Save transforms.json
        if (colmap_model_path / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    recon_dir=colmap_model_path,
                    output_dir=self.output_dir,
                    image_id_to_depth_path=image_id_to_depth_path,
                    image_rename_map=image_rename_map,
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_frames, num_matched_frames))
        elif require_cameras_exist:
            CONSOLE.log(f"[bold red]Could not find existing COLMAP results ({colmap_model_path / 'cameras.bin'}).")
            sys.exit(1)
        else:
            CONSOLE.log("[bold yellow]Warning: could not find existing COLMAP results. Not generating transforms.json")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()

    def _run_colmap(self, image_dir, colmap_dir):
        (sfm_tool, feature_type, matcher_type) = process_data_utils.find_tool_feature_matcher_combination(
            self.sfm_tool, self.feature_type, self.matcher_type
        )
        # check that sfm_tool is hloc if using refine_pixsfm
        if self.refine_pixsfm:
            assert sfm_tool == "hloc", "refine_pixsfm only works with sfm_tool hloc"

        if sfm_tool == "colmap":
            colmap_utils.run_colmap(
                image_dir=image_dir,
                colmap_dir=colmap_dir,
                camera_model=CAMERA_MODELS[self.camera_type],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                colmap_cmd=self.colmap_cmd,
            )
        elif sfm_tool == "hloc":
            hloc_utils.run_hloc(
                image_dir=image_dir,
                colmap_dir=colmap_dir,
                camera_model=CAMERA_MODELS[self.camera_type],
                verbose=self.verbose,
                matching_method=self.matching_method,
                feature_type=feature_type,
                matcher_type=matcher_type,
                refine_pixsfm=self.refine_pixsfm,
            )
        else:
            CONSOLE.log("[bold red]Invalid combination of sfm_tool, feature_type, and matcher_type, exiting")
            sys.exit(1)


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
    camera_type: Literal["perspective", "fisheye", "equirectangular"] = "perspective"
    """Camera model to use."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed and
        accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos."""
    sfm_tool: Literal["any", "colmap", "hloc"] = "any"
    """Structure from motion tool to use. Colmap will use sift features, hloc can use many modern methods
       such as superpoint features and superglue matcher"""
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
    ] = "any"
    """Type of feature to use."""
    matcher_type: Literal[
        "any", "NN", "superglue", "superglue-fast", "NN-superpoint", "NN-ratio", "NN-mutual", "adalam"
    ] = "any"
    """Matching algorithm."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
    images_per_equirect: Literal[8, 14] = 8
    """Number of samples per image to take from each equirectangular image.
       Used only when camera-type is equirectangular.
    """
    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""
    use_sfm_depth: bool = False
    """If True, export and use depth maps induced from SfM points."""
    include_depth_debug: bool = False
    """If --use-sfm-depth and this flag is True, also export debug images showing SfM overlaid upon input images."""
    gpu: bool = True
    """If True, use GPU."""
    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:  # pylint: disable=R0915
        """Process video into a nerfstudio dataset."""
        install_checks.check_ffmpeg_installed()
        install_checks.check_colmap_installed()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []
        # Convert video to images
        if self.camera_type == "equirectangular":
            # create temp images folder to store the equirect and perspective images
            temp_image_dir = self.output_dir / "temp_images"
            temp_image_dir.mkdir(parents=True, exist_ok=True)
            summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
                self.data,
                image_dir=temp_image_dir,
                num_frames_target=self.num_frames_target,
                crop_factor=(0.0, 0.0, 0.0, 0.0),
                verbose=self.verbose,
            )
        else:
            summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
                self.data,
                image_dir=image_dir,
                num_frames_target=self.num_frames_target,
                crop_factor=self.crop_factor,
                verbose=self.verbose,
            )

        # Generate planar projections if equirectangular
        if self.camera_type == "equirectangular":
            perspective_image_size = equirect_utils.compute_resolution_from_equirect(
                self.output_dir / "temp_images", self.images_per_equirect
            )
            image_dir = equirect_utils.generate_planar_projections_from_equirectangular(
                self.output_dir / "temp_images",
                perspective_image_size,
                self.images_per_equirect,
                crop_factor=self.crop_factor,
            )

            # copy the perspective images to the image directory
            process_data_utils.copy_images(
                self.output_dir / "temp_images" / "planar_projections",
                image_dir=self.output_dir / "images",
                verbose=False,
            )
            image_dir = self.output_dir / "images"

            # remove the temp_images folder
            shutil.rmtree(self.output_dir / "temp_images", ignore_errors=True)

        # # Create mask
        mask_path = process_data_utils.save_mask(
            image_dir=image_dir,
            num_downscales=self.num_downscales,
            crop_factor=(0.0, 0.0, 0.0, 0.0),
            percent_radius=self.percent_radius_crop,
        )
        if mask_path is not None:
            summary_log.append(f"Saved mask to {mask_path}")

        # # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run Colmap
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            (sfm_tool, feature_type, matcher_type) = process_data_utils.find_tool_feature_matcher_combination(
                self.sfm_tool, self.feature_type, self.matcher_type
            )

            if sfm_tool == "colmap":
                colmap_utils.run_colmap(
                    image_dir=image_dir,
                    colmap_dir=colmap_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                    camera_mask_path=mask_path,
                    gpu=self.gpu,
                    verbose=self.verbose,
                    matching_method=self.matching_method,
                    colmap_cmd=self.colmap_cmd,
                )
            elif sfm_tool == "hloc":
                if mask_path is not None:
                    CONSOLE.log(
                        "[bold red]Cannot use a mask with hloc. Please remove the cropping options and try again."
                    )
                    sys.exit(1)
                hloc_utils.run_hloc(
                    image_dir=image_dir,
                    colmap_dir=colmap_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                    verbose=self.verbose,
                    matching_method=self.matching_method,
                    feature_type=feature_type,
                    matcher_type=matcher_type,
                )
            else:
                CONSOLE.log("[bold red]Invalid combination of sfm_tool, feature_type, and matcher_type, exiting")
                sys.exit(1)

        # Export depth maps
        if self.use_sfm_depth:
            depth_dir = self.output_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            image_id_to_depth_path = colmap_utils.create_sfm_depth(
                recon_dir=colmap_dir / "sparse" / "0",
                output_dir=depth_dir,
                include_depth_debug=self.include_depth_debug,
                input_images_dir=image_dir,
                verbose=self.verbose,
            )
            summary_log.append(
                process_data_utils.downscale_images(
                    depth_dir, self.num_downscales, folder_name="depths", nearest_neighbor=True, verbose=self.verbose
                )
            )
        else:
            image_id_to_depth_path = None

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    recon_dir=colmap_dir / "sparse" / "0",
                    output_dir=self.output_dir,
                    image_id_to_depth_path=image_id_to_depth_path,
                    camera_mask_path=mask_path,
                    image_rename_map=None,
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
                "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
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
    use_depth: bool = False
    """If True, processes the generated depth maps from Polycam"""
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
            if not self.use_uncorrected_images:
                CONSOLE.print("[bold yellow]Corrected images not found, using raw images.")

        if not polycam_image_dir.exists():
            raise ValueError(f"Image directory {polycam_image_dir} doesn't exist")

        if not (self.data / "keyframes" / "depth").exists():
            depth_dir = self.data / "keyframes" / "depth"
            raise ValueError(f"Depth map directory {depth_dir} doesn't exist")

        (image_processing_log, polycam_image_filenames) = polycam_utils.process_images(
            polycam_image_dir,
            image_dir,
            crop_border_pixels=self.crop_border_pixels,
            max_dataset_size=self.max_dataset_size,
            num_downscales=self.num_downscales,
            verbose=self.verbose,
        )

        summary_log.extend(image_processing_log)

        polycam_depth_filenames = []
        if self.use_depth:
            polycam_depth_image_dir = self.data / "keyframes" / "depth"
            depth_dir = self.output_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            (depth_processing_log, polycam_depth_filenames) = polycam_utils.process_depth_maps(
                polycam_depth_image_dir,
                depth_dir,
                num_processed_images=len(polycam_image_filenames),
                crop_border_pixels=self.crop_border_pixels,
                max_dataset_size=self.max_dataset_size,
                num_downscales=self.num_downscales,
                verbose=self.verbose,
            )
            summary_log.extend(depth_processing_log)

        summary_log.extend(
            polycam_utils.polycam_to_json(
                image_filenames=polycam_image_filenames,
                depth_filenames=polycam_depth_filenames,
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
        image_filenames, num_orig_images = process_data_utils.get_image_filenames(self.data, self.max_dataset_size)
        copied_image_paths = process_data_utils.copy_images_list(
            image_filenames,
            image_dir=image_dir,
            verbose=self.verbose,
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        original_names = [image_path.stem for image_path in image_filenames]
        image_filename_map = dict(zip(original_names, copied_image_paths))

        if self.max_dataset_size > 0 and num_frames != num_orig_images:
            summary_log.append(f"Started with {num_frames} images out of {num_orig_images} total")
            summary_log.append(
                "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
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


@dataclass
class ProcessRealityCapture:
    """Process RealityCapture data into a nerfstudio dataset.

    This script assumes that cameras have been aligned using RealityCapture. After alignment, it is necessary to
    export the camera poses as a `.csv` file using the `Internal/External camera parameters` option.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts RealityCapture poses into the nerfstudio format.
    """

    data: Path
    """Path to a folder of images."""
    csv: Path
    """Path to the RealityCapture cameras CSV file."""
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

        if self.csv.suffix != ".csv":
            raise ValueError(f"CSV file {self.csv} must have a .csv extension")
        if not self.csv.exists:
            raise ValueError(f"CSV file {self.csv} doesn't exist")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        # Copy images to output directory
        image_filenames, num_orig_images = process_data_utils.get_image_filenames(self.data, self.max_dataset_size)
        copied_image_paths = process_data_utils.copy_images_list(
            image_filenames,
            image_dir=image_dir,
            verbose=self.verbose,
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        original_names = [image_path.stem for image_path in image_filenames]
        image_filename_map = dict(zip(original_names, copied_image_paths))

        if self.max_dataset_size > 0 and num_frames != num_orig_images:
            summary_log.append(f"Started with {num_frames} images out of {num_orig_images} total")
            summary_log.append(
                "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
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
            realitycapture_utils.realitycapture_to_json(
                image_filename_map=image_filename_map,
                csv_filename=self.csv,
                output_dir=self.output_dir,
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
    Annotated[ProcessRealityCapture, tyro.conf.subcommand(name="realitycapture")],
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
