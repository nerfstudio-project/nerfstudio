#!/usr/bin/env python
"""Add additional query images to a pre-existing nerfstudio compatible dataset."""


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
class AddQueryImages:
    """Add additional images into a nerfstudio dataset.

    This script does the following:

    1. Scales additional images to a specified size.
    2. Registers the additional query images into a pre-existing `COLMAP <https://colmap.github.io/>`_ database.
    """

    dataset: Path
    """Path to the pre-existing Nerfstudio dataset data directory, containing colmap database.db file"""
    query_dir: Path
    """Path to the directory containing additional query images to be registered."""
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
    """If True, skips COLMAP and updates transforms.json if possible."""
    skip_image_processing: bool = False
    """If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled"""
    colmap_model_path: Path = DEFAULT_COLMAP_PATH
    """Optionally sets the path of the colmap model. Used only when --skip-colmap is set to True.
       The path is relative to the dataset directory.
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
    bundle_adjust: bool = True
    """If True, runs bundle adjustment on query images (recommended)"""

    def main(self) -> None:  # pylint: disable=R0915
        """Process images into a nerfstudio dataset."""
        # pylint: disable=too-many-statements
        require_cameras_exist = False
        colmap_model_path = self.dataset / Path(self.colmap_model_path)
        if self.colmap_model_path != DEFAULT_COLMAP_PATH:
            if not self.skip_colmap:
                CONSOLE.log("[bold red]The --colmap-model-path can only be used when --skip-colmap is not set.")
                sys.exit(1)
            elif not (self.dataset / self.colmap_model_path).exists():
                CONSOLE.log(
                    f"[bold red]The colmap-model-path {self.dataset / self.colmap_model_path} does not exist."
                )
                sys.exit(1)
            require_cameras_exist = True

        install_checks.check_ffmpeg_installed()
        install_checks.check_colmap_installed()

        image_rename_map: Optional[Dict[str, str]] = None
        #self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.query_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Generate planar projections if equirectangular
        if self.camera_type == "equirectangular":
            pers_size = equirect_utils.compute_resolution_from_equirect(self.query_dir, self.images_per_equirect)
            CONSOLE.log(f"Generating {self.images_per_equirect} {pers_size} sized images per equirectangular image")
            self.query_dir = equirect_utils.generate_planar_projections_from_equirectangular(
                self.query_dir, pers_size, self.images_per_equirect, crop_factor=self.crop_factor
            )

        summary_log = []

        # Copy and downscale query images
        if not self.skip_image_processing:
            # Copy images to output directory
            image_rename_map_paths = process_data_utils.copy_images(
                self.query_dir, image_dir=image_dir, crop_factor=self.crop_factor, verbose=self.verbose
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
        
        # Register query images using COLMAP
        colmap_dir = self.query_dir / "colmap" # consider changing to self.dataset
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)
            colmap_model_path = colmap_dir / "sparse" / "0"
            require_cameras_exist = True

            self.register_images(
                query_dir=self.query_dir,
                colmap_dir=self.colmap_model_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                colmap_cmd=self.colmap_cmd,
                bundle_adjust=self.bundle_adjust
            )

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

    def register_images(self,
        query_dir: Path,
        colmap_dir: Path,
        camera_model: CameraModel = CAMERA_MODELS['perspective'],
        camera_mask_path: Optional[Path] = None,
        gpu: bool = False,
        verbose: bool = False,
        matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
        colmap_cmd: str = "colmap",
        bundle_adjust: bool = True,
    ) -> None:
        """Registers additional query images into a pre-existing COLMAP database.

        Args:
            query_dir: Path to the directory containing query images.
            colmap_dir: Path to existing colmap directory containing database.db file.
            camera_model: Camera model to use.
            camera_mask_path: Path to the camera mask.
            gpu: If True, use GPU.
            verbose: If True, logs the output of the command.
            matching_method: Matching method to use.
            colmap_cmd: Path to the COLMAP executable.
        """

        colmap_database_path = colmap_dir / "database.db"

        # Feature extraction
        feature_extractor_cmd = [
            f"{self.colmap_cmd} feature_extractor",
            f"--database_path {colmap_dir / 'database.db'}",
            f"--image_path {query_dir}",
            "--ImageReader.single_camera 1",
            f"--ImageReader.camera_model {self.camera_model.value}",
            f"--SiftExtraction.use_gpu {int(gpu)}",
            "--ImageReader.existing_camera_id 1"
        ]
        if camera_mask_path is not None:
            feature_extractor_cmd.append(f"--ImageReader.camera_mask_path {camera_mask_path}")
        feature_extractor_cmd = " ".join(feature_extractor_cmd)
        with status(msg="[bold yellow]Running COLMAP feature extractor...", spinner="moon", verbose=verbose):
            run_command(feature_extractor_cmd, verbose=verbose)
        CONSOLE.log("[bold green]:tada: Done extracting COLMAP features on query image(s).")

        # Feature matching
        feature_matcher_cmd = [
            f"{colmap_cmd} {matching_method}_matcher",
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

        # Image registration
        sparse_dir = colmap_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        image_registration_cmd = [
            f"{colmap_cmd} image_registrator",
            f"--database_path {colmap_dir / 'database.db'}",
            f"--input_path {sparse_dir}/0",
            f"--output_path {query_dir}",
        ]
        image_registration_cmd = " ".join(image_registration_cmd)
        with status(msg="[bold yellow]Running COLMAP image registration...", spinner="runner", verbose=verbose):
            run_command(image_registration_cmd, verbose=verbose)
        CONSOLE.log("[bold green]:tada: Done registering query images")

        # Bundle adjust (Optional)
        if bundle_adjust:
            bundle_adjuster_cmd = [
                f"{colmap_cmd} bundle_adjuster",
                f"--input_path {query_dir}",
                f"--output_path {query_dir}",
            ]
            bundle_adjuster_cmd = " ".join(bundle_adjuster_cmd)
            with status(msg="[bold yellow]Running COLMAP bundle adjustment on query images...", spinner="runner", verbose=verbose):
                run_command(bundle_adjuster_cmd, verbose=verbose)
            CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment on query images.")

def test():
    dataset = Path('./data/nerfstudio/vegetation')
    query_dir = Path('./data/nerfstudio/vegetation/queries')

    AddQueryImages(dataset=dataset, query_dir=query_dir, verbose=True).main()

if __name__ =='__main__':
    test()