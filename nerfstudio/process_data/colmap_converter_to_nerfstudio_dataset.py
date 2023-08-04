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

"""Base class to processes a video or image sequence to a nerfstudio compatible dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from nerfstudio.process_data import colmap_utils, hloc_utils, process_data_utils
from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import install_checks
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ColmapConverterToNerfstudioDataset(BaseConverterToNerfstudioDataset):
    """Base class to process images or video into a nerfstudio dataset using colmap"""

    camera_type: Literal["perspective", "fisheye", "equirectangular"] = "perspective"
    """Camera model to use."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed
    and accuracy. Exhaustive is slower but more accurate. Sequential is faster but
    should only be used for videos."""
    sfm_tool: Literal["any", "colmap", "hloc"] = "any"
    """Structure from motion tool to use. Colmap will use sift features, hloc can use
    many modern methods such as superpoint features and superglue matcher"""
    refine_pixsfm: bool = False
    """If True, runs refinement using Pixel Perfect SFM.
    Only works with hloc sfm_tool"""
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
        "any",
        "NN",
        "superglue",
        "superglue-fast",
        "NN-superpoint",
        "NN-ratio",
        "NN-mutual",
        "adalam",
    ] = "any"
    """Matching algorithm."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    skip_image_processing: bool = False
    """If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled"""
    colmap_model_path: Path = Path("colmap/sparse/0")
    """Optionally sets the path of the colmap model. Used only when --skip-colmap is set to True. The path is relative
       to the output directory.
    """
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
    images_per_equirect: Literal[8, 14] = 8
    """Number of samples per image to take from each equirectangular image.
       Used only when camera-type is equirectangular.
    """
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""
    crop_bottom: float = 0.0
    """Portion of the image to crop from the bottom.
       Can be used instead of `crop-factor 0.0 [num] 0.0 0.0` Should be in [0,1].
    """
    gpu: bool = True
    """If True, use GPU."""
    use_sfm_depth: bool = False
    """If True, export and use depth maps induced from SfM points."""
    include_depth_debug: bool = False
    """If --use-sfm-depth and this flag is True, also export debug images showing Sf overlaid upon input images."""
    same_dimensions: bool = True
    """Whether to assume all images are same dimensions and so to use fast downscaling with no autorotation."""

    @staticmethod
    def default_colmap_path() -> Path:
        return Path("colmap/sparse/0")

    @property
    def absolute_colmap_model_path(self) -> Path:
        return self.output_dir / self.colmap_model_path

    @property
    def absolute_colmap_path(self) -> Path:
        return self.output_dir / "colmap"

    def _save_transforms(
        self,
        num_frames: int,
        image_id_to_depth_path: Optional[Dict[int, Path]] = None,
        camera_mask_path: Optional[Path] = None,
        image_rename_map: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Save colmap transforms into the output folder

        Args:
            image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
            image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        """
        summary_log = []
        if (self.absolute_colmap_model_path / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    recon_dir=self.absolute_colmap_model_path,
                    output_dir=self.output_dir,
                    image_id_to_depth_path=image_id_to_depth_path,
                    camera_mask_path=camera_mask_path,
                    image_rename_map=image_rename_map,
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_frames, num_matched_frames))

        else:
            CONSOLE.log(
                "[bold yellow]Warning: Could not find existing COLMAP results. " "Not generating transforms.json"
            )
        return summary_log

    def _export_depth(self) -> Tuple[Optional[Dict[int, Path]], List[str]]:
        """If SFM is used for creating depth image, this method will create the depth images from image in
        `self.image_dir`.

        Returns:
            Depth file paths indexed by COLMAP image id, logs
        """
        summary_log = []
        if self.use_sfm_depth:
            depth_dir = self.output_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            image_id_to_depth_path = colmap_utils.create_sfm_depth(
                recon_dir=self.output_dir / self.default_colmap_path(),
                output_dir=depth_dir,
                include_depth_debug=self.include_depth_debug,
                input_images_dir=self.image_dir,
                verbose=self.verbose,
            )
            summary_log.append(
                process_data_utils.downscale_images(
                    depth_dir,
                    self.num_downscales,
                    folder_name="depths",
                    nearest_neighbor=True,
                    verbose=self.verbose,
                )
            )
            return image_id_to_depth_path, summary_log
        return None, summary_log

    def _run_colmap(self, mask_path: Optional[Path] = None):
        """
        Args:
            mask_path: Path to the camera mask. Defaults to None.
        """
        self.absolute_colmap_path.mkdir(parents=True, exist_ok=True)

        (
            sfm_tool,
            feature_type,
            matcher_type,
        ) = process_data_utils.find_tool_feature_matcher_combination(
            self.sfm_tool, self.feature_type, self.matcher_type
        )
        # check that sfm_tool is hloc if using refine_pixsfm
        if self.refine_pixsfm:
            assert sfm_tool == "hloc", "refine_pixsfm only works with sfm_tool hloc"

        # set the image_dir if didn't copy
        if self.skip_image_processing:
            image_dir = self.data
        else:
            image_dir = self.image_dir

        if sfm_tool == "colmap":
            colmap_utils.run_colmap(
                image_dir=image_dir,
                colmap_dir=self.absolute_colmap_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                camera_mask_path=mask_path,
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                colmap_cmd=self.colmap_cmd,
            )
        elif sfm_tool == "hloc":
            if mask_path is not None:
                raise RuntimeError("Cannot use a mask with hloc. Please remove the cropping options " "and try again.")

            assert feature_type is not None
            assert matcher_type is not None
            assert matcher_type != "NN"  # Only used for colmap.
            hloc_utils.run_hloc(
                image_dir=image_dir,
                colmap_dir=self.absolute_colmap_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                verbose=self.verbose,
                matching_method=self.matching_method,
                feature_type=feature_type,
                matcher_type=matcher_type,
                refine_pixsfm=self.refine_pixsfm,
            )
        else:
            raise RuntimeError("Invalid combination of sfm_tool, feature_type, and matcher_type, " "exiting")

    def __post_init__(self) -> None:
        super().__post_init__()
        install_checks.check_ffmpeg_installed()
        install_checks.check_colmap_installed()

        if self.crop_bottom < 0.0 or self.crop_bottom > 1:
            raise RuntimeError("crop_bottom must be set between 0 and 1.")

        if self.crop_bottom > 0.0:
            self.crop_factor = (0.0, self.crop_bottom, 0.0, 0.0)
