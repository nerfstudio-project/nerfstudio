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

"""Processes an panorama sequence to a nerfstudio compatible dataset."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple

import numpy as np

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import (
    BaseConverterToNerfstudioDataset,
)
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import io

@dataclass
class ProcessAlignedPano(BaseConverterToNerfstudioDataset):
    """Process panorama data into a nerfstudio dataset. Skip the COLMAP

    This script does the following:

    1. Crops panoramas to planar images, and restore corresponding perspective camera poses.
    1. Scales images to a specified size.
    2. Converts perpective camera poses into the nerfstudio format.
    """
    
    metadata: Path = ""
    """Path the metadata of the panoramas sequence."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size: int = 300
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""
    images_per_equirect: Literal[8, 14] = 8
    """Number of samples per image to take from each equirectangular image.
       Used only when camera-type is equirectangular.
    """
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""
    skip_image_processing: bool = False
    """If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled"""
    is_HDR: bool = False
    """If True, process the .exr files as HDR images."""
    use_mask: bool = False
    """If True, process the .exr files with mask."""
    
    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        mask_dir = self.output_dir / "masks"
        if self.use_mask:
            mask_dir.mkdir(parents=True, exist_ok=True)
        
        summary_log = []
        pers_size = equirect_utils.compute_resolution_from_equirect(self.data, self.images_per_equirect)
        CONSOLE.log(f"Generating {self.images_per_equirect} {pers_size} sized images per equirectangular image")
        self.data = equirect_utils.generate_planar_projections_from_equirectangular_GT(
            self.metadata, self.data, pers_size, 
            self.images_per_equirect, crop_factor=self.crop_factor, clip_output = False,
            use_mask = self.use_mask
        )
        self.camera_type = "perspective"
        metadata_dict = io.load_from_json(self.data / "transforms.json")
        # Copy images to output directory
        cropped_images_filename = []
        cropped_masks_filename = []
        for frame in metadata_dict["frames"]:
            cropped_images_filename.append(Path(frame["file_path"]))
            cropped_masks_filename.append(Path(frame["mask_path"]))
        # Copy images to output directory
        if self.is_HDR:
            copied_image_paths = process_data_utils.copy_images_list_EXR(
                cropped_images_filename, image_dir=image_dir, verbose=self.verbose, num_downscales=self.num_downscales
            )
        else:
            copied_image_paths = process_data_utils.copy_images_list(
                cropped_images_filename, image_dir=image_dir, verbose=self.verbose, num_downscales=self.num_downscales
            )
        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        
        if self.use_mask:
            copied_mask_paths = process_data_utils.copy_images_list(
                cropped_masks_filename, image_dir=mask_dir, verbose=self.verbose, num_downscales=self.num_downscales
            )
            copied_mask_paths = [Path("masks/" + copied_mask_path.name) for copied_mask_path in copied_mask_paths]

        num_frames = len(copied_image_paths)


        summary_log.append(f"Used {num_frames} images out of {num_frames} total")
        if self.max_dataset_size > 0:
            summary_log.append(
                "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
            )

        metadata_path = self.output_dir / "transforms.json"
        for i, frame in enumerate(metadata_dict["frames"]):
            frame["file_path"] = str(copied_image_paths[i])
            if self.use_mask:
                frame["mask_path"] = str(copied_mask_paths[i])

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=4)
        
        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()