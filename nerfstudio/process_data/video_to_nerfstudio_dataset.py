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

"""Processes a video to a nerfstudio compatible dataset."""

import shutil
from dataclasses import dataclass

from rich.console import Console

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import (
    ColmapConverterToNerfstudioDataset,
)

CONSOLE = Console(width=120)


@dataclass
class VideoToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process videos into a nerfstudio dataset.

    This script does the following:

    1. Converts the video into images.
    2. Scales images to a specified size.
    3. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    num_frames_target: int = 300
    """Target number of frames to use for the dataset, results may not be exact."""
    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""

    def main(self) -> None:  # pylint: disable=R0915
        """Process video into a nerfstudio dataset."""

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
                image_dir=self.image_dir,
                num_frames_target=self.num_frames_target,
                crop_factor=self.crop_factor,
                verbose=self.verbose,
            )

        # Generate planar projections if equirectangular
        if self.camera_type == "equirectangular":
            perspective_image_size = equirect_utils.compute_resolution_from_equirect(
                self.output_dir / "temp_images", self.images_per_equirect
            )

            equirect_utils.generate_planar_projections_from_equirectangular(
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

            # remove the temp_images folder
            shutil.rmtree(self.output_dir / "temp_images", ignore_errors=True)

        # # Create mask
        mask_path = process_data_utils.save_mask(
            image_dir=self.image_dir,
            num_downscales=self.num_downscales,
            crop_factor=(0.0, 0.0, 0.0, 0.0),
            percent_radius=self.percent_radius_crop,
        )
        if mask_path is not None:
            summary_log.append(f"Saved mask to {mask_path}")

        # # Downscale images
        summary_log.append(
            process_data_utils.downscale_images(self.image_dir, self.num_downscales, verbose=self.verbose)
        )

        # Run Colmap
        if not self.skip_colmap:
            self._run_colmap(mask_path)

        # Export depth maps
        image_id_to_depth_path, log_tmp = self._export_depth()
        summary_log += log_tmp

        summary_log += self._save_transforms(num_extracted_frames, image_id_to_depth_path, mask_path)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)
