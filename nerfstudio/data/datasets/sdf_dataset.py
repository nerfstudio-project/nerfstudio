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

"""
SDFStudio dataset.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class SDFDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth", "normal"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        # can be none if monoprior not included
        self.depth_filenames = self.metadata["depth_filenames"]
        self.normal_filenames = self.metadata["normal_filenames"]
        self.camera_to_worlds = self.metadata["camera_to_worlds"]
        # can be none if auto orient not enabled in dataparser
        self.transform = self.metadata["transform"]
        self.include_mono_prior = self.metadata["include_mono_prior"]

    def get_metadata(self, data: Dict) -> Dict:
        # TODO supports foreground_masks
        metadata = {}
        if self.include_mono_prior:
            depth_filepath = self.depth_filenames[data["image_idx"]]
            normal_filepath = self.normal_filenames[data["image_idx"]]
            camtoworld = self.camera_to_worlds[data["image_idx"]]

            # Scale depth images to meter units and also by scaling applied to cameras
            depth_image, normal_image = self.get_depths_and_normals(
                depth_filepath=depth_filepath, normal_filename=normal_filepath, camtoworld=camtoworld
            )
            metadata["depth"] = depth_image
            metadata["normal"] = normal_image

        return metadata

    def get_depths_and_normals(self, depth_filepath: Path, normal_filename: Path, camtoworld: np.ndarray):
        """function to process additional depths and normal information
        Args:
            depth_filepath: path to depth file
            normal_filename: path to normal file
            camtoworld: camera to world transformation matrix
        """

        # load mono depth
        depth = np.load(depth_filepath)
        depth = torch.from_numpy(depth).float()

        # load mono normal
        normal = np.load(normal_filename)

        # transform normal to world coordinate system
        normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
        normal = torch.from_numpy(normal).float()

        rot = torch.from_numpy(camtoworld[:3, :3])

        normal_map = normal.reshape(3, -1)
        normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

        normal_map = rot @ normal_map
        normal = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)

        if self.transform is not None:
            h, w, _ = normal.shape
            normal = self.transform[:3, :3] @ normal.reshape(-1, 3).permute(1, 0)
            normal = normal.permute(1, 0).reshape(h, w, 3)

        return depth, normal
