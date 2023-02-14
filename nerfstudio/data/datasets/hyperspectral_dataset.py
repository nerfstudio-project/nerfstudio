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

"""
Hyperspectral dataset.
"""

from typing import Dict

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class HyperspectralDataset(InputDataset):
    """Dataset that returns hyperspectral images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert scale_factor == 1, 'Scale factors not yet supported for hyperspectral'
        assert (
            "hs_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["hs_filenames"] is not None
        )
        self.hs_filenames = self.metadata["hs_filenames"]

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.hs_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        hs_image = torch.load(filepath)
        assert hs_image.shape == (height, width, 128), "HS image has incorrect shape"

        return {"hs_image": hs_image}
