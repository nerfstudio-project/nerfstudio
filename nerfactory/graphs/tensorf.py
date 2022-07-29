# Copyright 2022 The Plenoptix Team. All rights reserved.
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
TensorRF implementation.
"""

from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure

from pyrad.fields.modules.encoding import TensorVMEncoding, TensorCPEncoding
from pyrad.fields.nerf_field import NeRFField
from pyrad.graphs.vanilla_nerf import NeRFGraph


class TensoRFGraph(NeRFGraph):
    """
    TensoRF Graph
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def populate_fields(self):
        """Set the fields."""

        position_encoding = TensorVMEncoding(resolution=512, num_components=24)
        direction_encoding = TensorVMEncoding(resolution=64, num_components=16)

        self.field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
