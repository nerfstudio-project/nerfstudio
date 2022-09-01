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
Pose and Intrinsics Optimizers
"""
from enum import Enum, auto
from turtle import forward
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torchtyping import TensorType


class PoseOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    def __init__(self) -> None:
        super().__init__()
        # maybe some other stuff?

    def forward(self, c2w: TensorType["num_cameras", 3, 4]) -> TensorType["num_cameras", 3, 4]:
        return c2w  # no-op
