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

from typing import Literal, Tuple

import torch


def get_best_device() -> Tuple[Literal["cpu", "cuda", "mps"], str]:
    """Determine the best available device to run nerfstudio inference.

    Returns:
        tuple: (device_type, reason) where device_type is the selected device and
        reason is an explanation of why it was chosen
    """
    if torch.cuda.is_available():
        return "cuda", "CUDA GPU available - using for optimal performance"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Metal (MPS) available - using for accelerated performance"
    else:
        return "cpu", "No GPU/MPS detected - falling back to CPU"
