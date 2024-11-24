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

from typing import List, Literal

import torch


def get_available_devices() -> List[Literal["cpu", "cuda", "mps"]]:
    """Determine the available devices on the machine

    Returns:
        list: List of available device types
    """
    available_devices: List[Literal["cpu", "cuda", "mps"]] = []
    if torch.cuda.is_available():
        available_devices.append("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        available_devices.append("mps")
    available_devices.append("cpu")
    return available_devices
