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

"""Config used for sampling rays."""

from dataclasses import dataclass
from typing import Optional

from nerfstudio.configs.base_config import (
    InstantiateConfig,
)


@dataclass
class SamplingConfig(InstantiateConfig):
    """Full config contents for defining a pixel sampler."""

    patch_size: int = 1
    """Patch size for sampling rays."""
    method_name: Optional[str] = None
    """Method name. Required to set in python or via cli"""
