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
Put all the method implementations in one location.
"""

from nerfactory.configs.instant_ngp_config import InstantNGPConfig
from nerfactory.configs.mipnerf_360_config import MipNerf360Config
from nerfactory.configs.mipnerf_config import MipNerfConfig
from nerfactory.configs.nerfw_config import NerfWConfig
from nerfactory.configs.semantic_nerf_config import SemanticNerfConfig
from nerfactory.configs.vanilla_nerf_config import VanillaNerfConfig

base_configs = {
    "instant_ngp": InstantNGPConfig(),
    "mipnerf_360": MipNerf360Config(),
    "mipnerf": MipNerfConfig(),
    "nerfw": NerfWConfig(),
    "semantic_nerf": SemanticNerfConfig(),
    "vanilla_nerf": VanillaNerfConfig(),
}
