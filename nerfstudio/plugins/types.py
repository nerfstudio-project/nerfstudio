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
This package contains specifications used to register plugins.
"""

from dataclasses import dataclass

from nerfstudio.engine.trainer import TrainerConfig


@dataclass
class MethodSpecification:
    """
    Method specification class used to register custom methods with Nerfstudio.
    The registered methods will be available in commands such as `ns-train`
    """

    config: TrainerConfig
    """Trainer configuration"""
    description: str
    """Method description shown in `ns-train` help"""
