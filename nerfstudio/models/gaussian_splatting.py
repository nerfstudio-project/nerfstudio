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
3D Gaussian Splatting model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

from torch.nn import Parameter


from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """3D Gaussian Splatting Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)


class GaussianSplattingModel(Model):
    """3D Gaussian Splatting Model

    Args:
        config: configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

        return {}
