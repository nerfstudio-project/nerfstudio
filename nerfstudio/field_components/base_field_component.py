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
The field module baseclass.
"""

from abc import abstractmethod
from typing import Optional

from jaxtyping import Shaped
from torch import Tensor, nn


class FieldComponent(nn.Module):
    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Output dimension to module.
    """

    def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        """
        Returns processed tensor

        Args:
            in_tensor: Input tensor to process
        """
        raise NotImplementedError
