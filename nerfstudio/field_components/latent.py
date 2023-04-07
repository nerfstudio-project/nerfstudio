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
Latent Data Structure
"""

from typing import TypedDict

import torch
from torchtyping import TensorType # FIXME - 이걸 사용해서 latent 설정을 좀 잘해둬야 겠다. 


# @dataclass
# class Latent:
#     """Dataparser outputs for the which will be used by the DataManager
#     for creating RayBundle and RayGT objects."""

#     sampler : 

#     latent shape

#     latent 종류 film인지 뭔지 등등.. # 이거는 MLP input인 latent임 따라서 여기서 정의하는 것은 적절하지 못함. 

#     def sampler


#     """Scale applied by the dataparser."""
#  # z sampler를 여기에다가 둘까?


class FilmLatent(TypedDict):
    freq : torch.Tensor
    phase_shift : torch.Tensor

class WLatent(TypedDict):
    latent : torch.Tensor