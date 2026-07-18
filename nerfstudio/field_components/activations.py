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
Special activation functions.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

try:
    # torch >= 2.4: custom_fwd/custom_bwd live in torch.amp and require device_type
    from torch.amp import custom_bwd as _custom_bwd, custom_fwd as _custom_fwd

    def custom_fwd(*args, **kwargs):
        return _custom_fwd(*args, device_type="cuda", **kwargs)

    custom_bwd = _custom_bwd(device_type="cuda")
except ImportError:
    # torch < 2.4: custom_fwd/custom_bwd live in torch.cuda.amp (no device_type argument)
    from torch.cuda.amp import custom_bwd, custom_fwd  # type: ignore


class _TruncExp(Function):
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


if TYPE_CHECKING:

    def trunc_exp(_: Float[Tensor, "*bs"], /) -> Float[Tensor, "*bs"]:
        """Same as torch.exp, but with the backward pass clipped to prevent vanishing/exploding
        gradients."""
        raise NotImplementedError()

else:
    trunc_exp = _TruncExp.apply
    """Same as torch.exp, but with the backward pass clipped to prevent vanishing/exploding
    gradients."""
