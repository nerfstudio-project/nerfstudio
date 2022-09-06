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

"""CUDA Functions"""
from typing import Callable

import torch
from torch.cuda.amp import custom_bwd, custom_fwd


def _make_lazy_cuda(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from nerfactory.cuda.backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


packbits = _make_lazy_cuda("packbits")
ray_aabb_intersect = _make_lazy_cuda("ray_aabb_intersect")
morton3D = _make_lazy_cuda("morton3D")
morton3D_invert = _make_lazy_cuda("morton3D_invert")
raymarching = _make_lazy_cuda("raymarching")
volumetric_rendering_forward = _make_lazy_cuda("volumetric_rendering_forward")
volumetric_rendering_backward = _make_lazy_cuda("volumetric_rendering_backward")
occupancy_query = _make_lazy_cuda("occupancy_query")

# pylint: disable=abstract-method,arguments-differ
class VolumeRenderer(torch.autograd.Function):
    """CUDA Volumetirc Renderer"""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, packed_info, starts, ends, sigmas, rgbs, opacities):
        accumulated_weight, accumulated_depth, accumulated_color, mask = volumetric_rendering_forward(
            packed_info, starts, ends, sigmas, rgbs, opacities
        )
        # TODO(ruilongli): accelerate for torch.no_grad()?
        ctx.save_for_backward(
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            packed_info,
            starts,
            ends,
            sigmas,
            rgbs,
            opacities,
        )
        return accumulated_weight, accumulated_depth, accumulated_color, mask

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weight, grad_depth, grad_color, _grad_mask):
        (
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            packed_info,
            starts,
            ends,
            sigmas,
            rgbs,
            opacities,
        ) = ctx.saved_tensors
        grad_sigmas, grad_rgbs = volumetric_rendering_backward(
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            grad_weight,
            grad_depth,
            grad_color,
            packed_info,
            starts,
            ends,
            sigmas,
            rgbs,
            opacities,
        )
        # corresponds to the input argument list of forward()
        return None, None, None, grad_sigmas, grad_rgbs, None


def unpackbits(x: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bits back to a boolen mask.

    Args:
        x: uint8 bit tensor with shape [N]

    Returns:
        unpacked boolen tensor with shape [N * 8]
    """
    assert x.dtype == torch.uint8 and x.dim() == 1
    bits = x.element_size() * 8
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).flip(-1).bool().reshape(-1)
