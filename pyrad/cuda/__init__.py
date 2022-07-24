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
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from pyrad.cuda.backend import _C

packbits = _C.packbits
ray_aabb_intersect = _C.ray_aabb_intersect
morton3D = _C.morton3D
morton3D_invert = _C.morton3D_invert
raymarching = _C.raymarching
volumetric_rendering_forward = _C.volumetric_rendering_forward
volumetric_rendering_backward = _C.volumetric_rendering_backward

# pylint: disable=abstract-method,arguments-differ
class VolumeRenderer(torch.autograd.Function):
    """CUDA Volumetirc Renderer"""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, packed_info, starts, ends, sigmas, rgbs):
        accumulated_weight, accumulated_depth, accumulated_color, mask = volumetric_rendering_forward(
            packed_info, starts, ends, sigmas, rgbs
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
        )
        # corresponds to the input argument list of forward()
        return None, None, None, grad_sigmas, grad_rgbs
