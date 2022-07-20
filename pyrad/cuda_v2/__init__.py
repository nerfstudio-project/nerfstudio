"""TODO: DocStr"""
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from pyrad.cuda_v2.backend import _C

packbits = _C.packbits
ray_aabb_intersect = _C.ray_aabb_intersect
morton3D = _C.morton3D
morton3D_invert = _C.morton3D_invert
raymarching_train = _C.raymarching_train
volumetric_rendering = _C.volumetric_rendering
volumetric_rendering_backward = _C.volumetric_rendering_backward

# pylint: disable=abstract-method,arguments-differ
class VolumeRenderer(torch.autograd.Function):
    """TODO: DocStr"""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, packed_info, positions, deltas, ts, sigmas, rgbs):
        accumulated_weight, accumulated_depth, accumulated_color, mask = volumetric_rendering(
            packed_info, positions, deltas, ts, sigmas, rgbs
        )
        ctx.save_for_backward(
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            packed_info,
            deltas,
            ts,
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
            deltas,
            ts,
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
            deltas,
            ts,
            sigmas,
            rgbs,
        )
        # grad_sigmas *= 1e4
        # grad_rgbs *= 1e4
        # print(
        #     "backward input",
        #     grad_color.min(),
        #     grad_color.max(),
        #     "backward output",
        #     grad_sigmas.min(),
        #     grad_sigmas.max(),
        #     grad_rgbs.min(),
        #     grad_rgbs.max(),
        # )
        # if grad_sigmas.isnan().any():
        #     print(
        #         "input",
        #         accumulated_weight.min(),
        #         accumulated_weight.max(),
        #         accumulated_depth.min(),
        #         accumulated_depth.max(),
        #         accumulated_color.min(),
        #         accumulated_color.max(),
        #         grad_weight.min(),
        #         grad_weight.max(),
        #         grad_depth.min(),
        #         grad_depth.max(),
        #         grad_color.min(),
        #         grad_color.max(),
        #         packed_info.min(),
        #         packed_info.max(),
        #         deltas.min(),
        #         deltas.max(),
        #         ts.min(),
        #         ts.max(),
        #         sigmas.min(),
        #         sigmas.max(),
        #         rgbs.min(),
        #         rgbs.max(),
        #     )

        #     exit()
        # print(grad_weight.mean(), grad_depth.mean(), grad_color.mean())
        # print(grad_sigmas.shape, grad_rgbs.shape, grad_sigmas.mean(), grad_rgbs.mean())
        # grad_sigmas = torch.zeros_like(sigmas)
        # grad_rgbs = torch.zeros_like(rgbs)
        return None, None, None, None, grad_sigmas, grad_rgbs
