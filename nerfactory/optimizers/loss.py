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
Collection of Losses.
"""

import torch
from torch import nn

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7


def searchsorted(a, v):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L109
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L30
    """
    i = torch.arange(a.shape[-1], device=a.device)
    v_ge_a = v[..., None, :] >= a[..., :, None]
    idx_lo = torch.where(v_ge_a, i[..., :, None], i[..., :1, None]).max(dim=-2).values
    idx_hi = torch.where(~v_ge_a, i[..., :, None], i[..., -1:, None]).min(dim=-2).values
    return idx_lo, idx_hi


def inner_outer(t0, t1, y1):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64
    """

    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)
    idx_lo, idx_hi = searchsorted(t1, t0)

    cy1_lo = torch.take_along_dim(cy1, idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1, idx_hi, dim=-1)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]
    y0_inner = torch.where(
        idx_hi[..., :-1] <= idx_lo[..., 1:],
        cy1_lo[..., 1:] - cy1_hi[..., :-1],
        torch.zeros_like(cy1_lo[..., 1:]),
    )

    return y0_inner, y0_outer


def lossfun_outer(t, w, t_env, w_env):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80
    """
    _, w_outer = inner_outer(t, t_env, w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def ray_samples_to_sdist(ray_samples, near_plane, far_plane):
    """Convert ray samples to s space"""
    starts = (ray_samples.frustums.starts - near_plane) / (far_plane - near_plane)
    ends = (ray_samples.frustums.ends - near_plane) / (far_plane - near_plane)
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def interlevel_loss(weights_list, ray_samples_list, near_plane, far_plane):
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1], near_plane, far_plane).detach()
    w = weights_list[-1][..., 0].detach()
    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples, near_plane, far_plane)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel
