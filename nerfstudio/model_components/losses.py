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
Collection of Losses.
"""
from enum import Enum

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7

# Sigma scale factor from Urban Radiance Fields (Rematas et al., 2022)
URF_SIGMA_SCALE_FACTOR = 3.0


class DephtLossType(Enum):
    """Types of depth losses for depth supervision."""

    DS_NERF = 1
    URF = 2


def outer(
    t0_starts: TensorType[..., "num_samples_0"],
    t0_ends: TensorType[..., "num_samples_0"],
    t1_starts: TensorType[..., "num_samples_1"],
    t1_ends: TensorType[..., "num_samples_1"],
    y1: TensorType[..., "num_samples_1"],
) -> TensorType[..., "num_samples_0"]:
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(
    t: TensorType[..., "num_samples+1"],
    w: TensorType[..., "num_samples"],
    t_env: TensorType[..., "num_samples+1"],
    w_env: TensorType[..., "num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping histogram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def interlevel_loss(weights_list, ray_samples_list):
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel


# Verified
def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights_list, ray_samples_list):
    """From mipnerf360"""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


def nerfstudio_distortion_loss(
    ray_samples: RaySamples,
    densities: TensorType["bs":..., "num_samples", 1] = None,
    weights: TensorType["bs":..., "num_samples", 1] = None,
) -> TensorType["bs":..., 1]:
    """Ray based distortion loss proposed in MipNeRF-360. Returns distortion Loss.

    .. math::

        \\mathcal{L}(\\mathbf{s}, \\mathbf{w}) =\\iint\\limits_{-\\infty}^{\\,\\,\\,\\infty}
        \\mathbf{w}_\\mathbf{s}(u)\\mathbf{w}_\\mathbf{s}(v)|u - v|\\,d_{u}\\,d_{v}

    where :math:`\\mathbf{w}_\\mathbf{s}(u)=\\sum_i w_i \\mathbb{1}_{[\\mathbf{s}_i, \\mathbf{s}_{i+1})}(u)`
    is the weight at location :math:`u` between bin locations :math:`s_i` and :math:`s_{i+1}`.

    Args:
        ray_samples: Ray samples to compute loss over
        densities: Predicted sample densities
        weights: Predicted weights from densities and sample locations
    """
    if torch.is_tensor(densities):
        assert not torch.is_tensor(weights), "Cannot use both densities and weights"
        # Compute the weight at each sample location
        weights = ray_samples.get_weights(densities)
    if torch.is_tensor(weights):
        assert not torch.is_tensor(densities), "Cannot use both densities and weights"

    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends

    assert starts is not None and ends is not None, "Ray samples must have spacing starts and ends"
    midpoints = (starts + ends) / 2.0  # (..., num_samples, 1)

    loss = (
        weights * weights[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
    )  # (..., num_samples, num_samples)
    loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
    loss = loss + 1 / 3.0 * torch.sum(weights**2 * (ends - starts), dim=-2)

    return loss


def orientation_loss(
    weights: TensorType["bs":..., "num_samples", 1],
    normals: TensorType["bs":..., "num_samples", 3],
    viewdirs: TensorType["bs":..., 3],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    v = viewdirs
    n_dot_v = (n * v[..., None, :]).sum(axis=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


def pred_normal_loss(
    weights: TensorType["bs":..., "num_samples", 1],
    normals: TensorType["bs":..., "num_samples", 3],
    pred_normals: TensorType["bs":..., "num_samples", 3],
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)


def ds_nerf_depth_loss(
    weights: TensorType[..., "num_samples", 1],
    termination_depth: TensorType[..., 1],
    steps: TensorType[..., "num_samples", 1],
    lengths: TensorType[..., "num_samples", 1],
    sigma: TensorType[0],
) -> TensorType[..., 1]:
    """Depth loss from Depth-supervised NeRF (Deng et al., 2022).

    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        steps: Sampling distances along rays.
        lengths: Distances between steps.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    loss = -torch.log(weights + EPS) * torch.exp(-((steps - termination_depth[:, None]) ** 2) / (2 * sigma)) * lengths
    return torch.mean(loss.sum(-2))


def urban_radiance_field_depth_loss(
    weights: TensorType[..., "num_samples", 1],
    termination_depth: TensorType[..., 1],
    predicted_depth: TensorType[..., 1],
    steps: TensorType[..., "num_samples", 1],
    sigma: TensorType[0],
) -> TensorType[..., 1]:
    """Lidar losses from Urban Radiance Fields (Rematas et al., 2022).

    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        steps: Sampling distances along rays.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    # Expected depth loss
    expected_depth_loss = (termination_depth - predicted_depth) ** 2

    # Line of sight losses
    target_distribution = torch.distributions.normal.Normal(0.0, sigma / URF_SIGMA_SCALE_FACTOR)
    termination_depth = termination_depth[:, None]
    line_of_sight_loss_near_mask = torch.logical_and(
        steps <= termination_depth + sigma, steps >= termination_depth - sigma
    )
    line_of_sight_loss_near = (weights - torch.exp(target_distribution.log_prob(steps - termination_depth))) ** 2
    line_of_sight_loss_near = (line_of_sight_loss_near_mask * line_of_sight_loss_near).sum(-2)
    line_of_sight_loss_empty_mask = steps < termination_depth - sigma
    line_of_sight_loss_empty = (line_of_sight_loss_empty_mask * weights**2).sum(-2)
    line_of_sight_loss = line_of_sight_loss_near + line_of_sight_loss_empty

    return torch.mean(expected_depth_loss + line_of_sight_loss)


def depth_loss(
    weights: TensorType[..., "num_samples", 1],
    ray_samples: RaySamples,
    termination_depth: TensorType[..., 1],
    predicted_depth: TensorType[..., 1],
    sigma: TensorType[0],
    directions_norm: TensorType[..., 1],
    is_euclidean: bool,
    depth_loss_type: DephtLossType,
) -> TensorType[0]:
    """Implementation of depth losses.

    Args:
        weights: Weights predicted for each sample.
        ray_samples: Samples along rays corresponding to weights.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        sigma: Uncertainty around depth value.
        directions_norm: Norms of ray direction vectors in the camera frame.
        is_euclidean: Whether ground truth depths corresponds to normalized direction vectors.
        depth_loss_type: Type of depth loss to apply.

    Returns:
        Depth loss scalar.
    """
    if not is_euclidean:
        termination_depth = termination_depth / directions_norm
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

    if depth_loss_type == DephtLossType.DS_NERF:
        lengths = ray_samples.frustums.ends - ray_samples.frustums.starts
        return ds_nerf_depth_loss(weights, termination_depth, steps, lengths, sigma)

    if depth_loss_type == DephtLossType.URF:
        return urban_radiance_field_depth_loss(weights, termination_depth, predicted_depth, steps, sigma)

    raise NotImplementedError("Provided depth loss type not implemented.")
