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

"""Implements the temporal grid used by NeRFPlayer (https://arxiv.org/abs/2210.15947).
A time conditioned sliding window is applied on the feature channels, so
that the feature vectors become time-aware.
(A large) Part of the code are adapted from (@ashawkey) https://github.com/ashawkey/torch-ngp/
"""
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torchtyping import TensorType

import nerfstudio.field_components.cuda as _C


# pylint: disable=abstract-method, arguments-differ
class TemporalGridEncodeFunc(Function):
    """Class for autograd in pytorch."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        inputs: TensorType["bs", "input_dim"],
        temporal_row_index: TensorType["bs", "temporal_index_dim"],
        embeddings: TensorType["table_size", "embed_dim"],
        offsets: TensorType["num_levels+1"],
        per_level_scale: float,
        base_resolution: int,
        calc_grad_inputs: bool = False,
        gridtype: int = 0,
        align_corners: bool = False,
    ) -> TensorType["bs", "output_dim"]:
        """Call forward and interpolate the feature from embeddings

        Args:
            inputs: the input coords
            temporal_row_index: the input index of channels for doing the interpolation
            embeddings: the saved (hashing) table for the feature grid (of the full sequence)
            offsets: offsets for each level in the multilevel table, used for locating in cuda kernels
            per_level_scale: scale parameter for the table; same as InstantNGP
            base_resolution: base resolution for the table; same as InstantNGP
            calc_grad_inputs: bool indicator for calculating gradients on the inputs
            gridtype: 0 == hash, 1 == tiled; tiled is a baseline in InstantNGP (not random collision)
            align_corners: same as other interpolation operators
        """

        inputs = inputs.contiguous()
        temporal_row_index = temporal_row_index.contiguous()

        B, D = inputs.shape  # batch size, coord dim
        L = offsets.shape[0] - 1  # level
        grid_channel = embeddings.shape[1]  # embedding dim for each level
        C = temporal_row_index.shape[1] // 4  # output embedding dim for each level
        S = np.log2(per_level_scale)  # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution  # base resolution

        # torch.half used by torch-ngp, but we disable it
        # (could be of negative impact on the performance? not sure, but feel free to inform me and help improve it!)
        # # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        # if torch.is_autocast_enabled() and C % 2 == 0:
        #     embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        _C.temporal_grid_encode_forward(
            inputs,
            temporal_row_index,
            embeddings,
            offsets,
            outputs,
            B,
            D,
            grid_channel,
            C,
            L,
            S,
            H,
            dy_dx,
            gridtype,
            align_corners,
        )

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, temporal_row_index, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, grid_channel, C, L, S, H, gridtype]
        ctx.align_corners = align_corners

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        inputs, temporal_row_index, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, grid_channel, C, L, S, H, gridtype = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings).contiguous()

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _C.temporal_grid_encode_backward(
            grad,
            inputs,
            temporal_row_index,
            embeddings,
            offsets,
            grad_embeddings,
            B,
            D,
            grid_channel,
            C,
            L,
            S,
            H,
            dy_dx,
            grad_inputs,
            gridtype,
            align_corners,
        )

        if grad_inputs is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, None, grad_embeddings, None, None, None, None, None, None


class TemporalGridEncoder(nn.Module):
    """Class for temporal grid encoding.
    This class extends the grid encoding (from InstantNGP) by allowing the output time-dependent feature channels.
    For example, for time 0 the interpolation uses channels [0,1], then for time 1 channels [2,1] are used.
    This operation can be viewed as applying a time-dependent sliding window on the feature channels.

    Args:
        temporal_dim: the dimension of temporal modeling; a higher dim indicates a higher freq on the time axis
        input_dim: the dimension of input coords
        num_levels: number of levels for multi-scale hashing; same as InstantNGP
        level_dim: the dim of output feature vector for each level; same as InstantNGP
        per_level_scale: scale factor; same as InstantNGP
        base_resolution: base resolution for the table; same as InstantNGP
        log2_hashmap_size: the size of the table; same as InstantNGP
        desired_resolution: desired resolution at the last level; same as InstantNGP
        gridtype: "tiled" or "hash"
        align_corners: same as other interpolation operators
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        input_dim: int = 3,
        num_levels: int = 16,
        level_dim: int = 2,
        per_level_scale: float = 2.0,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        desired_resolution: Optional[int] = None,
        gridtype: str = "hash",
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.temporal_dim = temporal_dim
        self.input_dim = input_dim  # coord dims, 2 or 3
        self.num_levels = num_levels  # num levels, each level multiply resolution by 2
        self.level_dim = level_dim  # encode channels per level
        self.per_level_scale = per_level_scale  # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        _gridtype_to_id = {"hash": 0, "tiled": 1}
        self.gridtype_id = _gridtype_to_id[gridtype]  # "tiled" or "hash"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2**log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale**i))
            params_in_level = min(
                self.max_params, (resolution if align_corners else resolution + 1) ** input_dim
            )  # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer("offsets", offsets)
        self.n_params = offsets[-1] * level_dim
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim + temporal_dim))
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize the parameters:
        1. Uniform initialization of the embeddings
        2. Temporal interpolation index initialization:
            For each temporal dim, we initialize a interpolation candidate.
            For example, if temporal dim 0, we use channels [0,1,2,3], then for temporal dim 1,
            we use channels [4,1,2,3]. After that, temporal dim 2, we use channels [4,5,2,3].
            This is for the alignment of the channels. I.e., each temporal dim should differ
            on only one channel, otherwise moving from one temporal dim to the next one is not
            that consistent.
            To associate time w.r.t. temporal dim, we evenly distribute time into the temporal dims.
            That is, if we have 16 temporal dims, then the 16th channel combinations is the time 1.
            (Time should be within 0 and 1.) Given a time, we first look up which temporal dim should
            be used. And then compute the linear combination weights.
            For implementing it, a table for all possible channel combination are used. Each row in
            the table is the candidate feature channels, and means we move from one temporal dim to
            the next one. For example, the first row will use feature channels [0,1,2,3,4]. Each row
            is of length `num_of_output_channel*4`. The expanding param 4 is for saving the combination
            weights and channels. The first row will be [?,0,?,1, 1,2,0,0, 1,3,0,0, 1,4,0,0]. Each
            4 tuple means
                `[weight_for_channel_A, index_for_channel_A, weight_for_channel_B, index_for_channel_B]`
            If `weight_for_channel_A` is 1, then there is no interpolation on this channel.
        """
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)
        # generate sampling index
        temporal_grid_rows = self.temporal_dim
        index_init = [0, self.level_dim] + list(range(1, self.level_dim))
        permute_base = list(range(2, self.level_dim + 1))
        last_entry = 0  # insert into ith place
        permute_init = permute_base[:last_entry] + [0] + permute_base[last_entry:]
        index_list = [torch.as_tensor(index_init, dtype=torch.long)]
        permute_list = [torch.as_tensor(permute_init, dtype=torch.long)]

        # converts a list of channel candidates into sampling row
        def to_sampling_index(index, permute, last_entry):
            row = index[permute]
            row = torch.stack([torch.ones_like(row), row, torch.zeros_like(row), torch.zeros_like(row)], 1)
            row = row.reshape([-1])
            mask_a = torch.zeros_like(row).bool()
            mask_b = torch.zeros_like(row).bool()
            row[last_entry * 4 + 3] = index[1]
            mask_a[last_entry * 4] = 1
            mask_b[last_entry * 4 + 2] = 1
            return row, mask_a, mask_b

        row, mask_a, mask_b = to_sampling_index(index_list[0], permute_list[0], last_entry)
        sampling_index = [row]
        index_a_mask, index_b_mask = [mask_a], [mask_b]
        # iterate on all temporal grid to get all rows
        for _ in range(1, temporal_grid_rows - 1):
            # the following lines are a little confusing...
            # the basic idea is to keep a buffer and then move to the next channel
            last_entry += 1
            if last_entry >= self.level_dim:
                last_entry = 0
            last_index_max = index_list[-1].max().item()
            last_index_min = index_list[-1].min().item()
            tem_permute_list = permute_list[-1].clone()  # for rearrange
            tem_permute_list[tem_permute_list == 0] += 1
            prev = index_list[-1][1:][tem_permute_list - 1].tolist()
            prev.pop(last_entry)
            new_index = [last_index_min + 1, last_index_max + 1] + prev
            new_index = torch.as_tensor(new_index, dtype=torch.long)
            new_permute = permute_base[:last_entry] + [0] + permute_base[last_entry:]
            new_permute = torch.as_tensor(new_permute, dtype=torch.long)
            index_list.append(torch.as_tensor(new_index, dtype=torch.long))
            permute_list.append(torch.as_tensor(new_permute, dtype=torch.long))
            row, mask_a, mask_b = to_sampling_index(index_list[-1], permute_list[-1], last_entry)
            sampling_index.append(row)
            index_a_mask.append(mask_a)
            index_b_mask.append(mask_b)
        self.register_buffer("index_list", torch.stack(index_list))
        self.register_buffer("sampling_index", torch.stack(sampling_index))
        # index_a_mask and index_b_mask are for inserting the combination weights
        self.register_buffer("index_a_mask", torch.stack(index_a_mask))
        self.register_buffer("index_b_mask", torch.stack(index_b_mask))

    def __repr__(self) -> str:
        """For debug and logging purpose."""
        return (
            f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} "
            f"level_dim={self.level_dim} resolution={self.base_resolution} -> "
            f"{int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} "
            f"per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} "
            f"gridtype={self.gridtype} align_corners={self.align_corners}"
        )

    def get_temporal_index(self, time: TensorType["bs"]) -> TensorType["bs", "temporal_index_dim"]:
        """Convert the time into sampling index lists."""
        row_idx_value = time * (len(self.sampling_index) - 1)
        row_idx = row_idx_value.long()
        row_idx[time == 1] = len(self.sampling_index) - 1
        temporal_row_index = self.sampling_index[row_idx].float()
        mask_a = self.index_a_mask[row_idx]
        mask_b = self.index_b_mask[row_idx]
        temporal_row_index[mask_a] = row_idx + 1 - row_idx_value
        temporal_row_index[mask_b] = row_idx_value - row_idx
        return temporal_row_index

    def forward(self, xyz: TensorType["bs", "input_dim"], time: TensorType["bs", 1]) -> TensorType["bs", "output_dim"]:
        """Forward and sampling feature vectors from the embedding.

        Args:
            xyz: input coords, should be in [0,1]
            time: input time, should be in [0,1] with shape [bs, 1]
        """
        outputs = TemporalGridEncodeFunc.apply(
            xyz,
            self.get_temporal_index(time[:, 0].float()),
            self.embeddings,
            self.offsets,
            self.per_level_scale,
            self.base_resolution,
            xyz.requires_grad,
            self.gridtype_id,
            self.align_corners,
        )
        return outputs

    def get_temporal_tv_loss(self) -> TensorType[()]:
        """Apply TV loss on the temporal channels.
        Sample a random channel combination (i.e., row for the combination table),
        and then compute loss on it.
        """
        row_idx = torch.randint(0, len(self.index_list), [1]).item()
        feat_idx = self.index_list[row_idx]
        return (self.embeddings[:, feat_idx[0]] - self.embeddings[:, feat_idx[1]]).abs().mean()
