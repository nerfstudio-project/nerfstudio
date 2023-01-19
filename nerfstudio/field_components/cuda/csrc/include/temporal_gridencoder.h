/*
 * Code adapted from (@ashawkey) https://github.com/ashawkey/torch-ngp/
 * Author: @lsongx
 */

#ifndef _TEMPORAL_HASH_ENCODE_H
#define _TEMPORAL_HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

// inputs: input coordinates, [B, D], float, in [0, 1]
// temporal_row_index: row index for sampling from channels, [B, 4*num_of_channels], uint32_t
// embeddings: the grid embedding, [sO, grid_C], float
// offsets: offsets for different levels used in NGP, [L + 1], uint32_t
// outputs: interpolated outputs, [B, L * C], float
// grid_C: number of channels for the grid embedding
// B: batch size 
// D: coord dim
// L: number of levels
// S: resolution multiplier at each level
// H: base resolution

void temporal_grid_encode_forward(
    const at::Tensor inputs, 
    const at::Tensor temporal_row_index, 
    const at::Tensor embeddings, 
    const at::Tensor offsets, 
    at::Tensor outputs, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t grid_C, 
    const uint32_t C, 
    const uint32_t L, 
    const float S, 
    const uint32_t H, 
    at::optional<at::Tensor> dy_dx, 
    const uint32_t gridtype, 
    const bool align_corners
);

void temporal_grid_encode_backward(
    const at::Tensor grad, 
    const at::Tensor inputs, 
    const at::Tensor temporal_row_index, 
    const at::Tensor embeddings, 
    const at::Tensor offsets, 
    at::Tensor grad_embeddings, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t grid_C, 
    const uint32_t C, 
    const uint32_t L, 
    const float S, 
    const uint32_t H, 
    const at::optional<at::Tensor> dy_dx, 
    at::optional<at::Tensor> grad_inputs, 
    const uint32_t gridtype, 
    const bool align_corners
);

#endif