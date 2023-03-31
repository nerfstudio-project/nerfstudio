/*
 * Code adapted from (@ashawkey) https://github.com/ashawkey/torch-ngp/blob/main/gridencoder/src/gridencoder.cu
 * Author: @lsongx
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK( \
    x.scalar_type() == at::ScalarType::Float || \
    x.scalar_type() == at::ScalarType::Half || \
    x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor" \
)


// just for compatability of half precision in AT_DISPATCH_FLOATING_TYPES_AND_HALF...
static inline  __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
    // requires CUDA >= 10 and ARCH >= 70
    // this is very slow compared to float or __half2, and never used.
    // The following line was commented in torch-ngp; uncomment it in case someone want to try fp16...
    return atomicAdd(reinterpret_cast<__half*>(address), val);
}


template <typename T>
static inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}


template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t pos_grid[D]) {
    
    // coherent type of hashing
    constexpr uint32_t primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };

    uint32_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}

// locate the index of grid with channel (ch)
template <uint32_t D>
__device__ uint32_t get_grid_index(
    const uint32_t gridtype, // gridtype: 0 == hash, 1 == tiled
    const uint32_t grid_C, // number of total channels for the embedding
    const bool align_corners, 
    const uint32_t ch, // number of output channels
    const uint32_t hashmap_size, 
    const uint32_t resolution, 
    const uint32_t pos_grid[D] // location of the grid
) {
    uint32_t stride = 1;
    uint32_t index = 0;

    #pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        index += pos_grid[d] * stride;
        stride *= align_corners ? resolution: (resolution + 1);
    }

    // NOTE: for NeRF, the hash is in fact not necessary. Check https://github.com/NVlabs/instant-ngp/issues/97.
    // gridtype: 0 == hash, 1 == tiled
    if (gridtype == 0 && stride > hashmap_size) {
        index = fast_hash<D>(pos_grid);
    }

    return (index % hashmap_size) * grid_C + ch;
}

// grid sampling kernel
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid(
    const float * __restrict__ inputs, // input coordinates
    const float * __restrict__ temporal_row_index, // row index for sampling from channels
    const scalar_t * __restrict__ grid, // the grid embedding
    const int * __restrict__ offsets, // offsets for different levels used in NGP
    scalar_t * __restrict__ outputs,
    const uint32_t B, // batch size
    const uint32_t grid_C, // number of channels for the grid embedding
    const uint32_t L, // number of levels
    const float S, // resolution multiplier at each level
    const uint32_t H, // base resolution
    scalar_t * __restrict__ dy_dx, // gradient to the input coords
    const uint32_t gridtype, // gridtype: 0 == hash, 1 == tiled
    const bool align_corners
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    grid += (uint32_t)offsets[level] * grid_C;
    inputs += b * D;
    outputs += level * B * C + b * C;
    temporal_row_index += b*C*4;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0; 
        }
        if (dy_dx) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0; 
                }       
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    
    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    // interpolate
    scalar_t results[C] = {0}; // temp results in register

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        // writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            if (temporal_row_index[ch*4] == 1) {
                // if a temporal_row_index == 1, then no interpolation between channels are used
                uint32_t index = get_grid_index<D>(gridtype, grid_C, align_corners, 
                    __float2uint_rn(temporal_row_index[ch*4+1]), hashmap_size, resolution, pos_grid_local);
                    results[ch] += w * grid[index];
            } else {
                // else, we need to interpolate between channels
                uint32_t index_a = get_grid_index<D>(gridtype, grid_C, align_corners, 
                    __float2uint_rn(temporal_row_index[ch*4+1]), hashmap_size, resolution, pos_grid_local);
                uint32_t index_b = get_grid_index<D>(gridtype, grid_C, align_corners, 
                    __float2uint_rn(temporal_row_index[ch*4+3]), hashmap_size, resolution, pos_grid_local);
                results[ch] += w * (grid[index_a]*temporal_row_index[ch*4]+grid[index_b]*temporal_row_index[ch*4+2]);
            } 
        }
    }

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch]; 
    }

    // prepare dy_dx
    // differentiable (soft) indexing: https://discuss.pytorch.org/t/differentiable-indexing/17647/9
    if (dy_dx) {

        dy_dx += b * D * L * C + level * D * C; // B L D C

        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {

            scalar_t results_grad[C] = {0};

            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                float w = scale;
                uint32_t pos_grid_local[D];

                #pragma unroll
                for (uint32_t nd = 0; nd < D - 1; nd++) {
                    const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                    if ((idx & (1 << nd)) == 0) {
                        w *= 1 - pos[d];
                        pos_grid_local[d] = pos_grid[d];
                    } else {
                        w *= pos[d];
                        pos_grid_local[d] = pos_grid[d] + 1;
                    }
                }

                scalar_t left[C] = {0};
                scalar_t right[C] = {0};
                pos_grid_local[gd] = pos_grid[gd];
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    if (temporal_row_index[ch*4] == 1) {
                        uint32_t index = get_grid_index<D>(gridtype, grid_C, align_corners, 
                            __float2uint_rn(temporal_row_index[ch*4+1]), hashmap_size, resolution, pos_grid_local);
                        left[ch] += grid[index];
                    } else {
                        uint32_t index_a = get_grid_index<D>(gridtype, grid_C, align_corners, 
                            __float2uint_rn(temporal_row_index[ch*4+1]), hashmap_size, resolution, pos_grid_local);
                        uint32_t index_b = get_grid_index<D>(gridtype, grid_C, align_corners, 
                            __float2uint_rn(temporal_row_index[ch*4+3]), hashmap_size, resolution, pos_grid_local);
                        left[ch] += grid[index_a]*temporal_row_index[ch*4]+grid[index_b]*temporal_row_index[ch*4+2];
                    } 
                }

                pos_grid_local[gd] = pos_grid[gd] + 1;
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    if (temporal_row_index[ch*4] == 1) {
                        uint32_t index = get_grid_index<D>(gridtype, grid_C, align_corners, 
                            __float2uint_rn(temporal_row_index[ch*4+1]), hashmap_size, resolution, pos_grid_local);
                        right[ch] += grid[index];
                    } else {
                        uint32_t index_a = get_grid_index<D>(gridtype, grid_C, align_corners, 
                            __float2uint_rn(temporal_row_index[ch*4+1]), hashmap_size, resolution, pos_grid_local);
                        uint32_t index_b = get_grid_index<D>(gridtype, grid_C, align_corners, 
                            __float2uint_rn(temporal_row_index[ch*4+3]), hashmap_size, resolution, pos_grid_local);
                        right[ch] += grid[index_a]*temporal_row_index[ch*4]+grid[index_b]*temporal_row_index[ch*4+2];
                    } 
                }

                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    results_grad[ch] += w * (right[ch] - left[ch]);
                }
            }

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                dy_dx[gd * C + ch] = results_grad[ch];
            }
        }
    }
}

// grid backward kernel; N_C is kept here though always set to 1 (maybe useful for future updates)
template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward(
    const scalar_t * __restrict__ grad,
    const float * __restrict__ inputs, 
    const float * __restrict__ temporal_row_index, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grid, 
    const uint32_t B, 
    const uint32_t grid_C,
    const uint32_t L, 
    const float S, 
    const uint32_t H,
    const uint32_t gridtype,
    const bool align_corners
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    // embed channel (from the outputs)
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C; 

    // locate
    grad_grid += offsets[level] * grid_C;
    inputs += b * D;
    grad += level * B * C + b * C + ch; // L, B, C
    temporal_row_index += b*C*4 + ch*4;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    scalar_t grad_cur = grad[0];

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        // get the weights used for calculating the interpolation
        if (temporal_row_index[0] == 1) {
            // if no interpolation, directly return grad
            uint32_t index = get_grid_index<D>(gridtype, grid_C, align_corners, 
                __float2uint_rn(temporal_row_index[1]), hashmap_size, resolution, pos_grid_local);
            atomicAdd(&grad_grid[index], w*grad_cur);
        } else {
            // if interpolation used, return with weights multiplied
            uint32_t index_a = get_grid_index<D>(gridtype, grid_C, align_corners, 
                __float2uint_rn(temporal_row_index[1]), hashmap_size, resolution, pos_grid_local);
            uint32_t index_b = get_grid_index<D>(gridtype, grid_C, align_corners, 
                __float2uint_rn(temporal_row_index[3]), hashmap_size, resolution, pos_grid_local);
            atomicAdd(&grad_grid[index_a], temporal_row_index[0]*w*grad_cur);
            atomicAdd(&grad_grid[index_b], temporal_row_index[2]*w*grad_cur);
        }
    }
}

// coord backward kernel
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ dy_dx,  
    scalar_t * __restrict__ grad_inputs, 
    uint32_t B, 
    uint32_t L
) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= B * D) return;

    const uint32_t b = t / D;
    const uint32_t d = t - b * D;

    dy_dx += b * L * D * C;

    scalar_t result = 0;
    
    # pragma unroll
    for (int l = 0; l < L; l++) {
        # pragma unroll
        for (int ch = 0; ch < C; ch++) {
            result += grad[l * B * C + b * C + ch] * dy_dx[l * D * C + d * C + ch];
        }
    }

    grad_inputs[t] = result;
}


template <typename scalar_t, uint32_t D>
void kernel_grid_wrapper(
    const float *inputs, 
    const float *temporal_row_index, 
    const scalar_t *embeddings, 
    const int *offsets, 
    scalar_t *outputs, 
    const uint32_t B, 
    const uint32_t grid_C, 
    const uint32_t C, 
    const uint32_t L, 
    const float S, 
    const uint32_t H, 
    scalar_t *dy_dx, 
    const uint32_t gridtype, 
    const bool align_corners
) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, L, S, H, dy_dx, gridtype, align_corners); break;
        case 2: kernel_grid<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, L, S, H, dy_dx, gridtype, align_corners); break;
        case 4: kernel_grid<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, L, S, H, dy_dx, gridtype, align_corners); break;
        case 8: kernel_grid<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, L, S, H, dy_dx, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t>
void temporal_grid_encode_forward_cuda(
    const float *inputs, 
    const float *temporal_row_index, 
    const scalar_t *embeddings, 
    const int *offsets, 
    scalar_t *outputs, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t grid_C, 
    const uint32_t C, 
    const uint32_t L, 
    const float S, 
    const uint32_t H, 
    scalar_t *dy_dx, 
    const uint32_t gridtype, 
    const bool align_corners
) {
    switch (D) {
        case 1: kernel_grid_wrapper<scalar_t, 1>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, C, L, S, H, dy_dx, gridtype, align_corners); break;
        case 2: kernel_grid_wrapper<scalar_t, 2>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, C, L, S, H, dy_dx, gridtype, align_corners); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, C, L, S, H, dy_dx, gridtype, align_corners); break;
        case 4: kernel_grid_wrapper<scalar_t, 4>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, C, L, S, H, dy_dx, gridtype, align_corners); break;
        case 5: kernel_grid_wrapper<scalar_t, 5>(
            inputs, temporal_row_index, embeddings, offsets, outputs, 
            B, grid_C, C, L, S, H, dy_dx, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: D must be 1, 2, 3, 4, or 5."};
    }
}

template <typename scalar_t, uint32_t D>
void kernel_grid_backward_wrapper(
    const scalar_t *grad, 
    const float *inputs, 
    const float *temporal_row_index, 
    const scalar_t *embeddings, 
    const int *offsets, 
    scalar_t *grad_embeddings, 
    const uint32_t B, 
    const uint32_t grid_C, 
    const uint32_t C, 
    const uint32_t L, 
    const float S, 
    const uint32_t H, 
    scalar_t *dy_dx, 
    scalar_t *grad_inputs, 
    const uint32_t gridtype, 
    const bool align_corners
) {
    static constexpr uint32_t N_THREAD = 256;
    const dim3 blocks_hashgrid = { div_round_up(B * C, N_THREAD), L, 1 };
    switch (C) {
        case 1: 
            kernel_grid_backward<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(
                grad, inputs, temporal_row_index, 
                embeddings, offsets, grad_embeddings, B, grid_C, L, S, H, gridtype, align_corners); 
            if (dy_dx) {
                kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(
                    grad, dy_dx, grad_inputs, B, L);
            }
            break;
        case 2: 
            kernel_grid_backward<scalar_t, D, 2, 1><<<blocks_hashgrid, N_THREAD>>>(
                grad, inputs, temporal_row_index, 
                embeddings, offsets, grad_embeddings, B, grid_C, L, S, H, gridtype, align_corners);
            if (dy_dx) {
                kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(
                    grad, dy_dx, grad_inputs, B, L);
            }
            break;
        case 4: 
            kernel_grid_backward<scalar_t, D, 4, 1><<<blocks_hashgrid, N_THREAD>>>(
                grad, inputs, temporal_row_index, 
                embeddings, offsets, grad_embeddings, B, grid_C, L, S, H, gridtype, align_corners);
            if (dy_dx) {
                kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(
                    grad, dy_dx, grad_inputs, B, L);
            }
            break;
        case 8: 
            kernel_grid_backward<scalar_t, D, 8, 1><<<blocks_hashgrid, N_THREAD>>>(
                grad, inputs, temporal_row_index, 
                embeddings, offsets, grad_embeddings, B, grid_C, L, S, H, gridtype, align_corners);
            if (dy_dx) {
                kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(
                    grad, dy_dx, grad_inputs, B, L);
            }
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t>
void temporal_grid_encode_backward_cuda(
    const scalar_t *grad, 
    const float *inputs, 
    const float *temporal_row_index, 
    const scalar_t *embeddings, 
    const int *offsets, 
    scalar_t *grad_embeddings, 
    const uint32_t B, 
    const uint32_t D, 
    const uint32_t grid_C, 
    const uint32_t C, 
    const uint32_t L, 
    const float S, 
    const uint32_t H, 
    scalar_t *dy_dx, 
    scalar_t *grad_inputs, 
    const uint32_t gridtype, 
    const bool align_corners
) {
    switch (D) {
        case 1: kernel_grid_backward_wrapper<scalar_t, 1>(
            grad, inputs, temporal_row_index, embeddings, offsets, 
            grad_embeddings, B, grid_C, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners); break;
        case 2: kernel_grid_backward_wrapper<scalar_t, 2>(
            grad, inputs, temporal_row_index, embeddings, offsets, 
            grad_embeddings, B, grid_C, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(
            grad, inputs, temporal_row_index, embeddings, offsets, 
            grad_embeddings, B, grid_C, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners); break;
        case 4: kernel_grid_backward_wrapper<scalar_t, 4>(
            grad, inputs, temporal_row_index, embeddings, offsets, 
            grad_embeddings, B, grid_C, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners); break;
        case 5: kernel_grid_backward_wrapper<scalar_t, 5>(
            grad, inputs, temporal_row_index, embeddings, offsets, 
            grad_embeddings, B, grid_C, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: D must be 1, 2, 3, 4, or 5."};
    }
}

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
) {
    CHECK_INPUT(inputs);
    CHECK_INPUT(temporal_row_index);
    CHECK_INPUT(embeddings);
    CHECK_INPUT(offsets);
    CHECK_INPUT(outputs);
    // CHECK_INPUT(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(temporal_row_index);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(outputs);
    // CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "temporal_grid_encode_forward", ([&] {
        temporal_grid_encode_forward_cuda<scalar_t>(
            inputs.data_ptr<float>(), temporal_row_index.data_ptr<float>(), embeddings.data_ptr<scalar_t>(), 
            offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, D, grid_C, C, L, S, H, 
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, 
            gridtype, align_corners);
    }));
}

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
) {
    CHECK_INPUT(grad);
    CHECK_INPUT(inputs);
    CHECK_INPUT(temporal_row_index);
    CHECK_INPUT(embeddings);
    CHECK_INPUT(offsets);
    CHECK_INPUT(grad_embeddings);
    // CHECK_INPUT(dy_dx);
    // CHECK_INPUT(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(temporal_row_index);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad_embeddings);
    // CHECK_IS_FLOATING(dy_dx);
    // CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "temporal_grid_encode_backward", ([&] {
        temporal_grid_encode_backward_cuda<scalar_t>(
            grad.data_ptr<scalar_t>(), inputs.data_ptr<float>(), temporal_row_index.data_ptr<float>(), 
            embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<scalar_t>(), 
            B, D, grid_C, C, L, S, H, 
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, 
            grad_inputs.has_value() ? grad_inputs.value().data_ptr<scalar_t>() : nullptr, 
            gridtype, align_corners);
    }));
}
