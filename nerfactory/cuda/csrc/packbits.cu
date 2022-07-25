#include "include/helpers_cuda.h"


template <typename scalar_t>
__global__ void kernel_packbits(
    const int N,
    const float threshold,
    const scalar_t* data,
    uint8_t* bitfield
){
    CUDA_GET_THREAD_ID(thread_id, N);

    data += thread_id * 8;
    bitfield += thread_id;

    uint8_t bits = 0;
    #pragma unroll 8
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (data[i] > threshold) ? ((uint8_t)1 << i) : 0;
    }
    bitfield[0] = bits;
}

/**
 * @brief Pack data into uint8 bits based on threshold.
 * 
 * @param data: Tensor with any shape. Total elements must be N * 8.
 * @param threshold:
 * @return Tensor: bitfield that has shape of [N]
 */
torch::Tensor packbits(
    const torch::Tensor data, const float threshold
) {
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    TORCH_CHECK(data.numel() % 8 == 0)

    const int N = data.numel() / 8;

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(N, threads);

    torch::Tensor bitfield = torch::empty(
        {N,}, data.options().dtype(torch::kUInt8));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data.scalar_type(), "packbits", 
        ([&] {
            kernel_packbits<scalar_t><<<blocks, threads>>>(
                N,
                threshold,
                data.data_ptr<scalar_t>(),
                bitfield.data_ptr<uint8_t>()
            );
        })
    );

    return bitfield;
}