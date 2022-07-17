#include "utils.h"


template <typename scalar_t>
__global__ void packbits_kernel(
    const scalar_t* __restrict__ density_grid,
    const int N,
    const float density_threshold,
    uint8_t* density_bitfield
){
    // parallel per byte
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    density_grid += n * 8;
    uint8_t bits = 0;

    #pragma unroll 8
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (density_grid[i] > density_threshold) ? ((uint8_t)1 << i) : 0;
    }

    density_bitfield[n] = bits;
}


void packbits_cu(
    const torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
){
    const int N = density_bitfield.size(0);

    const int threads = 256, blocks = (N+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(density_grid.type(), "packbits_cu", 
    ([&] {
        packbits_kernel<scalar_t><<<blocks, threads>>>(
            density_grid.data_ptr<scalar_t>(),
            N,
            density_threshold,
            density_bitfield.data_ptr<uint8_t>()
        );
    }));
}