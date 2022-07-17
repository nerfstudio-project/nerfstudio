#include "include/helper_cuda.h"


inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t xx = __expand_bits(x);
    uint32_t yy = __expand_bits(y);
    uint32_t zz = __expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

template <typename index_t>
__global__ void kernel_morton3D(
    const int N,
    const index_t* coords,
    index_t* indices
){
    CUDA_GET_THREAD_ID(thread_id, N);

    // locate
    coords += thread_id * 3;
    indices += thread_id;

    uint32_t index = __morton3D(
        static_cast<uint32_t>(coords[0]), 
        static_cast<uint32_t>(coords[1]), 
        static_cast<uint32_t>(coords[2])
    );
    *indices = static_cast<index_t>(index);
    return;
}

/**
 * @brief Convert coords to indices
 * 
 * @param coords Shape [N, 3]
 * @return torch::Tensor Indices with shape [N]
 */
torch::Tensor morton3D(const torch::Tensor coords){
    DEVICE_GUARD(coords);
    const int N = coords.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(N, threads);

    torch::Tensor indices = torch::empty({N}, coords.options());

    AT_DISPATCH_INDEX_TYPES(
        coords.scalar_type(), "morton3D", 
        ([&] {
            kernel_morton3D<index_t><<<blocks, threads>>>(
                N,
                coords.data_ptr<index_t>(),
                indices.data_ptr<index_t>()
            );
        })
    );

    return indices;
}


// __global__ void morton3D_invert_kernel(
//     const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indices,
//     torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> coords
// ){
//     const int n = threadIdx.x + blockIdx.x * blockDim.x;
//     if (n >= coords.size(0)) return;

//     const int ind = indices[n];
//     coords[n][0] = __morton3D_invert(ind >> 0);
//     coords[n][1] = __morton3D_invert(ind >> 1);
//     coords[n][2] = __morton3D_invert(ind >> 2);
// }


// torch::Tensor morton3D_invert_cu(const torch::Tensor indices){
//     int N = indices.size(0);

//     auto coords = torch::zeros({N, 3}, indices.options());

//     const int threads = 256, blocks = (N+threads-1)/threads;

//     AT_DISPATCH_INTEGRAL_TYPES(indices.type(), "morton3D_invert_cu", 
//     ([&] {
//         morton3D_invert_kernel<<<blocks, threads>>>(
//             indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
//             coords.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
//         );
//     }));

//     return coords;
// }
