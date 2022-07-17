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


template <typename index_t>
__global__ void kernel_morton3D_invert(
    const int N,
    const index_t* indices,
    index_t* coords
){
    CUDA_GET_THREAD_ID(thread_id, N);

    // locate
    coords += thread_id * 3;
    indices += thread_id;

    uint32_t index = static_cast<uint32_t>(indices[0]);
    coords[0] = static_cast<index_t>(__morton3D_invert(index >> 0));
    coords[1] = static_cast<index_t>(__morton3D_invert(index >> 1));
    coords[2] = static_cast<index_t>(__morton3D_invert(index >> 2));
    return;
}

/**
 * @brief Convert coords to indices
 * 
 * @param coords Integer coords with shape [N, 3]
 * @return torch::Tensor Integer indices with shape [N]
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

/**
 * @brief Convert indices to coords
 * 
 * @param indices Integer indices with shape [N]
 * @return torch::Tensor Integer coords with shape [N, 3]
 */
torch::Tensor morton3D_invert(const torch::Tensor indices){
    DEVICE_GUARD(indices);
    const int N = indices.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(N, threads);

    torch::Tensor coords = torch::empty({N, 3}, indices.options());

    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "morton3D_invert", 
        ([&] {
            kernel_morton3D_invert<index_t><<<blocks, threads>>>(
                N,
                indices.data_ptr<index_t>(),
                coords.data_ptr<index_t>()
            );
        })
    );

    return coords;
}