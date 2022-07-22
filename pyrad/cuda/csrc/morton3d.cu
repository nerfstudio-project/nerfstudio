#include "include/helpers_cuda.h"


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
 * @brief Convert 3D coords to indices in z-order
 * 
 * @param coords Integer coords with shape [N, 3]
 * @return torch::Tensor Integer indices with shape [N]
 */
torch::Tensor morton3D(const torch::Tensor coords){
    DEVICE_GUARD(coords);
    CHECK_INPUT(coords);
    TORCH_CHECK(coords.ndimension() == 2 & coords.size(1) == 3)

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
 * @brief Convert 3D indices to coords in z-order
 * 
 * @param indices Integer indices with shape [N]
 * @return torch::Tensor Integer coords with shape [N, 3]
 */
torch::Tensor morton3D_invert(const torch::Tensor indices){
    DEVICE_GUARD(indices);
    CHECK_INPUT(indices);
    TORCH_CHECK(indices.ndimension() == 1)

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