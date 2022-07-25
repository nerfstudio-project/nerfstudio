#include "include/helpers_cuda.h"


template <typename scalar_t>
inline __host__ __device__ void _ray_aabb_intersect(
    const scalar_t* rays_o,
    const scalar_t* rays_d,
    const scalar_t* aabb,
    scalar_t* near,
    scalar_t* far
) {
    // aabb is [xmin, ymin, zmin, xmax, ymax, zmax]
    scalar_t tmin = (aabb[0] - rays_o[0]) / rays_d[0];
    scalar_t tmax = (aabb[3] - rays_o[0]) / rays_d[0];
    if (tmin > tmax) __swap(tmin, tmax);

    scalar_t tymin = (aabb[1] - rays_o[1]) / rays_d[1];
    scalar_t tymax = (aabb[4] - rays_o[1]) / rays_d[1];
    if (tymin > tymax) __swap(tymin, tymax);

    if (tmin > tymax || tymin > tmax){
        *near = std::numeric_limits<scalar_t>::max();
        *far = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    scalar_t tzmin = (aabb[2] - rays_o[2]) / rays_d[2];
    scalar_t tzmax = (aabb[5] - rays_o[2]) / rays_d[2];
    if (tzmin > tzmax) __swap(tzmin, tzmax);

    if (tmin > tzmax || tzmin > tmax){
        *near = std::numeric_limits<scalar_t>::max();
        *far = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    *near = tmin;
    *far = tmax;
    return;
}


template <typename scalar_t>
__global__ void kernel_ray_aabb_intersect(
    const int N,
    const scalar_t* rays_o,
    const scalar_t* rays_d,
    const scalar_t* aabb,
    scalar_t* t_min,
    scalar_t* t_max
){
    // aabb is [xmin, ymin, zmin, xmax, ymax, zmax]
    CUDA_GET_THREAD_ID(thread_id, N);

    // locate
    rays_o += thread_id * 3;
    rays_d += thread_id * 3;
    t_min += thread_id;
    t_max += thread_id;
    
    _ray_aabb_intersect<scalar_t>(rays_o, rays_d, aabb, t_min, t_max);
    
    scalar_t zero = static_cast<scalar_t>(0.f);
    *t_min = *t_min > zero ? *t_min : zero;
    return;
}

/**
 * @brief Ray AABB Test
 * 
 * @param rays_o Ray origins. Tensor with shape [N, 3].
 * @param rays_d Normalized ray directions. Tensor with shape [N, 3].
 * @param aabb Scene AABB [xmin, ymin, zmin, xmax, ymax, zmax]. Tensor with shape [6].
 * @return std::vector<torch::Tensor> 
 *  Ray AABB intersection {t_min, t_max} with shape [N] respectively. Note the t_min is 
 *  clipped to minimum zero. 
 */
std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o, const torch::Tensor rays_d, const torch::Tensor aabb
) {
    DEVICE_GUARD(rays_o);
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(aabb);
    TORCH_CHECK(rays_o.ndimension() == 2 & rays_o.size(1) == 3)
    TORCH_CHECK(rays_d.ndimension() == 2 & rays_d.size(1) == 3)
    TORCH_CHECK(aabb.ndimension() == 1 & aabb.size(0) == 6)
    
    const int N = rays_o.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(N, threads);

    torch::Tensor t_min = torch::empty({N}, rays_o.options());
    torch::Tensor t_max = torch::empty({N}, rays_o.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        rays_o.scalar_type(), "ray_aabb_intersect", 
        ([&] {
            kernel_ray_aabb_intersect<scalar_t><<<blocks, threads>>>(
                N,
                rays_o.data_ptr<scalar_t>(),
                rays_d.data_ptr<scalar_t>(),
                aabb.data_ptr<scalar_t>(),
                t_min.data_ptr<scalar_t>(),
                t_max.data_ptr<scalar_t>()
            );
        })
    );

    return {t_min, t_max};
}