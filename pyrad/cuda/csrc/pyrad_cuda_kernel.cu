/*
Note that this file has the _kernel.cu extension because
setuptools cannot handle files with the same name but different extensions.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "include/structures.cuh"
#include "include/functions.cuh"

namespace
{
    template <typename scalar_t>
    __device__ __forceinline__ scalar_t sigmoid(scalar_t z)
    {
        return 1.0 / (1.0 + exp(-z));
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z)
    {
        const auto s = sigmoid(z);
        return (1.0 - s) * s;
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_tanh(scalar_t z)
    {
        const auto t = tanh(z);
        return 1 - (t * t);
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0)
    {
        return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0)
    {
        const auto e = exp(z);
        const auto d_relu = z < 0.0 ? 0.0 : 1.0;
        return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
    }

    template <typename scalar_t>
    __global__ void sample_uniformly_along_ray_bundle_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> origins,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> directions,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> nears,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> fars,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> offsets,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_time_steps,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_samples,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output_time_steps_mask)
    {

        const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        const int max_num_samples = output_samples.size(1);

        scalar_t t_min = nears[thread_id];
        scalar_t t_max = fars[thread_id];
        scalar_t offset = offsets[thread_id];
        int sample_index = 0;

        // #pragma unroll
        for (scalar_t t = t_min; t <= t_max; t += offset)
        {
            // #pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                if (sample_index < max_num_samples)
                {
                    scalar_t pos_i = origins[thread_id][i] + t * directions[thread_id][i];
                    output_samples[thread_id][sample_index][i] = pos_i;
                }
            }
            if (sample_index < max_num_samples)
            {
                output_time_steps[thread_id][sample_index] = t;
                output_time_steps_mask[thread_id][sample_index] = 1.0;
            }
            sample_index += 1;
        }
    }
} // namespace

std::vector<torch::Tensor> sample_uniformly_along_ray_bundle(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor nears,
    torch::Tensor fars,
    torch::Tensor offsets,
    int max_num_samples)
{
    std::cout << "sample_uniformly_along_ray_bundle -- start" << std::endl;

    TORCH_CHECK(origins.is_floating_point()); // should be float32
    TORCH_CHECK(origins.ndimension() == 2);   // and check that shape is (num_rays, 3)
    TORCH_CHECK(origins.size(1) == 3);
    const auto num_rays = origins.size(0);

    // TODO: change the number of threads based on GPU
    const int threads = 1024; // number of threads in a thread block
    // const int blocks = num_rays / threads;
    const int blocks = CUDA_N_BLOCKS_NEEDED(num_rays, threads);

    // std::cout << origins.options() << std::endl;
    torch::Tensor output_time_steps = torch::zeros({num_rays, max_num_samples}, origins.options());
    torch::Tensor output_samples = torch::zeros({num_rays, max_num_samples, 3}, origins.options());
    torch::Tensor output_time_steps_mask = torch::zeros({num_rays, max_num_samples}, origins.options());

    // The second argument is used for error messages.
    // __FUNCTION__ macro == "sample_uniformly_along_ray_bundle" in this case.
    AT_DISPATCH_FLOATING_TYPES(origins.type(), __FUNCTION__, [&]
                               { sample_uniformly_along_ray_bundle_kernel<scalar_t><<<blocks, threads>>>(
                                     origins.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                     directions.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                     nears.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                     fars.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                     offsets.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                     output_time_steps.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                     output_samples.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                     output_time_steps_mask.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()); });

    std::cout << "sample_uniformly_along_ray_bundle -- end" << std::endl;
    return {output_time_steps, output_samples, output_time_steps_mask};
}

template <typename scalar_t>
__global__ void kernel_generate_ray_samples_uniform(
    const int num_rays,
    const int num_samples,
    const scalar_t* ray_origins, 
    const scalar_t* ray_directions, 
    const scalar_t* ray_pixel_area,
    const int* ray_camera_indices,
    const scalar_t* ray_nears, 
    const scalar_t* ray_fars, 
    const bool* ray_valid_mask,
    const int grid_num_cascades,
    const int grid_resolution,
    const scalar_t* grid_aabb, 
    const scalar_t* grid_data,
    // outputs
    scalar_t* frustum_origins,
    scalar_t* frustum_directions,
    scalar_t* frustum_starts,
    scalar_t* frustum_ends,
    scalar_t* frustum_pixel_area,
    int* packed_indices,
    int* camera_indices,
    scalar_t* deltas
) {
    CUDA_GET_THREAD_ID(thread_id, num_rays);

    // locate
    const scalar_t* ray_o = ray_origins + thread_id * 3;
    const scalar_t* ray_d = ray_directions + thread_id * 3;
    const scalar_t area = (ray_pixel_area + thread_id)[0];
    const scalar_t startt = (ray_nears + thread_id)[0];
    const scalar_t endt = (ray_fars + thread_id)[0];

    int camera_id;
    if (ray_camera_indices) {
        camera_id = (ray_camera_indices + thread_id)[0];
    }
    
    // skip invalid ray
    if (ray_valid_mask) {
        const bool valid = (ray_valid_mask + thread_id)[0];
        if (!valid) {
            return;
        }
    }

    // step size for this ray
    scalar_t dt = (endt - startt) / num_samples;

    scalar_t t = startt;
    int j = 0;
    while (t < endt && j < num_samples) {
        // current point
        const float x = ray_o[0] + t * ray_d[0];
        const float y = ray_o[1] + t * ray_d[1];
        const float z = ray_o[2] + t * ray_d[2];

        scalar_t density[1];
        grid_sample_3d<scalar_t>(
            x, y, z, 
            grid_data, 
            grid_resolution, 
            grid_resolution, 
            grid_resolution, 
            grid_num_cascades,
            density
        );
        // weight = 

        // if (density > 0.0001f) {

        // }

    //         positions_out[j * 3 + 0] = x;
    //         positions_out[j * 3 + 1] = y;
    //         positions_out[j * 3 + 2] = z;
    //         dirs_out[j * 3 + 0] = dx,
    //         dirs_out[j * 3 + 1] = dy,
    //         dirs_out[j * 3 + 2] = dz,
    //         deltas_out[j + 0] = dt;   
    //         ts_out[j + 0] = t;         
    //         // coords_out(j)->set_with_optional_extra_dims(
    //         //     warp_position(x, y, z, aabb), 
    //         //     warp_direction(dx, dy, dz), 
    //         //     warp_dt(dt), 
    //         //     extra_dims, 
    //         //     coords_out.stride_in_bytes
    //         // );
    //         ++j;
    //         t += dt;
    //     }
    //     else {
    //     }
    }

    return;
}


RaySamples generate_ray_samples_uniform(
    RayBundle& ray_bundle, int num_samples, DensityGrid& grid
) {
    DEVICE_GUARD(ray_bundle.origins);
    const int num_rays = ray_bundle.origins.size(0);
    const int max_samples = num_rays * num_samples;

    const int cuda_n_threads = std::min<int>(num_rays, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(num_rays, cuda_n_threads);

    // create outputs
    auto options = ray_bundle.origins.options();
    Frustums frustums = {
        torch::empty({max_samples, 3}, options),  // origins
        torch::empty({max_samples, 3}, options),  // directions
        torch::empty({max_samples, 1}, options),  // starts
        torch::empty({max_samples, 1}, options),  // ends
        torch::empty({max_samples, 1}, options)  // pixel_area   
    };
    RaySamples ray_samples = {
        frustums,
        torch::empty({num_rays, 3}, options.dtype(torch::kInt32)),  // packed_indices
        torch::empty({max_samples, 1}, options.dtype(torch::kInt32)),  // camera_indices
        torch::empty({max_samples, 1}, options)  // deltas
    };

    AT_DISPATCH_FLOATING_TYPES(
        ray_bundle.origins.scalar_type(),
        "generate_ray_samples_uniform",
        ([&]
         { kernel_generate_ray_samples_uniform<<<blocks, cuda_n_threads>>>(
                // inputs
                num_rays,
                num_samples,
                ray_bundle.origins.data_ptr<scalar_t>(), 
                ray_bundle.directions.data_ptr<scalar_t>(), 
                ray_bundle.pixel_area.data_ptr<scalar_t>(),
                ray_bundle.camera_indices.defined() ? 
                    ray_bundle.camera_indices.data_ptr<int>() : nullptr,
                ray_bundle.nears.data_ptr<scalar_t>(), 
                ray_bundle.fars.data_ptr<scalar_t>(), 
                ray_bundle.valid_mask.defined() ? 
                    ray_bundle.valid_mask.data_ptr<bool>() : nullptr, 
                grid.num_cascades,
                grid.resolution,
                grid.aabb.data_ptr<scalar_t>(), 
                grid.data.data_ptr<scalar_t>(),
                // outputs
                ray_samples.frustums.origins.data_ptr<scalar_t>(),
                ray_samples.frustums.directions.data_ptr<scalar_t>(),
                ray_samples.frustums.starts.data_ptr<scalar_t>(),
                ray_samples.frustums.ends.data_ptr<scalar_t>(),
                ray_samples.frustums.pixel_area.data_ptr<scalar_t>(),
                ray_samples.packed_indices.data_ptr<int>(),
                ray_samples.camera_indices.data_ptr<int>(),
                ray_samples.deltas.data_ptr<scalar_t>()
            ); 
        }));

    return ray_samples;
}

template <typename scalar_t>
__global__ void kernel_grid_sample(
	const int num_samples,
    const int grid_num_cascades,
    const int grid_resolution,
    const scalar_t* grid_data,  // [x, y, z, c]
    const scalar_t* positions,  // [num_samples, 3] valued in [0, 1]
    scalar_t* out  // [num_samples, c]
) {
    // CUDA_GET_THREAD_ID(thread_id, num_samples);

    CUDA_KERNEL_LOOP_TYPE(thread_id, num_samples, int) {
        grid_sample_3d<scalar_t>(
            positions[thread_id * 3 + 0],
            positions[thread_id * 3 + 1],
            positions[thread_id * 3 + 2],
            grid_data,
            grid_resolution, 
            grid_resolution, 
            grid_resolution, 
            grid_num_cascades,
            out + thread_id * grid_num_cascades
        );
    }
    return;
}


torch::Tensor grid_sample(
    torch::Tensor positions, DensityGrid& grid
) {
    DEVICE_GUARD(positions);
    grid.check();

    const int num_samples = positions.size(0);

    const int cuda_n_threads = std::min<int>(num_samples, CUDA_MAX_THREADS);
    const int blocks = CUDA_N_BLOCKS_NEEDED(num_samples, cuda_n_threads);

    // create outputs
    auto options = grid.data.options();
    torch::Tensor out = torch::empty({num_samples, grid.num_cascades}, options);

    AT_DISPATCH_FLOATING_TYPES(
        grid.data.scalar_type(),
        "grid_sample",
        ([&]
         { kernel_grid_sample<scalar_t><<<blocks, cuda_n_threads>>>(
                num_samples,
                grid.num_cascades,
                grid.resolution,
                grid.data.data_ptr<scalar_t>(), 
                positions.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>()
            ); 
        }));

    return out;
}
