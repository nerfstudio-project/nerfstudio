/*
Note that this file has the _kernel.cu extension because
setuptools cannot handle files with the same name but different extensions.
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)

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
