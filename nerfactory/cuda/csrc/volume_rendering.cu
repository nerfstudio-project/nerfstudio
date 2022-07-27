#include "include/helpers_cuda.h"


template <typename scalar_t>
__global__ void volumetric_rendering_forward_kernel(
    const uint32_t n_rays,
    const int* packed_info,  // input ray & point indices.
    const scalar_t* starts,  // input start t
    const scalar_t* ends,  // input end t
    const scalar_t* sigmas,  // input density after activation
    const scalar_t* rgbs,  // input rgb after activation 
    // should be all-zero initialized
    scalar_t* accumulated_weight,  // output
    scalar_t* accumulated_depth,  // output
    scalar_t* accumulated_color,  // output
    bool* mask  // output
) {
    CUDA_GET_THREAD_ID(thread_id, n_rays);

    // locate
    const int i = packed_info[thread_id * 3 + 0];  // ray idx in {rays_o, rays_d}
    const int base = packed_info[thread_id * 3 + 1];  // point idx start.
    const int numsteps = packed_info[thread_id * 3 + 2];  // point idx shift.
    if (numsteps == 0) return;

    starts += base;
    ends += base;
    sigmas += base;
    rgbs += base * 3;

    accumulated_weight += i;
    accumulated_depth += i;
    accumulated_color += i * 3;
    mask += i;
    
    // accumulated rendering
    scalar_t T = 1.f;
    scalar_t EPSILON = 1e-4f;
    int j = 0;
    for (; j < numsteps; ++j) {
        if (T < EPSILON) {
            break;
        }
        const scalar_t delta = ends[j] - starts[j];
        const scalar_t t = (ends[j] + starts[j]) * 0.5f;

        const scalar_t alpha = 1.f - __expf(-sigmas[j] * delta);
        const scalar_t weight = alpha * T;
        accumulated_weight[0] += weight;
        accumulated_depth[0] += weight * t;
        accumulated_color[0] += weight * rgbs[j * 3 + 0];
        accumulated_color[1] += weight * rgbs[j * 3 + 1];
        accumulated_color[2] += weight * rgbs[j * 3 + 2];
        T *= (1.f - alpha);
    }
    mask[0] = true;
}


template <typename scalar_t>
__global__ void volumetric_rendering_backward_kernel(
    const uint32_t n_rays,
    const int* packed_info,  // input ray & point indices.
    const scalar_t* starts,  // input start t
    const scalar_t* ends,  // input end t
    const scalar_t* sigmas,  // input density after activation
    const scalar_t* rgbs,  // input rgb after activation 
    const scalar_t* accumulated_weight,  // forward output
    const scalar_t* accumulated_depth,  // forward output
    const scalar_t* accumulated_color,  // forward output
    const scalar_t* grad_weight,  // input
    const scalar_t* grad_depth,  // input
    const scalar_t* grad_color,  // input
    scalar_t* grad_sigmas,  // output
    scalar_t* grad_rgbs  // output
) {
    CUDA_GET_THREAD_ID(thread_id, n_rays);

    // locate
    const int i = packed_info[thread_id * 3 + 0];  // ray idx in {rays_o, rays_d}
    const int base = packed_info[thread_id * 3 + 1];  // point idx start.
    const int numsteps = packed_info[thread_id * 3 + 2];  // point idx shift.
    if (numsteps == 0) return;

    starts += base;
    ends += base;
    sigmas += base;
    rgbs += base * 3;

    grad_sigmas += base;
    grad_rgbs += base * 3;

    accumulated_weight += i;
    accumulated_depth += i;
    accumulated_color += i * 3;
    
    grad_weight += i;
    grad_depth += i;
    grad_color += i * 3;
    
    // backward of accumulated rendering
    scalar_t T = 1.f;
    scalar_t EPSILON = 1e-4f;
    int j = 0;
    scalar_t r = 0, g = 0, b = 0, d = 0;
    for (; j < numsteps; ++j) {
        if (T < EPSILON) {
            break;
        }
        const scalar_t delta = ends[j] - starts[j];
        const scalar_t t = (ends[j] + starts[j]) * 0.5f;

        const scalar_t alpha = 1.f - __expf(-sigmas[j] * delta);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[j * 3 + 0];
        g += weight * rgbs[j * 3 + 1];
        b += weight * rgbs[j * 3 + 2];
        d += weight * t;

        T *= (1.f - alpha);

        grad_rgbs[j * 3 + 0] = grad_color[0] * weight;
        grad_rgbs[j * 3 + 1] = grad_color[1] * weight;
        grad_rgbs[j * 3 + 2] = grad_color[2] * weight;

        grad_sigmas[j] = delta * (
            grad_color[0] * (T * rgbs[j * 3 + 0] - (accumulated_color[0] - r)) +
            grad_color[1] * (T * rgbs[j * 3 + 1] - (accumulated_color[1] - g)) +
            grad_color[2] * (T * rgbs[j * 3 + 2] - (accumulated_color[2] - b)) +
            grad_weight[0] * (1.f - accumulated_weight[0]) +
            grad_depth[0] * (t * T - (accumulated_depth[0] - d))
        );
    }
}

/**
 * @brief Volumetric Rendering: Accumulating samples in the forward pass.
 *  The inputs, excepct for `sigmas` and `rgbs`, are the outputs of our
 *  cuda ray marching function in `raymarching.cu`
 * 
 * @param packed_info Stores how to index the ray samples from the returned values.
 *  Shape of [n_rays, 3]. First value is the ray index. Second value is the sample 
 *  start index in the results for this ray. Third value is the number of samples for
 *  this ray. Note for rays that have zero samples, we simply skip them so the `packed_info`
 *  has some zero padding in the end.
 * @param starts: Where the frustum-shape sample starts along a ray. [total_samples, 1]
 * @param ends: Where the frustum-shape sample ends along a ray. [total_samples, 1]
 * @param sigmas Densities at those samples. [total_samples, 1]
 * @param rgbs RGBs at those samples. [total_samples, 3]
 * @return std::vector<torch::Tensor> 
 * - accumulated_weight: Ray opacity. [n_rays, 1]
 * - accumulated_depth: Ray depth. [n_rays, 1]
 * - accumulated_color: Ray color. [n_rays, 3]
 * - mask: Boolen value store if this ray has valid samples from packed_info. [n_rays]
 */
std::vector<torch::Tensor> volumetric_rendering_forward(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas, 
    torch::Tensor rgbs
) {
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 3);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(sigmas.ndimension() == 2 & sigmas.size(1) == 1);
    TORCH_CHECK(rgbs.ndimension() == 2 & rgbs.size(1) == 3);

    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor accumulated_weight = torch::zeros({n_rays, 1}, sigmas.options()); 
    torch::Tensor accumulated_depth = torch::zeros({n_rays, 1}, sigmas.options()); 
    torch::Tensor accumulated_color = torch::zeros({n_rays, 3}, sigmas.options()); 
    // The rays that are not skipped during sampling.
    torch::Tensor mask = torch::zeros({n_rays}, sigmas.options().dtype(torch::kBool)); 

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sigmas.scalar_type(),
        "volumetric_rendering_forward",
        ([&]
         { volumetric_rendering_forward_kernel<scalar_t><<<blocks, threads>>>(
                n_rays,
                packed_info.data_ptr<int>(), 
                starts.data_ptr<scalar_t>(),
                ends.data_ptr<scalar_t>(),
                sigmas.data_ptr<scalar_t>(),
                rgbs.data_ptr<scalar_t>(),
                accumulated_weight.data_ptr<scalar_t>(),
                accumulated_depth.data_ptr<scalar_t>(),
                accumulated_color.data_ptr<scalar_t>(),
                mask.data_ptr<bool>()
            ); 
        }));

    return {accumulated_weight, accumulated_depth, accumulated_color, mask};
}



/**
 * @brief Volumetric Rendering: Accumulating samples in the backward pass.
 */
std::vector<torch::Tensor> volumetric_rendering_backward(
    torch::Tensor accumulated_weight, 
    torch::Tensor accumulated_depth, 
    torch::Tensor accumulated_color, 
    torch::Tensor grad_weight, 
    torch::Tensor grad_depth, 
    torch::Tensor grad_color, 
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas, 
    torch::Tensor rgbs
) {
    DEVICE_GUARD(packed_info);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor grad_sigmas = torch::zeros(sigmas.sizes(), sigmas.options()); 
    torch::Tensor grad_rgbs = torch::zeros(rgbs.sizes(), rgbs.options()); 

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sigmas.scalar_type(),
        "volumetric_rendering_backward",
        ([&]
         { volumetric_rendering_backward_kernel<scalar_t><<<blocks, threads>>>(
                n_rays,
                packed_info.data_ptr<int>(), 
                starts.data_ptr<scalar_t>(),
                ends.data_ptr<scalar_t>(),
                sigmas.data_ptr<scalar_t>(),
                rgbs.data_ptr<scalar_t>(),
                accumulated_weight.data_ptr<scalar_t>(),
                accumulated_depth.data_ptr<scalar_t>(),
                accumulated_color.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                grad_depth.data_ptr<scalar_t>(),
                grad_color.data_ptr<scalar_t>(),
                grad_sigmas.data_ptr<scalar_t>(),
                grad_rgbs.data_ptr<scalar_t>()
            ); 
        }));

    return {grad_sigmas, grad_rgbs};
}