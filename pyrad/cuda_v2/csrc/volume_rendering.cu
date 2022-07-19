#include "include/helpers.h"


template <typename scalar_t>
__global__ void volumetric_rendering_kernel(
    const uint32_t n_rays,
    const int* indices,  // input ray & point indices.
    const scalar_t* positions,  // input samples
    const scalar_t* deltas,  // input delta t
    const scalar_t* ts,  // input t
    const scalar_t* sigmas,  // input density after activation
    const scalar_t* rgbs,  // input rgb after activation 
    // should be all-zero initialized
    scalar_t* accumulated_weight,  // output
    scalar_t* accumulated_depth,  // output
    scalar_t* accumulated_color  // output
) {
    CUDA_GET_THREAD_ID(thread_id, n_rays);

    // locate
    const int i = indices[thread_id * 3 + 0];  // ray idx in {rays_o, rays_d}
    const int base = indices[thread_id * 3 + 1];  // point idx start.
    const int numsteps = indices[thread_id * 3 + 2];  // point idx shift.
    if (numsteps == 0) return;

    positions += base * 3;
    deltas += base;
    ts += base;
    sigmas += base;
    rgbs += base * 3;

    accumulated_weight += i;
    accumulated_depth += i;
    accumulated_color += i * 3;
    
    // accumulated rendering
    scalar_t T = 1.f;
	scalar_t EPSILON = 1e-4f;
	int j = 0;
    for (; j < numsteps; ++j) {
		if (T < EPSILON) {
			break;
		}
		const scalar_t alpha = 1.f - __expf(-sigmas[j] * deltas[j]);
		const scalar_t weight = alpha * T;
		accumulated_weight[0] += weight;
        accumulated_depth[0] += weight * ts[j];
        accumulated_color[0] += weight * rgbs[j * 3 + 0];
        accumulated_color[1] += weight * rgbs[j * 3 + 1];
        accumulated_color[2] += weight * rgbs[j * 3 + 2];
		T *= (1.f - alpha);
	}
}


template <typename scalar_t>
__global__ void volumetric_rendering_backward_kernel(
    const uint32_t n_rays,
    const int* indices,  // input ray & point indices.
    const scalar_t* deltas,  // input delta t
    const scalar_t* ts,  // input t
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
    const int i = indices[thread_id * 3 + 0];  // ray idx in {rays_o, rays_d}
    const int base = indices[thread_id * 3 + 1];  // point idx start.
    const int numsteps = indices[thread_id * 3 + 2];  // point idx shift.
    if (numsteps == 0) return;

    deltas += base;
    ts += base;
    sigmas += base;
    rgbs += base * 3;

    grad_sigmas += base;
    grad_rgbs += base * 3;

    accumulated_weight += i;
    accumulated_depth += i;
    accumulated_color += i * 3;
    
    grad_weight += i;
    grad_color += i;
    grad_color += i * 3;
    
    // backward of accumulated rendering
    scalar_t T = 1.f;
	scalar_t EPSILON = 1e-4f;
	int j = 0;
    scalar_t r = 0, g = 0, b = 0, ws = 0, d = 0;
    const scalar_t r_accum = accumulated_color[0];
    const scalar_t g_accum = accumulated_color[1];
    const scalar_t b_accum = accumulated_color[2];
    for (; j < numsteps; ++j) {
		if (T < EPSILON) {
			break;
		}

		const scalar_t alpha = 1.f - __expf(-sigmas[j] * deltas[j]);
		const scalar_t weight = alpha * T;

        r += weight * rgbs[j * 3 + 0];
        g += weight * rgbs[j * 3 + 1];
        b += weight * rgbs[j * 3 + 2];
        d += weight * ts[j];
        ws += weight;

		T *= (1.f - alpha);

        grad_rgbs[j * 3 + 0] = grad_color[0] * weight;
        grad_rgbs[j * 3 + 1] = grad_color[1] * weight;
        grad_rgbs[j * 3 + 2] = grad_color[2] * weight;

        grad_sigmas[j] = deltas[j] * (
            grad_color[0] * (T * rgbs[j * 3 + 0] - (r_accum - r)) +
            grad_color[1] * (T * rgbs[j * 3 + 1] - (g_accum - g)) +
            grad_color[2] * (T * rgbs[j * 3 + 2] - (b_accum - b)) +
            grad_weight[0] * (1.f - accumulated_weight[0]) +
            grad_depth[0] * (ts[j] * T - (accumulated_depth[0] - d))
        );
	}
}


std::vector<torch::Tensor> volumetric_rendering(
    torch::Tensor indices, 
    torch::Tensor positions, 
    torch::Tensor deltas, 
    torch::Tensor ts, 
    torch::Tensor sigmas, 
    torch::Tensor rgbs
) {
    DEVICE_GUARD(indices);
    CHECK_INPUT(indices);
    CHECK_INPUT(positions);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    
    const uint32_t n_rays = indices.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor accumulated_weight = torch::zeros({n_rays, 1}, sigmas.options()); 
    torch::Tensor accumulated_depth = torch::zeros({n_rays, 1}, sigmas.options()); 
    torch::Tensor accumulated_color = torch::zeros({n_rays, 3}, sigmas.options()); 

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sigmas.scalar_type(),
        "volumetric_rendering",
        ([&]
         { volumetric_rendering_kernel<scalar_t><<<blocks, threads>>>(
                n_rays,
                indices.data_ptr<int>(), 
                positions.data_ptr<scalar_t>(),
                deltas.data_ptr<scalar_t>(),
                ts.data_ptr<scalar_t>(),
                sigmas.data_ptr<scalar_t>(),
                rgbs.data_ptr<scalar_t>(),
                accumulated_weight.data_ptr<scalar_t>(),
                accumulated_depth.data_ptr<scalar_t>(),
                accumulated_color.data_ptr<scalar_t>()
            ); 
        }));

    return {accumulated_weight, accumulated_depth, accumulated_color};
}


std::vector<torch::Tensor> volumetric_rendering_backward(
    torch::Tensor accumulated_weight, 
    torch::Tensor accumulated_depth, 
    torch::Tensor accumulated_color, 
    torch::Tensor grad_weight, 
    torch::Tensor grad_depth, 
    torch::Tensor grad_color, 
    torch::Tensor indices, 
    torch::Tensor deltas, 
    torch::Tensor ts, 
    torch::Tensor sigmas, 
    torch::Tensor rgbs
) {
    DEVICE_GUARD(indices);
    const uint32_t n_rays = indices.size(0);

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
                indices.data_ptr<int>(), 
                deltas.data_ptr<scalar_t>(),
                ts.data_ptr<scalar_t>(),
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