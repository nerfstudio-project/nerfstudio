#include "utils.h"


template <typename scalar_t>
__global__ void composite_train_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    if (N_samples==0 || start_idx+N_samples>=sigmas.size(0)) return; // no hit

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        rgb[ray_idx][0] += w*rgbs[s][0];
        rgb[ray_idx][1] += w*rgbs[s][1];
        rgb[ray_idx][2] += w*rgbs[s][2];
        depth[ray_idx] += w*ts[s];
        opacity[ray_idx] += w;
        T *= 1.0f-a;

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}


std::vector<torch::Tensor> composite_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float T_threshold
){
    const int N_rays = rays_a.size(0);

    auto opacity = torch::zeros({N_rays}, sigmas.options());
    auto depth = torch::zeros({N_rays}, sigmas.options());
    auto rgb = torch::zeros({N_rays, 3}, sigmas.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_fw_cu", 
    ([&] {
        composite_train_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {opacity, depth, rgb};
}


template <typename scalar_t>
__global__ void composite_train_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dopacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_ddepth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgb,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigmas,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgbs
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    if (N_samples==0 || start_idx+N_samples>=sigmas.size(0)) return; // no hit
        
    // front to back compositing
    int samples = 0;
    scalar_t R = rgb[ray_idx][0], G = rgb[ray_idx][1], B = rgb[ray_idx][2];
    scalar_t O = opacity[ray_idx], D = depth[ray_idx];
    scalar_t T = 1.0f, r = 0.0f, g = 0.0f, b = 0.0f, op = 0.0f, t = 0.0f, d = 0.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        r += w*rgbs[s][0]; g += w*rgbs[s][1]; b += w*rgbs[s][2];
        d += w*ts[s];
        op += w;
        T *= 1.0f-a;

        // compute gradients by math...
        dL_drgbs[s][0] = dL_drgb[ray_idx][0]*w;
        dL_drgbs[s][1] = dL_drgb[ray_idx][1]*w;
        dL_drgbs[s][2] = dL_drgb[ray_idx][2]*w;

        dL_dsigmas[s] = deltas[s] * (
            dL_drgb[ray_idx][0]*(rgbs[s][0]*T-(R-r)) + 
            dL_drgb[ray_idx][1]*(rgbs[s][1]*T-(G-g)) + 
            dL_drgb[ray_idx][2]*(rgbs[s][2]*T-(B-b)) + 
            dL_dopacity[ray_idx]*(1-O) + 
            dL_ddepth[ray_idx]*(t*T-(D-d))  // TODO: bug?
        );

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}


std::vector<torch::Tensor> composite_train_bw_cu(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddepth,
    const torch::Tensor dL_drgb,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    const torch::Tensor rgb,
    // const torch::Tensor rgb_bg,
    const float T_threshold
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    auto dL_drgbs = torch::zeros({N, 3}, sigmas.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_bw_cu", 
    ([&] {
        composite_train_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dopacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_ddepth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            T_threshold,
            dL_dsigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {dL_dsigmas, dL_drgbs};
}


template <typename scalar_t>
__global__ void composite_test_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> hits_t,
    torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> alive_indices,
    const scalar_t T_threshold,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> N_eff_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0)) return;

    if (N_eff_samples[n]==0){ // no hit
        alive_indices[n] = -1;
        return;
    }

    const size_t r = alive_indices[n]; // ray index

    // front to back compositing
    int s = 0; scalar_t T = 1-opacity[r];

    while (s < N_eff_samples[n]) {
        const scalar_t a = 1.0f - __expf(-sigmas[n][s]*deltas[n][s]);
        const scalar_t w = a * T;

        rgb[r][0] += w*rgbs[n][s][0];
        rgb[r][1] += w*rgbs[n][s][1];
        rgb[r][2] += w*rgbs[n][s][2];
        depth[r] += w*ts[n][s];
        opacity[r] += w;
        T *= 1.0f-a;

        if (T <= T_threshold){ // ray has enough opacity
            alive_indices[n] = -1;
            break;
        }
        s++;
    }
}


void composite_test_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor rgb
){
    const int N_rays = alive_indices.size(0);

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_test_fw_cu", 
    ([&] {
        composite_test_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            hits_t.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            alive_indices.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
            T_threshold,
            N_eff_samples.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));
}