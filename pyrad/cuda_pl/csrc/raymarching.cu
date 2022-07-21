#include "helper_math.h"
#include "utils.h"

#define SQRT3 1.73205080757f


inline __host__ __device__ float signf(const float x) { return copysignf(1.0f, x); }

// exponentially step t if f>0 (larger step size when sample moves away from the camera)
// default exp_step_factor is 0 for blender, 1/256 for real scene
inline __host__ __device__ float calc_dt(
    float t, float exp_step_factor, float scale,
    int max_samples, int grid_size, int cascades){
    return clamp(t*exp_step_factor,
                 SQRT3*2*scale/max_samples,
                 SQRT3*2*(1<<(cascades-1))/grid_size);
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const int cascades) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent; frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, ...
    return fminf(cascades-1, fmaxf(0, exponent+1));
}

inline __device__ int mip_from_dt(float dt, int grid_size, int cascades) {
    int exponent; frexpf(dt*2*grid_size, &exponent);
    return fminf(cascades-1, fmaxf(0, exponent));
}

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


__global__ void morton3D_kernel(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indices
){
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= coords.size(0)) return;

    indices[n] = __morton3D(coords[n][0], coords[n][1], coords[n][2]);
}


torch::Tensor morton3D_cu(const torch::Tensor coords){
    int N = coords.size(0);

    auto indices = torch::zeros({N}, coords.options());

    const int threads = 256, blocks = (N+threads-1)/threads;

    AT_DISPATCH_INTEGRAL_TYPES(coords.type(), "morton3D_cu", 
    ([&] {
        morton3D_kernel<<<blocks, threads>>>(
            coords.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>()
        );
    }));

    return indices;
}


__global__ void morton3D_invert_kernel(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> coords
){
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= coords.size(0)) return;

    const int ind = indices[n];
    coords[n][0] = __morton3D_invert(ind >> 0);
    coords[n][1] = __morton3D_invert(ind >> 1);
    coords[n][2] = __morton3D_invert(ind >> 2);
}


torch::Tensor morton3D_invert_cu(const torch::Tensor indices){
    int N = indices.size(0);

    auto coords = torch::zeros({N, 3}, indices.options());

    const int threads = 256, blocks = (N+threads-1)/threads;

    AT_DISPATCH_INTEGRAL_TYPES(indices.type(), "morton3D_invert_cu", 
    ([&] {
        morton3D_invert_kernel<<<blocks, threads>>>(
            indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            coords.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
        );
    }));

    return coords;
}


// below code is based on https://github.com/ashawkey/torch-ngp/blob/main/raymarching/src/raymarching.cu
__global__ void raymarching_train_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> hits_t,
    const uint8_t* __restrict__ density_bitfield,
    const int cascades,
    const int grid_size,
    const float scale,
    const float exp_step_factor,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> noise,
    const int max_samples,
    int* __restrict__ counter,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> xyzs,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> dirs,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> deltas,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> ts
){
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rays_o.size(0)) return;

    const int grid_size3 = grid_size*grid_size*grid_size;

    const float ox = rays_o[r][0], oy = rays_o[r][1], oz = rays_o[r][2];
    const float dx = rays_d[r][0], dy = rays_d[r][1], dz = rays_d[r][2];
    const float dx_inv = 1.0f/dx, dy_inv = 1.0f/dy, dz_inv = 1.0f/dz;
    float t1 = hits_t[r][0], t2 = hits_t[r][1];

    if (t1>=0) { // only perturb the starting t
        const float dt = 
            calc_dt(t1, exp_step_factor, scale, max_samples, grid_size, cascades);
        t1 += dt*noise[r];
    }

    // first pass: compute the number of samples on the ray
    float t = t1; int N_samples = 0;

    // if t1 < 0 (no hit) this loop will be skipped (N_samples will be 0)
    while (0<=t && t<t2 && N_samples<max_samples){
        const float x = ox+t*dx, y = oy+t*dy, z = oz+t*dz;

        const float dt = 
            calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
        const int mip = max(mip_from_pos(x, y, z, cascades),
                            mip_from_dt(dt, grid_size, cascades));

        // NOTE(ruilongli): I think this is a bug here.
        // the original code is `scalbnf`. Seems like bound?
        const float mip_bound = fminf(1<<mip, scale);
        const float mip_bound_inv = 0.5f/mip_bound;

        // round down to nearest grid position
        const int nx = clamp((x*mip_bound_inv+0.5f)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp((y*mip_bound_inv+0.5f)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp((z*mip_bound_inv+0.5f)*grid_size, 0.0f, grid_size-1.0f);

        // printf("[input] x %f, y %f, z %f, mip %d, grid_size %d, center %f\n", x, y, z, mip, grid_size, .0f);
        // printf("[output] mip_scale %f, ix %d iy %d, iz %d\n", mip_bound_inv, nx, ny, nz);

        // printf("mip %d, mip_scale %f, ix %d iy %d\n", mip, mip_bound_inv, nx, ny);
        // printf("mip pos %d, mip dt %d, cascades %d\n", mip_from_pos(x, y, z, cascades), mip_from_dt(dt, grid_size, cascades), cascades);
        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));
        // printf("%f %f %d %d\n", t, dt, occ, idx);

        // if (occ) {
        if (true) {
            t += dt; N_samples++;
        } else { // skip until the next voxel
            // calculate the distance to the next voxel
            const int res = grid_size>>mip;
            const float px = x*res, py = y*res, pz = z*res;
            const float tx = (floorf(px+0.5f*(1+signf(dx)))-px)*dx_inv;
            const float ty = (floorf(py+0.5f*(1+signf(dy)))-py)*dy_inv;
            const float tz = (floorf(pz+0.5f*(1+signf(dz)))-pz)*dz_inv;

            const float t_target = t+fmaxf(0.0f, fminf(tx, fminf(ty, tz))/res); // the t of the next voxel
            do {
                t += calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
            } while (t < t_target);
        }
    }

    // second pass: write to output
    int start_idx = atomicAdd(counter, N_samples);
    int ray_count = atomicAdd(counter+1, 1);

    rays_a[ray_count][0] = r;
    rays_a[ray_count][1] = start_idx; rays_a[ray_count][2] = N_samples;

    if (N_samples==0 || start_idx+N_samples>xyzs.size(0)) return;

    t = t1; int samples = 0;

    while (t<t2 && samples<N_samples){
        const float x = ox+t*dx, y = oy+t*dy, z = oz+t*dz;

        const float dt = 
            calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
        const int mip = max(mip_from_pos(x, y, z, cascades),
                            mip_from_dt(dt, grid_size, cascades));

        const float mip_bound = fminf(1<<mip, scale);
        const float mip_bound_inv = 1.0f/mip_bound;

        // round down to nearest grid position
        const int nx = clamp(0.5f*(x*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp(0.5f*(y*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp(0.5f*(z*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);

        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));

        // if (occ) {
        if (true) {
            const int s = start_idx + samples;
            xyzs[s][0] = x; xyzs[s][1] = y; xyzs[s][2] = z;
            dirs[s][0] = dx; dirs[s][1] = dy; dirs[s][2] = dz;
            ts[s] = t;
            deltas[s] = dt; // interval for volume rendering integral
            t += dt; samples++;
        } else { // skip until the next voxel
            // calculate the distance to the next voxel
            const int res = grid_size>>mip;
            const float px = x*res, py = y*res, pz = z*res;
            const float tx = (floorf(px+0.5f*(1+signf(dx)))-px)*dx_inv;
            const float ty = (floorf(py+0.5f*(1+signf(dy)))-py)*dy_inv;
            const float tz = (floorf(pz+0.5f*(1+signf(dz)))-pz)*dz_inv;

            const float t_target = t+fmaxf(0.0f, fminf(tx, fminf(ty, tz))/res); // the t of the next voxel
            do {
                t += calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
            } while (t < t_target);
        }
    }
}


std::vector<torch::Tensor> raymarching_train_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const torch::Tensor noise,
    const int grid_size,
    const int max_samples
){
    const int N_rays = rays_o.size(0);
    const int cascades = density_bitfield.size(0);

    // count the number of samples and the number of rays processed
    auto counter = torch::zeros({2}, torch::dtype(torch::kInt32).device(rays_o.device()));
    // ray attributes: ray_idx, start_idx, N_samples
    auto rays_a = torch::zeros({N_rays, 3},
                        torch::dtype(torch::kInt32).device(rays_o.device()));
    auto xyzs = torch::zeros({N_rays*max_samples, 3}, rays_o.options());
    auto dirs = torch::zeros({N_rays*max_samples, 3}, rays_o.options());
    auto deltas = torch::zeros({N_rays*max_samples}, rays_o.options());
    auto ts = torch::zeros({N_rays*max_samples}, rays_o.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.type(), "raymarching_train_cu", 
    ([&] {
        raymarching_train_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            hits_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            density_bitfield.data_ptr<uint8_t>(),
            cascades,
            grid_size,
            scale,
            exp_step_factor,
            // perturb,
            noise.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            max_samples,
            // rng,
            counter.data_ptr<int>(),
            rays_a.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            xyzs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            deltas.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            ts.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
        );
    }));

    return {rays_a, xyzs, dirs, deltas, ts, counter};
}


__global__ void raymarching_test_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> hits_t,
    const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> alive_indices,
    const uint8_t* __restrict__ density_bitfield,
    const int cascades,
    const int grid_size,
    const float scale,
    const float exp_step_factor,
    const int N_samples,
    const int max_samples,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> xyzs,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> dirs,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> deltas,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ts,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> N_eff_samples
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0)) return;

    const size_t r = alive_indices[n]; // ray index
    const int grid_size3 = grid_size*grid_size*grid_size;

    const float ox = rays_o[r][0], oy = rays_o[r][1], oz = rays_o[r][2];
    const float dx = rays_d[r][0], dy = rays_d[r][1], dz = rays_d[r][2];
    const float dx_inv = 1.0f/dx, dy_inv = 1.0f/dy, dz_inv = 1.0f/dz;

    float t = hits_t[r][0], t2 = hits_t[r][1];
    int s = 0;

    while (t<t2 && s<N_samples){
        const float x = ox+t*dx, y = oy+t*dy, z = oz+t*dz;

        const float dt = 
            calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
        const int mip = max(mip_from_pos(x, y, z, cascades),
                            mip_from_dt(dt, grid_size, cascades));

        const float mip_bound = fminf(1<<mip, scale);
        const float mip_bound_inv = 1.0f/mip_bound;

        // round down to nearest grid position
        const int nx = clamp(0.5f*(x*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp(0.5f*(y*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp(0.5f*(z*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);

        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));

        if (occ) {
            xyzs[n][s][0] = x; xyzs[n][s][1] = y; xyzs[n][s][2] = z;
            dirs[n][s][0] = dx; dirs[n][s][1] = dy; dirs[n][s][2] = dz;
            ts[n][s] = t;
            deltas[n][s] = dt; // interval for volume rendering integral
            t += dt;
            hits_t[r][0] = t; // modify the starting point for the next marching
            s++;
        } else { // skip until the next voxel
            // calculate the distance to the next voxel
            const int res = grid_size>>mip;
            const float px = x*res, py = y*res, pz = z*res;
            const float tx = (floorf(px+0.5f*(1+signf(dx)))-px)*dx_inv;
            const float ty = (floorf(py+0.5f*(1+signf(dy)))-py)*dy_inv;
            const float tz = (floorf(pz+0.5f*(1+signf(dz)))-pz)*dz_inv;

            const float t_target = t+fmaxf(0.0f, fminf(tx, fminf(ty, tz))/res); // the t of the next voxel
            do {
                t += calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
            } while (t < t_target);
        }
    }
    N_eff_samples[n] = s; // effective samples that hit occupied region (<=N_samples)
}


std::vector<torch::Tensor> raymarching_test_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const int grid_size,
    const int max_samples,
    const int N_samples
){
    const int N_rays = alive_indices.size(0);
    const int cascades = density_bitfield.size(0);

    auto xyzs = torch::zeros({N_rays, N_samples, 3}, rays_o.options());
    auto dirs = torch::zeros({N_rays, N_samples, 3}, rays_o.options());
    auto deltas = torch::zeros({N_rays, N_samples}, rays_o.options());
    auto ts = torch::zeros({N_rays, N_samples}, rays_o.options());
    auto N_eff_samples = torch::zeros({N_rays},
                            torch::dtype(torch::kInt32).device(rays_o.device()));

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.type(), "raymarching_test_cu", 
    ([&] {
        raymarching_test_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            hits_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            alive_indices.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
            density_bitfield.data_ptr<uint8_t>(),
            cascades,
            grid_size,
            scale,
            exp_step_factor,
            N_samples,
            max_samples,
            xyzs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dirs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            deltas.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ts.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            N_eff_samples.packed_accessor32<int, 1, torch::RestrictPtrTraits>()
        );
    }));

    return {xyzs, dirs, deltas, ts, N_eff_samples};
}