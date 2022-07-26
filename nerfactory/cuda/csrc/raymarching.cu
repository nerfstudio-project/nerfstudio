#include "include/helpers_cuda.h"


inline __device__ float min_step_size(uint32_t num_steps) { 
    return __SQRT3() / num_steps; 
}

// Maximum step size is the width of the coarsest gridsize cell.
inline __device__ float max_step_size(uint32_t grid_cascades, uint32_t grid_size) { 
    // Note(ruilongli): use mip=0 step size?
    return __SQRT3() * (1 << (grid_cascades-1)) / grid_size; 
}

// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
// stepping in larger scenes. 
inline __device__ float calc_dt(float t, float cone_angle, float dt_min, float dt_max) {
    // TODO(ruilongli): scene_scale related to cone_angle?
    return __clamp(t * cone_angle, dt_min, dt_max);
}

inline __device__ int mip_from_pos(float x, float y, float z, uint32_t grid_cascades, float grid_center, float grid_scale) {
    float maxval = fmaxf(fmaxf(fabsf(x - grid_center), fabsf(y - grid_center)), fabsf(z - grid_center)) / grid_scale;
    int exponent; frexpf(maxval, &exponent);
    return min(grid_cascades-1, max(0, exponent+1));
}

inline __device__ int mip_from_dt(
    float x, float y, float z, uint32_t grid_cascades, float dt, int grid_size, float grid_center, float grid_scale
) {
    int mip = mip_from_pos(x, y, z, grid_cascades, grid_center, grid_scale);
    dt *= 2 * grid_size;
    if (dt<1.f) return mip; // exponent would be zero
    int exponent; frexpf(dt, &exponent);
    return min(grid_cascades-1, max(exponent, mip));
}

inline __device__ uint32_t grid_mip_offset(uint32_t mip, int grid_size) {
    return (grid_size * grid_size * grid_size) * mip;
}

inline __device__ uint32_t cascaded_grid_idx_at(
    float x, float y, float z, uint32_t mip, int grid_size, float grid_center, float grid_scale
) {
    // TODO(ruilongli): if the x, y, z is outside the aabb, it will be clipped into aabb!!! We should just return false
    float mip_scale = scalbnf(1.0f, -mip) / grid_scale;
    int ix = (int)((mip_scale * (x - grid_center) + 0.5f) * grid_size);
    int iy = (int)((mip_scale * (y - grid_center) + 0.5f) * grid_size);
    int iz = (int)((mip_scale * (z - grid_center) + 0.5f) * grid_size);
    // printf("[input] x %f, y %f, z %f, mip %d, grid_size %d, grid_center %f\n", x, y, z, mip, grid_size, grid_center);
    // printf("[output] mip_scale %f, ix %d iy %d, iz %d\n", mip_scale, ix, iy, iz);
    uint32_t idx = __morton3D(
        __clamp(ix, 0, grid_size-1),
        __clamp(iy, 0, grid_size-1),
        __clamp(iz, 0, grid_size-1)
    );
    return idx;
}

inline __device__ bool grid_occupied_at(
    float x, float y, float z, 
    const uint8_t* grid_bitfield,
    uint32_t mip, int grid_size, float grid_center, float grid_scale
) {
    uint32_t idx = (
        cascaded_grid_idx_at(x, y, z, mip, grid_size, grid_center, grid_scale)
        + grid_mip_offset(mip, grid_size)
    );
    return grid_bitfield[idx/8] & (1<<(idx%8));
}

inline __device__ float distance_to_next_voxel(
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    float res
) { // dda like step
    // TODO: warning: expression has no effect?
    x, y, z = res * x, res * y, res * z;
    float tx = (floorf(x + 0.5f + 0.5f * __sign(dir_x)) - x) * idir_x;
    float ty = (floorf(y + 0.5f + 0.5f * __sign(dir_y)) - y) * idir_y;
    float tz = (floorf(z + 0.5f + 0.5f * __sign(dir_z)) - z) * idir_z;
    float t = min(min(tx, ty), tz);

    return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(
    float t,
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    float res, float dt_min) {
    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(
        x, y, z, dir_x, dir_y, dir_z, idir_x, idir_y, idir_z, res
    );
    do {
        t += dt_min;
    } while (t < t_target);
    return t;
}

template <typename scalar_t>
__global__ void kernel_raymarching(
    // rays info
    const uint32_t n_rays,
    const scalar_t* rays_o,
    const scalar_t* rays_d,
    const scalar_t* t_min,
    const scalar_t* t_max, 
    // density grid
    const float grid_center,
    const float grid_scale,
    const int grid_cascades,
    const int grid_size,
    const uint8_t* grid_bitfield,
    // sampling
    const int max_total_samples,
    const int num_steps,  // default 1024
    const float cone_angle,  // default 0. for nerf-syn and 1/256 for large scene
    const float step_scale,
    int* numsteps_counter,
    int* rays_counter,
    int* packed_info_out,
    scalar_t* origins_out,
    scalar_t* dirs_out,
    scalar_t* starts_out,
    scalar_t* ends_out 
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    rays_o += i * 3;
    rays_d += i * 3;
    t_min += i;
    t_max += i;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float near = t_min[0], far = t_max[0];

    // TODO(ruilongli): perturb `near` as in ngp_pl?
    // TODO(ruilongli): pre-compute `dt_min`, `dt_max` as it is a constant?
    float dt_min = min_step_size(num_steps) * step_scale;
    float dt_max = max_step_size(grid_cascades, grid_size) * grid_scale;    
    
    // first pass to compute an accurate number of steps
    uint32_t j = 0;
    float t0 = near;
    float dt = calc_dt(t0, cone_angle, dt_min, dt_max);
    float t1 = t0 + dt;
    float t_mid = (t0 + t1) * 0.5f;

    while (t_mid < far && j < num_steps) {
        // current center
        const float x = ox + t_mid * dx;
        const float y = oy + t_mid * dy;
        const float z = oz + t_mid * dz;
        uint32_t mip = mip_from_dt(x, y, z, grid_cascades, dt, grid_size, grid_center, grid_scale);
        
        if (grid_occupied_at(x, y, z, grid_bitfield, mip, grid_size, grid_center, grid_scale)) {
            ++j;
            // march to next sample
            t0 = t1;
            dt = calc_dt(t0, cone_angle, dt_min, dt_max);
            t1 = t0 + dt;
            t_mid = (t0 + t1) * 0.5f;
        }
        else {
            // march to next sample
            float res = (grid_size >> mip) * grid_scale;
            t_mid = advance_to_next_voxel(
                t_mid, x, y, z, dx, dy, dz, rdx, rdy, rdz, res, dt_min
            );
            dt = calc_dt(t_mid, cone_angle, dt_min, dt_max);
            t0 = t_mid - dt * 0.5f;
            t1 = t_mid + dt * 0.5f;
        }
    }
    if (j == 0) return;

    uint32_t numsteps = j;
    uint32_t base = atomicAdd(numsteps_counter, numsteps);
    if (base + numsteps > max_total_samples) return;
    
    // locate
    origins_out += base * 3;
    dirs_out += base * 3;
    starts_out += base;
    ends_out += base;

    uint32_t ray_idx = atomicAdd(rays_counter, 1);

    packed_info_out[ray_idx * 3 + 0] = i;  // ray idx in {rays_o, rays_d}
    packed_info_out[ray_idx * 3 + 1] = base;  // point idx start.
    packed_info_out[ray_idx * 3 + 2] = numsteps;  // point idx shift.

    // Second round
    j = 0;
    t0 = near;
    dt = calc_dt(t0, cone_angle, dt_min, dt_max);
    t1 = t0 + dt;
    t_mid = (t0 + t1) / 2.;

    while (t_mid < far && j < num_steps) {
        // current center
        const float x = ox + t_mid * dx;
        const float y = oy + t_mid * dy;
        const float z = oz + t_mid * dz;
        uint32_t mip = mip_from_dt(x, y, z, grid_cascades, dt, grid_size, grid_center, grid_scale);
        
        if (grid_occupied_at(x, y, z, grid_bitfield, mip, grid_size, grid_center, grid_scale)) {
            origins_out[j * 3 + 0] = ox;
            origins_out[j * 3 + 1] = oy;
            origins_out[j * 3 + 2] = oz;
            dirs_out[j * 3 + 0] = dx;
            dirs_out[j * 3 + 1] = dy;
            dirs_out[j * 3 + 2] = dz;
            starts_out[j] = t0;   
            ends_out[j] = t1;     
            ++j;
            // march to next sample
            t0 = t1;
            dt = calc_dt(t0, cone_angle, dt_min, dt_max);
            t1 = t0 + dt;
            t_mid = (t0 + t1) * 0.5f;
        }
        else {
            // march to next sample
            float res = (grid_size >> mip) * grid_scale;
            t_mid = advance_to_next_voxel(
                t_mid, x, y, z, dx, dy, dz, rdx, rdy, rdz, res, dt_min
            );
            dt = calc_dt(t_mid, cone_angle, dt_min, dt_max);
            t0 = t_mid - dt * 0.5f;
            t1 = t_mid + dt * 0.5f;
        }
    }
    return;
}


template <typename scalar_t>
__global__ void kernel_occupancy_query(
    // samples info
    const uint32_t n_samples,
    const scalar_t* positions, 
    const scalar_t* deltas, 
    // density grid
    const float grid_center,
    const float grid_scale,
    const int grid_cascades,
    const int grid_size,
    const uint8_t* grid_bitfield,
    // outputs
    int32_t* mip_levels, // output mip level
    int32_t* indices,  // output indices
    bool* occupancies  // output occupancy
) {
    CUDA_GET_THREAD_ID(i, n_samples);

    // locate
    positions += i * 3;
    deltas += i;
    mip_levels += i;
    indices += i;
    occupancies += i;

    // current point
    const float x = positions[0];
    const float y = positions[1];
    const float z = positions[2];
    const float dt = deltas[0];
    
    int32_t mip = mip_from_dt(x, y, z, grid_cascades, dt, grid_size, grid_center, grid_scale);
    mip_levels[0] = mip;
    
    int32_t idx = (
        cascaded_grid_idx_at(x, y, z, mip, grid_size, grid_center, grid_scale)
        + grid_mip_offset(mip, grid_size)
    );
    indices[0] = idx;

    if (grid_bitfield) {
        bool occ = grid_bitfield[idx/8] & (1<<(idx%8));
        occupancies[0] = occ;
    }
    return;
}


/**
 * @brief Sample points by ray marching.
 * 
 * @param rays_o Ray origins Shape of [n_rays, 3].
 * @param rays_d Normalized ray directions. Shape of [n_rays, 3].
 * @param t_min Near planes of rays. Shape of [n_rays].
 * @param t_max Far planes of rays. Shape of [n_rays].
 * @param grid_center Density grid center. TODO: support 3-dims.
 * @param grid_scale Density grid base level scale. TODO: support 3-dims.
 * @param grid_cascades Density grid levels.
 * @param grid_size Density grid resolution.
 * @param grid_bitfield Density grid uint8 bit field.
 * @param max_total_samples Maximum total number of samples in this batch.
 * @param num_steps Used to define the minimal step size: SQRT3() / num_steps.
 * @param cone_angle 0. for nerf-synthetic and 1./256 for real scenes.
 * @param step_scale Scale up the step size by this much. Usually equals to scene scale.
 * @return std::vector<torch::Tensor> 
 * - packed_info: Stores how to index the ray samples from the returned values.
 *  Shape of [n_rays, 3]. First value is the ray index. Second value is the sample 
 *  start index in the results for this ray. Third value is the number of samples for
 *  this ray. Note for rays that have zero samples, we simply skip them so the `packed_info`
 *  has some zero padding in the end.
 * - origins: Ray origins for those samples. [max_total_samples, 3]
 * - dirs: Ray directions for those samples. [max_total_samples, 3]
 * - starts: Where the frustum-shape sample starts along a ray. [max_total_samples, 1]
 * - ends: Where the frustum-shape sample ends along a ray. [max_total_samples, 1]
 */
std::vector<torch::Tensor> raymarching(
    // rays
    const torch::Tensor rays_o, 
    const torch::Tensor rays_d, 
    const torch::Tensor t_min, 
    const torch::Tensor t_max,
    // density grid
    const float grid_center,
    const float grid_scale,
    const int grid_cascades,
    const int grid_size,
    const torch::Tensor grid_bitfield, 
    // sampling args
    const int max_total_samples,
    const int num_steps,
    const float cone_angle,
    const float step_scale
) {
    DEVICE_GUARD(rays_o);

    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(t_min);
    CHECK_INPUT(t_max);
    CHECK_INPUT(grid_bitfield);
    
    const int n_rays = rays_o.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // helper counter
    torch::Tensor numsteps_counter = torch::zeros(
        {1}, rays_o.options().dtype(torch::kInt32));
    torch::Tensor rays_counter = torch::zeros(
        {1}, rays_o.options().dtype(torch::kInt32));

    // output samples
    torch::Tensor packed_info = torch::zeros(
        {n_rays, 3}, rays_o.options().dtype(torch::kInt32));  // ray_id, sample_id, num_samples
    torch::Tensor origins = torch::zeros({max_total_samples, 3}, rays_o.options());
    torch::Tensor dirs = torch::zeros({max_total_samples, 3}, rays_o.options());
    torch::Tensor starts = torch::zeros({max_total_samples, 1}, rays_o.options());
    torch::Tensor ends = torch::zeros({max_total_samples, 1}, rays_o.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        rays_o.scalar_type(),
        "raymarching_train",
        ([&]
         { kernel_raymarching<scalar_t><<<blocks, threads>>>(
                // rays
                n_rays,
                rays_o.data_ptr<scalar_t>(),
                rays_d.data_ptr<scalar_t>(),
                t_min.data_ptr<scalar_t>(),
                t_max.data_ptr<scalar_t>(),
                // density grid
                grid_center,
                grid_scale,
                grid_cascades,
                grid_size,
                grid_bitfield.data_ptr<uint8_t>(),
                // sampling args
                max_total_samples,
                num_steps,
                cone_angle,
                step_scale,
                numsteps_counter.data_ptr<int>(),  // total samples.
                rays_counter.data_ptr<int>(),  // total rays.
                packed_info.data_ptr<int>(), 
                origins.data_ptr<scalar_t>(),
                dirs.data_ptr<scalar_t>(), 
                starts.data_ptr<scalar_t>(),
                ends.data_ptr<scalar_t>()
            ); 
        }));

    return {packed_info, origins, dirs, starts, ends};
}


std::vector<torch::Tensor> occupancy_query(
    // samples
    const torch::Tensor positions, 
    const torch::Tensor deltas, 
    // density grid
    const float grid_center,
    const float grid_scale,
    const int grid_cascades,
    const int grid_size,
    const torch::Tensor grid_bitfield
) {
    DEVICE_GUARD(positions);

    CHECK_INPUT(positions);
    CHECK_INPUT(deltas);
    CHECK_INPUT(grid_bitfield);
    
    const int n_samples = positions.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_samples, threads);

    // outputs
    torch::Tensor mip_levels = torch::empty({n_samples}, positions.options().dtype(torch::kInt32));
    torch::Tensor indices = torch::empty({n_samples}, positions.options().dtype(torch::kInt32));
    torch::Tensor occupancies = torch::empty({n_samples}, positions.options().dtype(torch::kBool));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        positions.scalar_type(),
        "occupancy_query",
        ([&]
         { kernel_occupancy_query<scalar_t><<<blocks, threads>>>(
                // samples
                n_samples,
                positions.data_ptr<scalar_t>(),
                deltas.data_ptr<scalar_t>(),
                // density grid
                grid_center,
                grid_scale,
                grid_cascades,
                grid_size,
                grid_bitfield.data_ptr<uint8_t>(),
                // outputs
                mip_levels.data_ptr<int32_t>(),
                indices.data_ptr<int32_t>(),
                occupancies.data_ptr<bool>()
            ); 
        }));

    return {mip_levels, indices, occupancies};
}

