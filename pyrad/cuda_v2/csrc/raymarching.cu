#include "include/helpers.h"


inline __device__ float min_step_size(uint32_t num_steps) { 
    return __SQRT3() / num_steps; 
}

// Maximum step size is the width of the coarsest gridsize cell.
inline __device__ float max_step_size(uint32_t num_steps, uint32_t cascades, uint32_t grid_size) { 
    return __SQRT3() * (1 << (cascades-1)) / grid_size; 
}

// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
// stepping in larger scenes. 
inline __device__ float calc_dt(float t, float cone_angle, float dt_min, float dt_max) {
	return __clamp(t * cone_angle, dt_min, dt_max);
}

inline __device__ int mip_from_pos(float x, float y, float z, uint32_t cascades) {
    float maxval = fmaxf(fmaxf(fabsf(x - 0.5f), fabsf(y - 0.5f)), fabsf(z - 0.5f));
	int exponent; frexpf(maxval, &exponent);
	return min(cascades-1, max(0, exponent+1));
}

inline __device__ int mip_from_dt(
    float x, float y, float z, uint32_t cascades, float dt, int grid_size
) {
	int mip = mip_from_pos(x, y, z, cascades);
	dt *= 2 * grid_size;
	if (dt<1.f) return mip; // exponent would be zero
	int exponent; frexpf(dt, &exponent);
	return min(cascades-1, max(exponent, mip));
}

inline __device__ uint32_t grid_mip_offset(uint32_t mip, int grid_size) {
	return (grid_size * grid_size * grid_size) * mip;
}

inline __device__ uint32_t cascaded_grid_idx_at(
    float x, float y, float z, uint32_t mip, int grid_size
) {
	float mip_scale = scalbnf(1.0f, -mip);
    int ix = (int)((mip_scale * (x - 0.5f) + 0.5f) * grid_size);
    int iy = (int)((mip_scale * (y - 0.5f) + 0.5f) * grid_size);
    int iz = (int)((mip_scale * (z - 0.5f) + 0.5f) * grid_size);
    // printf("mip %d, mip_scale %f, ix %d iy %d\n", mip, mip_scale, ix, iy);
	uint32_t idx = __morton3D(
		__clamp(ix, 0, grid_size-1),
		__clamp(iy, 0, grid_size-1),
		__clamp(iz, 0, grid_size-1)
	);
	return idx;
}

inline __device__ bool density_grid_occupied_at(
    float x, float y, float z, 
    const uint8_t* density_grid_bitfield,
    uint32_t mip, int grid_size
) {
	uint32_t idx = (
        cascaded_grid_idx_at(x, y, z, mip, grid_size)
        + grid_mip_offset(mip, grid_size)
    );
	return density_grid_bitfield[idx/8] & (1<<(idx%8));
}

inline __device__ float distance_to_next_voxel(
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    uint32_t res
) { // dda like step
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
    uint32_t res, float dt_min) {
	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + distance_to_next_voxel(
        x, y, z, dir_x, dir_y, dir_z, idir_x, idir_y, idir_z, res
    );
    // printf("advance_to_next_voxel dt_min %f t_target %f \n", dt_min, t_target);
	do {
		t += dt_min;
	} while (t < t_target);
	return t;
}

// The scene should be bounded in [0, 1].
template <typename scalar_t>
__global__ void kernel_raymarching_train(
    // rays info
    const uint32_t n_rays,
    const scalar_t* rays_o,
    const scalar_t* rays_d,
    const scalar_t* t_min,
    const scalar_t* t_max, 
    // density grid
    const int cascades,
    const int grid_size,
    const uint8_t* density_bitfield,
    // sampling
    const float cone_angle,  // default 0. for nerf-syn and 1/256 for large scene
    const int num_steps,  // default 1024
    const int max_samples,
    int* numsteps_counter,
    int* rays_counter,  // total rays.
    int* indices_out,  // output ray & point indices.
    scalar_t* positions_out,  // output samples
    scalar_t* dirs_out,  // output dirs
    scalar_t* deltas_out,  // output delta t
    scalar_t* ts_out  // output t
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
    const float startt = t_min[0], endt = t_max[0];

    // first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;

    // TODO(ruilongli): perturb `startt` as in ngp_pl?
    // TODO(ruilongli): pre-compute `dt_min`, `dt_max` as it is a constant?
    float dt_min = min_step_size(num_steps);
    float dt_max = max_step_size(num_steps, cascades, grid_size);    

	while (0 <= t && t < endt && j < num_steps) {
        // current point
        const float x = ox + t * dx;
        const float y = oy + t * dy;
        const float z = oz + t * dz;
                
        float dt = calc_dt(t, cone_angle, dt_min, dt_max);
		uint32_t mip = mip_from_dt(x, y, z, cascades, dt, grid_size);
        // printf("t %f \n", t);

        if (density_grid_occupied_at(x, y, z, density_bitfield, mip, grid_size)) {
            ++j;
			t += dt;
		}
        else {
			uint32_t res = grid_size >> mip;
			t = advance_to_next_voxel(
                t, x, y, z, dx, dy, dz, rdx, rdy, rdz, res, dt_min
            );
		}
	}
    if (j == 0) return;

    uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);
    if (base + numsteps > max_samples) return;
	
    // locate
    positions_out += base * 3;
    dirs_out += base * 3;
    deltas_out += base;

    uint32_t ray_idx = atomicAdd(rays_counter, 1);

	indices_out[ray_idx * 3 + 0] = i;  // ray idx in {rays_o, rays_d}
    indices_out[ray_idx * 3 + 1] = base;  // point idx start.
    indices_out[ray_idx * 3 + 2] = numsteps;  // point idx shift.

	t = startt;
	j = 0;
    while (t < endt && j < num_steps) {
        // current point
        const float x = ox + t * dx;
        const float y = oy + t * dy;
        const float z = oz + t * dz;

        float dt = calc_dt(t, cone_angle, dt_min, dt_max);
		uint32_t mip = mip_from_dt(x, y, z, cascades, dt, grid_size);

        if (density_grid_occupied_at(x, y, z, density_bitfield, mip, grid_size)) {
            positions_out[j * 3 + 0] = x;
            positions_out[j * 3 + 1] = y;
            positions_out[j * 3 + 2] = z;
            dirs_out[j * 3 + 0] = dx,
            dirs_out[j * 3 + 1] = dy,
            dirs_out[j * 3 + 2] = dz,
            deltas_out[j] = dt;   
            ts_out[j] = t;         
            ++j;
			t += dt;
		}
        else {
			uint32_t res = grid_size >> mip;
			t = advance_to_next_voxel(
                t, x, y, z, dx, dy, dz, rdx, rdy, rdz, res, dt_min
            );
		}
	}

    return;
}

/**
 * @brief Sample points by ray marching during training.
 * 
 * @param rays_o Shape of [n_rays, 3]
 * @param rays_d Shape of [n_rays, 3]
 * @param t_min Shape of [n_rays]
 * @param t_max Shape of [n_rays]
 * @param cascades 
 * @param grid_size
 * @param density_bitfield Shape of [cascades * grid_size**3 // 8]
 * @param max_samples Default 1024
 * @param cone_angle Default 0 for nerf-syn and 1/256 for large scene
 * @return std::vector<torch::Tensor> 
 */
std::vector<torch::Tensor> raymarching_train(
    const torch::Tensor rays_o, 
    const torch::Tensor rays_d, 
    const torch::Tensor t_min, 
    const torch::Tensor t_max,
    const int cascades,
    const int grid_size,
    const torch::Tensor density_bitfield, 
    const int max_samples,
    const int num_steps,
    const float cone_angle
) {
    DEVICE_GUARD(rays_o);
    const int n_rays = rays_o.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // helper counter
    torch::Tensor numsteps_counter = torch::zeros(
        {1}, rays_o.options().dtype(torch::kInt32));
    torch::Tensor rays_counter = torch::zeros(
        {1}, rays_o.options().dtype(torch::kInt32));

    // output samples
    torch::Tensor indices = torch::zeros(
        {n_rays, 3}, rays_o.options().dtype(torch::kInt32));  // ray_id, sample_id, num_samples
    torch::Tensor positions = torch::empty({max_samples, 3}, rays_o.options());
    torch::Tensor dirs = torch::empty({max_samples, 3}, rays_o.options());
    torch::Tensor deltas = torch::empty({max_samples}, rays_o.options());
    torch::Tensor ts = torch::empty({max_samples}, rays_o.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        rays_o.scalar_type(),
        "raymarching_train",
        ([&]
         { kernel_raymarching_train<scalar_t><<<blocks, threads>>>(
                // rays info
                n_rays,
                rays_o.data_ptr<scalar_t>(),
                rays_d.data_ptr<scalar_t>(),
                t_min.data_ptr<scalar_t>(),
                t_max.data_ptr<scalar_t>(),
                // density grid
                cascades,
                grid_size,
                density_bitfield.data_ptr<uint8_t>(),
                // sampling
                cone_angle,
                num_steps,
                max_samples,
                numsteps_counter.data_ptr<int>(),  // total samples.
                rays_counter.data_ptr<int>(),  // total rays.
                indices.data_ptr<int>(),  // output ray indices.
                positions.data_ptr<scalar_t>(),  // output samples
                dirs.data_ptr<scalar_t>(),  // output dirs
                deltas.data_ptr<scalar_t>(),  // output delta t
                ts.data_ptr<scalar_t>()  // output t
            ); 
        }));

    return {indices, positions, dirs, deltas, ts};
}