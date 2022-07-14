#pragma once

#include "helpers.cuh"

template <typename scalar_t>
inline __host__ __device__ void grid_sample_3d(
    const scalar_t x, const scalar_t y, const scalar_t z, // valued in [0, 1]
    const scalar_t* grid_data, // [x, y, z, c]
    const int x_res, const int y_res, const int z_res, const int c_res,
    // const bool align_corners,  TODO(ruilongli): only support align_corners=True
    scalar_t* out
) {
    // initialize output to zero
    #pragma unroll
    for (int feature = 0; feature < c_res; ++feature) {
        out[feature] = 0;
    }
    // skip the samples outside the grid
    if (x < 0 || x > 1 || y < 0 || y > 1 || z < 0 || z > 1) {
        return;
    }
    
    scalar_t positions[3] = {x, y, z};
    int resolutions[3] = {x_res, y_res, z_res};

    scalar_t pos[3];
	int pos_grid[3];
    #pragma unroll
    for (int dim = 0; dim < 3; ++dim) {
        scalar_t tmp = positions[dim] * (resolutions[dim] - 1);
        pos_grid[dim] = floorf(tmp);
        pos[dim] = tmp - pos_grid[dim];
    }

    // trilinear interp
    #pragma unroll
    for (int idx = 0; idx < (1 << 3); ++idx) {
        scalar_t weight = 1;
        int pos_grid_local[3];

        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            if ((idx & (1 << dim)) == 0) {
                weight *= 1 - pos[dim];
                pos_grid_local[dim] = pos_grid[dim];
            } else {
                weight *= pos[dim];
                pos_grid_local[dim] = pos_grid[dim] + 1;
            }
        }

        const scalar_t* val = (
            grid_data 
            + pos_grid_local[0] * y_res * z_res * c_res
            + pos_grid_local[1] * z_res * c_res
            + pos_grid_local[2] * c_res
        );
        
        #pragma unroll
        for (int feature = 0; feature < c_res; ++feature) {
            out[feature] += weight * val[feature];
        }
    }
	return;
}