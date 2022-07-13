#pragma once

#include "helpers.cuh"


struct RayBundle {
    torch::Tensor origins;
    torch::Tensor directions;
    torch::Tensor pixel_area;
    torch::Tensor camera_indices;
    torch::Tensor nears;
    torch::Tensor fars;
    torch::Tensor valid_mask;
    int num_rays_per_chunk;

    inline void check() {
        CHECK_INPUT(origins);
        CHECK_INPUT(directions);
        CHECK_INPUT(pixel_area);
        if (camera_indices.defined()) CHECK_INPUT(camera_indices);
        if (nears.defined()) CHECK_INPUT(nears);
        if (fars.defined()) CHECK_INPUT(fars);
        if (valid_mask.defined()) CHECK_INPUT(valid_mask);
    }
};


struct Frustums {
    torch::Tensor origins;
    torch::Tensor directions;
    torch::Tensor starts;
    torch::Tensor ends;
    torch::Tensor pixel_area;

    inline void check() {
        CHECK_INPUT(origins);
        CHECK_INPUT(directions);
        CHECK_INPUT(starts);
        CHECK_INPUT(ends);
        CHECK_INPUT(pixel_area);
    }
};


struct RaySamples {
    Frustums frustums;
    torch::Tensor packed_indices;
    torch::Tensor camera_indices;
    torch::Tensor deltas;

    inline void check() {
        frustums.check();
        CHECK_INPUT(packed_indices);
        if (camera_indices.defined()) CHECK_INPUT(camera_indices);
        if (deltas.defined()) CHECK_INPUT(deltas);
    }
};

struct DensityGrid {
    uint32_t num_cascades;
    uint32_t resolution;
    torch::Tensor aabb;
    torch::Tensor data;  // [res, res, res, cascades]
    inline void check() {
        CHECK_INPUT(aabb);
        CHECK_INPUT(data);
    }
};
