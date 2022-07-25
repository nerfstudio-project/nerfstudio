#include "include/helpers.h"


torch::Tensor packbits(
    const torch::Tensor data, const float threshold
) {};

std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb
) {};

torch::Tensor morton3D(const torch::Tensor coords) {};

torch::Tensor morton3D_invert(const torch::Tensor indices) {};

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
) {};

std::vector<torch::Tensor> volumetric_rendering_forward(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas, 
    torch::Tensor rgbs
) {};

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
) {};

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
) {};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packbits", &packbits);
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("morton3D", &morton3D);
    m.def("morton3D_invert", &morton3D_invert);
    m.def("raymarching", &raymarching);
    m.def("volumetric_rendering_forward", &volumetric_rendering_forward);
    m.def("volumetric_rendering_backward", &volumetric_rendering_backward);
    m.def("occupancy_query", &occupancy_query);
}