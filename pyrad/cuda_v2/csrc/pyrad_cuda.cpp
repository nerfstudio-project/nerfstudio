#include "include/helpers.h"


torch::Tensor packbits(
    const torch::Tensor data, const float threshold
);

std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb
);

torch::Tensor morton3D(const torch::Tensor coords);

torch::Tensor morton3D_invert(const torch::Tensor indices);

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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packbits", &packbits);
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("morton3D", &morton3D);
    m.def("morton3D_invert", &morton3D_invert);
    m.def("raymarching_train", &raymarching_train);
}