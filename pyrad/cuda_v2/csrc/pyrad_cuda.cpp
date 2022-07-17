#include "include/helper_cuda.h"


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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packbits", &packbits);
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("morton3D", &morton3D);
    m.def("morton3D_invert", &morton3D_invert);
}