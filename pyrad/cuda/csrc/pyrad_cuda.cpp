/*
This file uses pybind to make CUDA calls.
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include "include/structures.cuh"
#include "include/helpers.cuh"

std::vector<torch::Tensor> sample_uniformly_along_ray_bundle(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor nears,
    torch::Tensor fars,
    torch::Tensor offsets,
    int max_num_samples);

RaySamples generate_ray_samples_uniform(
    RayBundle &ray_bundle, int num_samples, DensityGrid &grid
);

torch::Tensor grid_sample(torch::Tensor positions, DensityGrid &grid);

torch::Tensor unpack(
    torch::Tensor packed_data,  // ["num_elements", D]
    torch::Tensor packed_info,  // ["num_packs", N + 1]
    at::IntArrayRef output_size  // [C_1, C_2, ..., C_N, D]
);

torch::Tensor pack(
    torch::Tensor mask  // [C_1, C_2, ..., C_N]
);

std::vector<torch::Tensor> pack_single_tensor(
    torch::Tensor data,  // [C_1, C_2, ..., C_N, D]
    torch::Tensor mask  // [C_1, C_2, ..., C_N]
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<RayBundle>(m, "RayBundle")
        .def(py::init<>())
        .def_readwrite("origins", &RayBundle::origins)
        .def_readwrite("directions", &RayBundle::directions)
        .def_readwrite("pixel_area", &RayBundle::pixel_area)
        .def_readwrite("camera_indices", &RayBundle::camera_indices)
        .def_readwrite("nears", &RayBundle::nears)
        .def_readwrite("fars", &RayBundle::fars)
        .def_readwrite("valid_mask", &RayBundle::valid_mask)
        .def_readwrite("num_rays_per_chunk", &RayBundle::num_rays_per_chunk);

    py::class_<Frustums>(m, "Frustums")
        .def(py::init<>())
        .def_readwrite("origins", &Frustums::origins)
        .def_readwrite("directions", &Frustums::directions)
        .def_readwrite("starts", &Frustums::starts)
        .def_readwrite("ends", &Frustums::ends)
        .def_readwrite("pixel_area", &Frustums::pixel_area);

    py::class_<RaySamples>(m, "RaySamples")
        .def(py::init<>())
        .def_readwrite("frustums", &RaySamples::frustums)
        .def_readwrite("packed_indices", &RaySamples::packed_indices)
        .def_readwrite("camera_indices", &RaySamples::camera_indices)
        .def_readwrite("deltas", &RaySamples::deltas);

    py::class_<DensityGrid>(m, "DensityGrid")
        .def(py::init<>())
        .def_readwrite("num_cascades", &DensityGrid::num_cascades)
        .def_readwrite("resolution", &DensityGrid::resolution)
        .def_readwrite("aabb", &DensityGrid::aabb)
        .def_readwrite("data", &DensityGrid::data);

    m.def("sample_uniformly_along_ray_bundle", &sample_uniformly_along_ray_bundle);
    m.def("generate_ray_samples_uniform", &generate_ray_samples_uniform);
    m.def("grid_sample", &grid_sample);
    m.def("unpack", &unpack);
    m.def("pack", &pack);
    m.def("pack_single_tensor", &pack_single_tensor);
    // m.def("grid_sampler_3d_cuda", &at::native::grid_sampler_3d_cuda);
}