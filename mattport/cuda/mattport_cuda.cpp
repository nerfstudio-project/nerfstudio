/*
This file uses pybind to make CUDA calls.
*/

#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> sample_uniformly_along_ray_bundle(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor nears,
    torch::Tensor fars,
    torch::Tensor offsets,
    int max_num_samples);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sample_uniformly_along_ray_bundle", &sample_uniformly_along_ray_bundle);
}