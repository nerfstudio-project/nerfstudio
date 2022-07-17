#include "include/helper_cuda.h"


torch::Tensor packbits(
    const torch::Tensor data, const float threshold
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packbits", &packbits);
}