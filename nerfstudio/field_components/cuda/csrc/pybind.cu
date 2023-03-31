/*
 * Code adapted from (@ashawkey) https://github.com/ashawkey/torch-ngp/
 * Author: @lsongx
 */


#include <torch/extension.h>

#include "include/temporal_gridencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("temporal_grid_encode_forward", &temporal_grid_encode_forward, "temporal_grid_encode_forward (CUDA)");
    m.def("temporal_grid_encode_backward", &temporal_grid_encode_backward, "temporal_grid_encode_backward (CUDA)");
}