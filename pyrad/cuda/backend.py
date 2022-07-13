# Copyright 2022 The Plenoptix Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from torch.utils.cpp_extension import load

PATH = os.path.dirname(os.path.abspath(__file__))

extra_cflags = ["-O3", "-std=c++14"]
extra_cuda_cflags = [
    "-O3",
    "-std=c++14",
    # Use the internal PyTorch half operations instead of the ones from the CUDA libraries.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]
sources = [
    os.path.join(PATH, "csrc", "pyrad_cuda.cpp"),
    os.path.join(PATH, "csrc", "pyrad_cuda_kernel.cu"),
]

_C = load(name="pyrad_cuda", sources=sources, extra_cflags=extra_cflags, extra_cuda_cflags=extra_cuda_cflags)

__all__ = ["_C"]
