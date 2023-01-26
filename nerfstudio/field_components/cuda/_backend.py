# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""compiling the cuda kernels"""
import glob
import os
import shutil

from nerfacc.cuda._backend import cuda_toolkit_available
from torch.utils.cpp_extension import _get_build_directory, load

PATH = os.path.dirname(os.path.abspath(__file__))
NAME = "nerfstudio_field_components_cuda"
BUILD_DIR = _get_build_directory(NAME, verbose=False)


_C = None
if cuda_toolkit_available():
    if os.path.exists(os.path.join(BUILD_DIR, f"{NAME}.so")):
        # If the build exists, we assume the extension has been built
        # and we can load it.
        _C = load(
            name=NAME,
            sources=glob.glob(os.path.join(PATH, "csrc/*.cu")),
            extra_cflags=["-O3", "-std=c++14"],
            extra_cuda_cflags=["-O3", "-std=c++14"],
            extra_include_paths=[],
        )
    else:
        # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
        # if the build directory exists.
        shutil.rmtree(BUILD_DIR)
        print("nerfstudio field components: Setting up CUDA (This may take a few minutes the first time)")
        _C = load(
            name=NAME,
            sources=glob.glob(os.path.join(PATH, "csrc/*.cu")),
            extra_cflags=["-O3", "-std=c++14"],
            extra_cuda_cflags=["-O3", "-std=c++14"],
            extra_include_paths=[],
        )
        print("nerfstudio field components: Setting up CUDA finished")
else:
    print("nerfstudio field components: No CUDA toolkit found. Some models may fail.")


__all__ = ["_C"]
