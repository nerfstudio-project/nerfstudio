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
import glob
import os

from torch.utils.cpp_extension import load

PATH = os.path.dirname(os.path.abspath(__file__))

extra_cflags = ["-O2"]
extra_cuda_cflags = ["-O2"]
extra_include_paths = [os.path.join(PATH, "csrc/include")]
sources = glob.glob(os.path.join(PATH, "csrc/*.cpp")) + glob.glob(os.path.join(PATH, "csrc/*.cu"))

_C = load(
    name="pyrad_cuda_pl",
    sources=sources,
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    extra_include_paths=extra_include_paths,
)

__all__ = ["_C"]
