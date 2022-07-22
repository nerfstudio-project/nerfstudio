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
from subprocess import DEVNULL, call

from torch.utils.cpp_extension import load

PATH = os.path.dirname(os.path.abspath(__file__))


def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    # https://github.com/idiap/fast-transformers/blob/master/setup.py
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


if cuda_toolkit_available():
    sources = glob.glob(os.path.join(PATH, "csrc/*.cu"))
else:
    sources = glob.glob(os.path.join(PATH, "csrc/*.cpp"))

extra_cflags = ["-O2"]
extra_cuda_cflags = ["-O2"]
_C = load(
    name="pyrad_extension",
    sources=sources,
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
)

__all__ = ["_C"]
