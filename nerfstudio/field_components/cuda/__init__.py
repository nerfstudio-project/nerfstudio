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

"""init cuda functions"""
from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    """_make_lazy_cuda_func from nerfacc.cuda"""

    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


temporal_grid_encode_forward = _make_lazy_cuda_func("temporal_grid_encode_forward")
temporal_grid_encode_backward = _make_lazy_cuda_func("temporal_grid_encode_backward")
